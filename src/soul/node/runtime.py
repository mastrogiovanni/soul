"""Actual transformer stage execution on CPU, CUDA or Metal. No generation mocks."""

import copy
import gc
import math
import re
import time
from pathlib import Path

import psutil
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from soul.catalog import ModelSpec
from soul.node.quantized import PackedLinear
from soul.node.weights import WeightStore
from soul.protocol import ARCHITECTURES, SoulError
from soul.quantization import QUANTIZATIONS, quantized_modules, validate_config

WIRE_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def encode_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in WIRE_DTYPES:
        raise ValueError("Unsupported activation dtype")
    return {"dtype": dtype, "shape": list(tensor.shape), "data": tensor.view(torch.uint8).numpy().tobytes()}


def decode_tensor(value, device, dtype, hidden_size, max_tokens):
    shape = value["shape"]
    if (
        len(shape) != 3
        or shape[0] != 1
        or shape[2] != hidden_size
        or not isinstance(shape[1], int)
        or not 1 <= shape[1] <= max_tokens
    ):
        raise ValueError("Invalid activation shape")
    wire_dtype = WIRE_DTYPES[value["dtype"]]
    if len(value["data"]) != math.prod(shape) * torch.empty((), dtype=wire_dtype).element_size():
        raise ValueError("Activation byte count does not match shape")
    return torch.frombuffer(bytearray(value["data"]), dtype=wire_dtype).reshape(shape).to(device, dtype)


def select_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Runtime:
    def __init__(self, cache_dir: Path, device="auto", memory_gb=None, weight_mode="resident", dtype="auto"):
        self.device = select_device(device)
        self.dtype = (
            (torch.float32 if self.device == "cpu" else torch.float16)
            if dtype == "auto"
            else WIRE_DTYPES[dtype]
        )
        self.cache_dir = cache_dir
        self.weight_mode = weight_mode
        if self.device.startswith("cuda"):
            with torch.cuda.device(self.device):
                available = torch.cuda.mem_get_info()[0]
            self.device_name = torch.cuda.get_device_name(self.device)
        elif self.device == "mps":
            available = min(psutil.virtual_memory().available, torch.mps.recommended_max_memory())
            self.device_name = "Apple Metal (unified memory)"
        else:
            available = psutil.virtual_memory().available
            self.device_name = "CPU"
        self.budget = int(available * 0.75)
        if memory_gb is not None:
            self.budget = min(self.budget, int(memory_gb * 1024**3))
        self.model = self.spec = self.tokenizer = self.store = None
        self.sessions = {}
        self.start = self.end = 0
        self.epoch = None
        self.cached = {}

    def scan_cache(self):
        self.cached = {}
        for spec in WeightStore.manifests(self.cache_dir):
            self.refresh_cache(spec)
        return self.inventory()

    def refresh_cache(self, spec):
        store = WeightStore(self.cache_dir, spec)
        tokenizer = False
        if (store.root / "tokenizer").is_dir():
            try:
                AutoTokenizer.from_pretrained(store.root / "tokenizer", local_files_only=True)
                tokenizer = True
            except (OSError, ValueError):
                pass
        self.cached[spec.key] = {
            "spec": spec.to_dict(),
            "tensors": store.available(),
            "tokenizer": tokenizer,
        }

    def inventory(self):
        return [
            {**entry, "offer": self.offer(ModelSpec(**entry["spec"]))} for entry in list(self.cached.values())
        ]

    def preload(self, spec, start, end, progress=lambda *_: None):
        if not 0 <= start < end <= spec.layers:
            raise ValueError("Invalid layer assignment")
        store = WeightStore(self.cache_dir, spec)
        try:
            store.ensure(spec.names_for(start, end), progress)
            if start == 0:
                target = store.root / "tokenizer"
                try:
                    AutoTokenizer.from_pretrained(target, local_files_only=True)
                except (OSError, ValueError):
                    tokenizer = AutoTokenizer.from_pretrained(
                        spec.model_id, revision=spec.revision, trust_remote_code=False
                    )
                    tokenizer.save_pretrained(target)
        finally:
            self.refresh_cache(spec)

    def capabilities(self):
        return {
            "device": self.device,
            "device_name": self.device_name,
            "budget_bytes": self.budget,
            "dtype": str(self.dtype).removeprefix("torch."),
            "architectures": list(ARCHITECTURES),
            "quantizations": list(QUANTIZATIONS),
            "weight_mode": self.weight_mode,
            "ram_bytes": psutil.virtual_memory().total,
        }

    def offer(self, spec: ModelSpec):
        validate_config(spec.config)
        if spec.key not in self.cached:
            # Adopts tensor caches written by protocol-v1 clients as well.
            WeightStore(self.cache_dir, spec).remember()
            self.refresh_cache(spec)
        entry = self.cached[spec.key]
        entry["spec"] = spec.to_dict()
        cached = set(entry["tensors"])
        element = torch.empty((), dtype=self.dtype).element_size()
        layer_bytes = [
            sum(
                spec.tensor_bytes(n, element)
                for n, t in spec.tensors.items()
                if n.startswith(f"model.layers.{i}.")
            )
            for i in range(spec.layers)
        ]
        first = spec.weight_bytes(0, 1, element) - layer_bytes[0]
        last = spec.weight_bytes(spec.layers - 1, spec.layers, element) - layer_bytes[-1]
        # Bound workspace for chunked prefill, attention scores, MLP temporaries, and allocator slack.
        chunk = min(128, spec.max_context)
        workspace = max(
            256 * 1024**2,
            8 * chunk * spec.config.get("intermediate_size", 4 * spec.config["hidden_size"]) * element
            + spec.config["num_attention_heads"] * chunk * spec.max_context * 4,
        )
        workspace += spec.quantization_workspace()
        costs, missing_bytes = [], []
        for start in range(spec.layers):
            row, missing = [], []
            for end in range(start + 1, spec.layers + 1):
                weights = (
                    sum(layer_bytes[start:end])
                    if self.weight_mode == "resident"
                    else max(layer_bytes[start:end])
                )
                weights += (first if start == 0 else 0) + (last if end == spec.layers else 0)
                row.append(math.ceil(weights * 1.1 + (end - start) * spec.kv_per_layer(element) + workspace))
                missing.append(
                    sum(spec.tensors[n]["length"] for n in spec.names_for(start, end) if n not in cached)
                    + (1 if start == 0 and not entry["tokenizer"] else 0)
                )
            costs.append(row)
            missing_bytes.append(missing)
        return {
            "budget_bytes": self.budget,
            "costs": costs,
            "weight_mode": self.weight_mode,
            "missing_bytes": missing_bytes,
            "disk_free_bytes": psutil.disk_usage(self.cache_dir).free,
            "context": spec.max_context,
        }

    def unload(self):
        self.sessions.clear()
        self.model = self.spec = self.tokenizer = self.store = None
        self.epoch = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def load(
        self, spec: ModelSpec, start: int, end: int, epoch: str, progress=lambda *_: None, *, download=True
    ):
        if spec.config["model_type"] not in ARCHITECTURES or not 0 <= start < end <= spec.layers:
            raise ValueError("Unsupported assignment")
        if self.offer(spec)["costs"][start][end - start - 1] > self.budget:
            raise SoulError("Assignment exceeds this node's memory budget")
        self.unload()
        self.spec, self.start, self.end = spec, start, end
        self.store = WeightStore(self.cache_dir, spec)
        if download:
            self.preload(spec, start, end, progress)
        else:
            self.store.require(spec.names_for(start, end))
        data = dict(spec.config)
        model_type = data.pop("model_type")
        data.pop("quantization_config", None)
        config = AutoConfig.for_model(model_type, **data)
        config._attn_implementation = "sdpa"
        # Build only shape metadata before discarding every unassigned parameter.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        model.model.layers = nn.ModuleList(list(model.model.layers)[start:end])
        local_config = copy.deepcopy(config)
        local_config.num_hidden_layers = end - start
        if getattr(local_config, "layer_types", None):
            local_config.layer_types = local_config.layer_types[start:end]
        model.config = model.model.config = local_config
        for index, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = index
        if start != 0:
            model.model.embed_tokens = None
        if end != spec.layers:
            model.model.norm = nn.Identity()
            model.lm_head = nn.Identity()
        # Rotary buffers are derived, non-persistent, and must be recreated off meta.
        model.model.rotary_emb = type(model.model.rotary_emb)(config=config, device=torch.device(self.device))
        for prefix, (inputs, outputs) in quantized_modules(spec.config, spec.tensors).items():
            global_index = int(prefix.split(".")[2])
            if not start <= global_index < end:
                continue
            local = re.sub(r"^model\.layers\.\d+", f"model.layers.{global_index - start}", prefix)
            original = model.get_submodule(local)
            packed = {
                suffix: spec.tensors[prefix + "." + suffix]
                for suffix in ("qweight", "qzeros", "scales", "g_idx")
                if prefix + "." + suffix in spec.tensors
            }
            parent, attr = local.rsplit(".", 1)
            setattr(
                model.get_submodule(parent),
                attr,
                PackedLinear(
                    inputs,
                    outputs,
                    spec.quantization,
                    spec.config["quantization_config"]["group_size"],
                    packed,
                    original.bias is not None,
                ),
            )
        self.model = model
        for name in self._weight_names(model):
            if self.weight_mode == "stream" and name.startswith("model.layers."):
                continue
            self._assign(name)
        if self.weight_mode == "stream":
            for index, layer in enumerate(model.model.layers):

                def load_layer(module, args, i=index):
                    for name in self._weight_names(module):
                        self._assign(f"model.layers.{i}.{name}")

                def release_layer(module, args, output):
                    module.to_empty(device="meta")

                layer.register_forward_pre_hook(load_layer)
                layer.register_forward_hook(release_layer, always_call=True)
        model.eval()
        if start == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.store.root / "tokenizer", local_files_only=True, trust_remote_code=False
            )
        self.epoch = epoch

    @staticmethod
    def _weight_names(module):
        return [name for name, _ in module.named_parameters(remove_duplicate=False)] + [
            name
            for name, _ in module.named_buffers()
            if name.endswith((".qweight", ".qzeros", ".scales", ".g_idx"))
        ]

    def _assign(self, name):
        source = re.sub(r"^model\.layers\.(\d+)\.", lambda m: f"model.layers.{int(m[1]) + self.start}.", name)
        if source == "lm_head.weight" and source not in self.spec.tensors:
            source = "model.embed_tokens.weight"
        if source not in self.spec.tensors:
            raise SoulError(f"Adapter cannot load required tensor {source}", 400, "unsupported_model")
        parent, attr = name.rsplit(".", 1)
        module = self.model.get_submodule(parent)
        integer = attr in ("qweight", "qzeros", "g_idx")
        tensor = self.store.tensor(source, self.device, torch.int32 if integer else self.dtype)
        if attr == "g_idx" and (int(tensor.min()) < 0 or int(tensor.max()) >= module.scales.shape[0]):
            raise SoulError(f"Invalid GPTQ group indices: {source}", 400, "unsupported_model")
        if attr in module._buffers:
            setattr(module, attr, tensor)
        else:
            setattr(module, attr, nn.Parameter(tensor, requires_grad=False))

    def check(self, epoch):
        if self.model is None or epoch != self.epoch:
            raise SoulError("Stale deployment; reload the pipeline")

    def tokenize(self, payload):
        if self.tokenizer is None:
            raise ValueError("Tokenization belongs to the first stage")
        if "prompt" in payload:
            ids = self.tokenizer.encode(payload["prompt"], add_special_tokens=True)
        elif self.tokenizer.chat_template:
            ids = self.tokenizer.apply_chat_template(
                payload["messages"], tokenize=True, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in payload["messages"]) + "\nassistant:"
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids:
            ids = [self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0]
        eos = self.spec.generation_config.get(
            "eos_token_id", self.spec.config.get("eos_token_id", self.tokenizer.eos_token_id)
        )
        return {"input_ids": ids, "eos_ids": eos if isinstance(eos, list) else [eos]}

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    @torch.inference_mode()
    def forward(self, request):
        self.check(request["epoch"])
        session_id, position = request["session_id"], request["position"]
        now = time.monotonic()
        self.sessions = {k: v for k, v in self.sessions.items() if now - v[2] < 300}
        session = self.sessions.get(session_id)
        if position == 0:
            if self.sessions and session is None:
                raise SoulError("Node already has an active session")
            session = (DynamicCache(config=self.model.config), 0, now)
        if session is None or position != session[1]:
            raise SoulError("KV cache is missing or sequence position is stale")
        cache = session[0]
        if self.start == 0:
            ids = request["input_ids"]
            if (
                not ids
                or len(ids) > 128
                or any(not isinstance(i, int) or not 0 <= i < self.spec.config["vocab_size"] for i in ids)
            ):
                raise ValueError("Invalid input token IDs")
            inputs = {"input_ids": torch.tensor([ids], dtype=torch.long, device=self.device)}
            length = len(ids)
        else:
            hidden = decode_tensor(
                request["hidden"], self.device, self.dtype, self.spec.config["hidden_size"], 128
            )
            length = hidden.shape[1]
            inputs = {"inputs_embeds": hidden}
        if position < 0 or position + length > self.spec.max_context:
            raise SoulError("Context window exceeded", 400, "context_length_exceeded")
        positions = torch.arange(position, position + length, device=self.device)
        result = self.model.model(
            **inputs,
            past_key_values=cache,
            cache_position=positions,
            position_ids=positions.unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
        self.sessions[session_id] = (result.past_key_values, position + length, now)
        if self.end != self.spec.layers:
            return {"hidden": encode_tensor(result.last_hidden_state)}
        if not request.get("sample", True):
            return {"prefilled": length}
        logits = self.model.lm_head(result.last_hidden_state[:, -1, :]).float()[0]
        if request.get("return_logits"):
            return {"logits": logits.cpu().tolist()}
        temperature, top_p = request.get("temperature", 0.7), request.get("top_p", 1.0)
        if temperature == 0:
            token = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1:
                sorted_probs, indices = torch.sort(probs, descending=True)
                sorted_probs[(torch.cumsum(sorted_probs, 0) - sorted_probs) >= top_p] = 0
                probs = torch.zeros_like(probs).scatter_(0, indices, sorted_probs)
            # CPU sampling has consistent generator support on CUDA and MPS.
            generator = (
                torch.Generator().manual_seed(request["seed"] + position)
                if request.get("seed") is not None
                else None
            )
            token = int(torch.multinomial(probs.cpu(), 1, generator=generator).item())
        return {"token_id": token}
