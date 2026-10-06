import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    MistralConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen3Config,
)

CONFIGS = {"llama": LlamaConfig, "mistral": MistralConfig, "qwen2": Qwen2Config, "qwen3": Qwen3Config}
torch.set_num_threads(1)


def make_model(directory, architecture="llama", layers=4, tied=False):
    torch.manual_seed(42)
    config = CONFIGS[architecture](
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        sliding_window=8,
        tie_word_embeddings=tied,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        head_dim=8,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).eval()
    model.save_pretrained(directory)
    vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "Hello": 4,
        "world": 5,
        "user": 6,
        "assistant": 7,
        ":": 8,
        **{f"word{i}": i for i in range(9, 32)},
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]"
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{{ 'assistant: ' }}"
    )
    fast.save_pretrained(directory)
    return model


@pytest.fixture
def tiny_model(tmp_path):
    path = tmp_path / "model"
    model = make_model(path)
    return path, model
