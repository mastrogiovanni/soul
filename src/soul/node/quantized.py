"""Portable packed 4-bit linear layers; bounded dequantization, no custom kernels."""

import torch
from torch import nn

from soul.quantization import DEQUANT_BLOCK


class PackedLinear(nn.Module):
    def __init__(self, inputs, outputs, format, group_size, tensors, bias):
        super().__init__()
        self.in_features, self.out_features = inputs, outputs
        self.format = format
        self.group_size = inputs if group_size == -1 else group_size
        for name, tensor in tensors.items():
            dtype = torch.int32 if name in ("qweight", "qzeros", "g_idx") else torch.float32
            self.register_buffer(name, torch.empty(tensor["shape"], dtype=dtype, device="meta"))
        self.bias = nn.Parameter(torch.empty(outputs, device="meta"), requires_grad=False) if bias else None

    def forward(self, x):
        # Keep only a small output-column tile expanded. Packed weights stay packed
        # on disk AND in device memory, including when the rest of the layer streams.
        device = x.device
        shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
        if self.format == "awq-gemm-4":
            shifts = shifts[torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=device)]
            groups = torch.arange(self.in_features, device=device) // self.group_size
        else:
            groups = self.g_idx.long()
        output = x.new_empty((*x.shape[:-1], self.out_features))
        for start in range(0, self.out_features, DEQUANT_BLOCK):
            end = min(start + DEQUANT_BLOCK, self.out_features)
            zeros = ((self.qzeros[:, start // 8 : end // 8, None] >> shifts) & 15).reshape(-1, end - start)
            if self.format == "awq-gemm-4":
                weight = ((self.qweight[:, start // 8 : end // 8, None] >> shifts) & 15).reshape(
                    self.in_features, -1
                )
            else:
                # GPTQ v1 stores zero points minus one; weights pack input rows.
                zeros = zeros + 1
                weight = ((self.qweight[:, None, start:end] >> shifts[None, :, None]) & 15).reshape(
                    self.in_features, -1
                )
            weight = (weight - zeros[groups]).to(x.dtype)
            weight.mul_(self.scales[groups, start:end])
            output[..., start:end] = torch.matmul(x, weight)
        if self.bias is not None:
            output.add_(self.bias)
        return output
