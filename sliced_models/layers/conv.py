from torch import Tensor
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Optional, Union
from torch.nn import functional as F

class SlicedConv2d(nn.Conv2d):
    def __init__(
        self,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(
            *args,
            **kwargs
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor, d_in: Optional[int] = None, d_out: Optional[int] = None) -> Tensor:
        # reduced the input channels to d_in if specified
        if d_in is not None or d_out is not None:
            d_in = d_in if d_in is not None else self.in_channels
            d_out = d_out if d_out is not None else self.out_channels
            if d_in < input.shape[1]:
                input = input[:, :d_in, :, :]
            weight = self.weight[:d_out, :d_in, :, :] if d_in is not None else self.weight
            bias = self.bias[:d_out] if self.bias is not None else None
            return self._conv_forward(input, weight, bias)
    
        
        return self._conv_forward(input, self.weight, self.bias)