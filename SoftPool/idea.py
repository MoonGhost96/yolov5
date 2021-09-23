from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _pair

import softpool_cuda





class CUDA_SOFTPOOL2d(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None):
        # Create contiguous tensor (if tensor is not contiguous)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.size()
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)
        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1
        output = input.new_zeros((B, C, oH, oW))
        softpool_cuda.forward_2d(input.contiguous(), kernel, stride, output)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        # Create contiguous tensor (if tensor is not contiguous)
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        saved = [grad_output.contiguous()] + list(ctx.saved_tensors) + [ctx.kernel,ctx.stride] + [grad_input]
        softpool_cuda.backward_2d(*saved)
        # Gradient underflow
        saved[-1][torch.isnan(saved[-1])] = 0
        return saved[-1], None, None



'''
---  S T A R T  O F  F U N C T I O N  S O F T _ P O O L 2 D  ---
    [About]
        Function for dowsampling based on the exponenial proportion rate of pixels (soft pooling).
        If the tensor is in CUDA the custom operation is used. Alternatively, the function uses
        standard (mostly) in-place PyTorch operations for speed and reduced memory consumption.
        It is also possible to use non-inplace operations in order to improve stability.
    [Args]
        - x: PyTorch Tensor, could be in either cpu of CUDA. If in CUDA the homonym extension is used.
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - force_inplace: Bool, determines if in-place operations are to be used regardless of the CUDA
                         custom op. Mostly useful for time monitoring. Defaults to `False`.
    [Returns]
        - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL2d.apply(x, kernel_size, stride)
        # Replace `NaN's if found
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d] -> [b x c x d']
    x = F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))
'''
---  E N D  O F  F U N C T I O N  S O F T _ P O O L 2 D  ---
'''

