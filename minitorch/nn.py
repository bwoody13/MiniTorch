from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    if not input._tensor.is_contiguous():
        input = input.contiguous()

    new_height = height // kh
    new_width = width // kw
    intermediate = input.view(batch, channel, new_height, kh , new_width, kw)
    reorder = intermediate.permute(0, 1, 2, 4, 3, 5).contiguous()
    return reorder.view(batch, channel, new_height, new_width, kh * kw), new_height, new_width
    # raise NotImplementedError("Need to implement for Task 4.3")


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    tiled_in, new_h, new_w = tile(input, kernel)
    return tiled_in.mean(4).view(batch, channel, new_h, new_w)     # 4 is index of last thing in view (kh*kw)
    # raise NotImplementedError("Need to implement for Task 4.3")


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        ret = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, ret)
        return ret

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        (input, ret) = ctx.saved_values
        return (input == ret) * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    # TODO: Implement for Task 4.4.
    exp_in = input.exp()
    return exp_in / exp_in.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # ten_max = max_reduce(input.view(input.size), 0)
    # return input - ((input - ten_max).exp().sum(dim).log() + ten_max)
    # return input - input.exp().sum(dim).log()
    # m = max_reduce(input.contiguous().view(input.size), 0)
    m = max(input, dim)
    stable_in = input - m
    return input - stable_in.exp().sum(dim).log() - m


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    tiled_in, new_h, new_w = tile(input, kernel)
    return max(tiled_in, 4).view(batch, channel, new_h, new_w)
    # raise NotImplementedError("Need to implement for Task 4.4")


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if ignore:
        return input

    rand_tensor = rand(input.shape)
    keep = rand_tensor > rate
    return input * keep
    # raise NotImplementedError("Need to implement for Task 4.4")
