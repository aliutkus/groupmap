import torch
from torch import nn
from einops import rearrange
import contextlib
import warnings
import math


# some basic templates for quantile functions
def sparse(quantiles, ninety_quantile=0.05):
    """"returns a cdf for a distribution with maximum value 1,
    whose `ninety_quantiles` matches the indicated value.
    This means that 90% of the samples have a value smaller than
    the provided value."""
    exponent = int(math.log(ninety_quantile)/math.log(0.9))
    return quantiles ** exponent
def uniform(quantiles):
    return quantiles
def gaussian(quantiles):
    quantiles = torch.clamp(quantiles, min=1e-5, max=1-1e-5)
    return torch.sqrt(torch.as_tensor(2)) * torch.erfinv(2*quantiles-1)
def cauchy(quantiles):
    quantiles = torch.clamp(quantiles, min=1e-5, max=1-1e-5)
    return torch.tan(torch.tensor(math.pi)*(quantiles-0.5))


class Map(nn.Module):
    r"""Applies an optimal transportation plan over a multidimensional input. For each element
    of the input tensor, the following transformation is applied

    .. math::

        y = Q\left(F\left(x\right)\right)

    Where :math:`Q` is the target quantile function for the corresponding feature, while
    :math:`F` is the input cumulative distribution function, which is in practice simply
    estimated via sorting the input.
    This formula corresponds to the classical _increasing rearrangement_ method to optimally
    transport scalar input data distributed wrt a distribution :math:`\mu` to another scalar
    distribution :math:`\nu`, by mapping quantile to quantile (min to min, median to median,
    max to max, etc.)  

    .. note::
        In this implementation, the input cdf is always estimated from the input batch, at 
        both training and test time: there is no tracking of the statistics.

    Args:
        num_groups: number of groups to separate the channels into. Default: 1 (LayerNorm
            behavior)
        num_channels: the number of channels expected in input, of shape (N, C, ...)
        target_quantiles: the target quantiles function. must be a callable that takes a
            Tensor with entries between 0 and 1, and returns a Tensor with same shape.  
            Can notably be one of the provided `groupmap.gaussian` (default)
            `groupmap.uniform`, `groupmap.sparse`.
        eps: the standard deviation of the noise to add to the input before sorting to avoid
            duplicates for stability.
    Shape:
        - Input: :math:`(N, C ...)`
        - Output: :math:`(N, C, ...)` (same shape as input)

    """
    def __init__(
        self,
        num_groups:1,
        num_channels:None,
        target_quantiles=gaussian,
        eps=0,
        device=None,
        dtype=None,
    ):
        super(Map, self).__init__()
        if num_groups > 1:
            assert not (num_channels % num_groups), 'If `num_groups>1`, then the number of channels must be divisible by the number of groups.'

        # save parameters
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.target_quantiles = target_quantiles
        self.eps = eps

        # a temp tensor of shape (batchsize, num_channels, num_features)
        # containing a repeated linspace(0, 1, num_features). No need to compute it again
        # at each forward call so keeping it as a temp variable and computing it anew only
        # if invalidated.
        self.range_tensor = None

    def forward(self, x):
        """
        Applies optimal transport maps

        Args:
            x : (batch_size, num_channels, ...)
                the data to process
        """
        # getting input size
        input_shape = x.shape
        batch_size, num_channels = input_shape[:2]

        # flattening all features together
        x = x.view(batch_size, num_channels, -1)
        num_features = x.shape[-1]

        # making x (batch_size pergroup) num_groups, num_features
        x = rearrange(x, 'b (g p) f -> b g (p f)', g=self.num_groups)

        # add a small epsilon if required to avoid having doublons for stability
        if self.eps:
            x = x + torch.randn(x.shape, device=x.device) * self.eps

        # sort each group by ascending value
        indices = x.argsort(dim=-1)

        # assigns each value to its quantile
        if self.range_tensor is None or self.range_tensor.shape != x.shape:
            # need to create a range: 1...num_features array
            self.range_tensor = torch.linspace(0, 1, x.shape[-1], device=x.device)[None, None].repeat(batch_size, self.num_groups, 1)
        y = self.range_tensor.scatter(dim=-1, index=indices, src=self.range_tensor)

        # get the corresponding output value from the target quantiles
        y = self.target_quantiles(y)

        # restoring shape. First (batchsize, num_channels, num_features)
        y = rearrange(y, 'b g (p f) -> b (g p) f', f=num_features)

        # then view as input
        y = y.view(*input_shape)

        return y

class Affine(nn.Module):
    """
    Affine transformation module.

    For input of arbitrary shape (N, ...), first view it as (N,) + affine_shape  + (-1).
    Then applies an affine transformation
    $y_{ni} = x_{ni}*\gamma_{ni}[None]+\beta_{ni}[None]$.
    before getting back to the original shape.

    Several cases of interest:
    * `affine_shape` is (num_channels,): 
        corresponds to the behaviour of `InstanceNorm`, which applies the same weight-bias to
        all entries of a given channel
    * Input has shape `(N,)+affine_shape` exactly:
        corresponds to the behaviour of `LayerNorm`, which applies different weight-bias to each
        each entry of the input after normalization.
 

    The gain and bias parameters have shape `((num_channels,) + affine_shape`
    """
    def __init__(self, affine_shape, device=None, dtype=None):
        super(Affine, self).__init__()
        self.affine_shape = affine_shape
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(self.affine_shape, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(self.affine_shape, **factory_kwargs))

    def forward(self, x):
        original_shape = x.shape
        batchsize = original_shape[0]
        x = x.view(batchsize, *self.affine_shape, -1)
        x = x * self.weight[..., None] + self.bias[..., None]
        x = x.view(*original_shape)
        return x


class GroupMap(nn.Module):
    """Wrapper for a mapping equivalent of GroupNorm"""
    def __init__(self, num_groups, num_channels, eps=0, affine=True, target_quantiles=gaussian, device=None, dtype=None):
        super(GroupMap, self).__init__()
        self.mapper = Map(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            device=device,
            dtype=dtype,
            target_quantiles=target_quantiles
        )
        if affine:
            self.affine = Affine(affine_shape=(num_channels,), device=device, dtype=dtype)
        else:
            self.affine = None

    def forward(self, x):
        x = self.mapper(x)
        if self.affine:
            x = self.affine(x)
        return x

class LayerMap(nn.Module):
    """Wrapper for a mapping equivalent of LayerNorm"""
    def __init__(self, normalized_shape, eps=0, elementwise_affine=True, target_quantiles=gaussian, device=None, dtype=None):
        super(LayerMap, self).__init__()
        self.mapper = Map(
            num_groups=1,
            num_channels=None,
            eps=eps,
            device=device,
            dtype=dtype,
            target_quantiles=target_quantiles
        )
        if elementwise_affine:
            self.affine = Affine(affine_shape=normalized_shape, device=device, dtype=dtype)
        else:
            self.affine = None

    def forward(self, x):
        x = self.mapper(x)
        if self.affine:
            x = self.affine(x)
        return x

class InstanceMap(nn.Module):
    """Wrapper for a mapping equivalent of InstanceNorm. Doesn't make a difference if the input is of any dimension"""
    def __init__(self, num_features, eps=0, momentum=0.1, affine=False, track_running_stats=False, target_quantiles=gaussian, device=None, dtype=None):
        super(InstanceMap, self).__init__()
        self.mapper = Map(
            num_groups=num_features,
            num_channels=num_features,
            eps=eps,
            device=device,
            dtype=dtype,
            target_quantiles=target_quantiles
        )
        if affine:
            self.affine = Affine(affine_shape=(num_features,), device=device, dtype=dtype)
        if track_running_stats:
            warnings.warn('`track_running_stats` has been set for `InstanceMap`, but the feature is not implemented. Ignoring this and using input statistics in all cases.')

    def forward(self, x):
        x = self.mapper(x)
        if self.affine:
            x = self.affine(x)
        return x


InstanceMap2d = InstanceMap
