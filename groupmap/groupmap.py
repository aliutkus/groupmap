import torch
from torch import nn
import contextlib
from torchinterp1d import Interp1d
import warnings
import math
from .utils import qloss


# some basic templates for quantile functions
def sparse(num_quantiles, ninety_quantile=0.05):
    exponent = int(math.log(ninety_quantile)/math.log(0.9))
    return torch.linspace(0,1,num_quantiles) ** exponent
def uniform(num_quantiles):
    return torch.linspace(0,1,num_quantiles)
def gaussian(num_quantiles):
    return torch.sqrt(torch.as_tensor(2)) * torch.erfinv(2*torch.linspace(1e-5,1-1e-5,num_quantiles)-1)
def cauchy(num_quantiles):
    return torch.tan(torch.tensor(math.pi)*(torch.linspace(1e-5,1-1e-5,num_quantiles)-0.5))


class BatchOT(nn.Module):
    r"""Applies Batch Optimal transportation over a multidimensional input. For each element
    of the input tensor, the following transformation is applied

    .. math::

        y = Q_\nu\left(Q_\mu^{-1}\left(x\right)\right)

    Where :math:`Q^{-1}_\mu\left(v\right)=\mathbb{P}\left(x\geq v\right)` is the cumulative
    distribution function for the input distribution for the corresponding feature, while
    :math:`Q_\nu\left(q\right)` for :math:`q\in[0 1]` gives the :math:`q^th` quantile for
    the output distribution (0 is min, 0.5 is median, 1 is max, etc.).
    This formula corresponds to the classical _increasing rearrangement_ method to optimally
    transport scalar input data distributed wrt a distribution :math:`\mu` to another scalar
    distribution :math:`\nu`, by mapping quantile to quantile (min to min, median to median,
    max to max, etc.)  

    Both the input and output quantile functions are encoded as `num_quantiles` vectors, so
    that :math:`Q_\mu` has shape `(num_features, num_quantiles)`, while  :math:`Q_\nu` has
    shape `(num_quantiles,)`.
    The quantiles for the input distribution are estimated per-feature over
    the mini-batches and the target quantile function :math:`Q_\nu` is optionally learnable.
    By default, :math:`Q_\nu` is set to the quantile function of the uniform distribution over
    :math:`[0, 1]`.
    The input quantiles are estimated via minimization of the rotated hinge-loss function over
    the input data, through the use of an internal Adam optimizer with learning rate `lr`.

    .. note::
        This :attr:`lr` argument is different from one used in the optimizer otherwise used by
        the user. At each run, a `loss` attribute is updated for monitoring purposes.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, ...)`
        num_quantiles: the number of quantile used to characterize the input and
            output distributions. Must be at least 2.
        target_quantiles: the target quantiles. can be either:
            * a Tensor with shape `(num_quantiles,)`. Will be used directly.
            * `None`: will be initialized to the quantile function for uniform distribution as
              `linspace(0,1,num_quantiles)`
            * a callable in which case, this should accept `num_quantiles` and return a 
              Tensor with shape `(num_quantiles,)`. Can for instance be one of the templates
              provided in the module: `batchot.uniform`, `batchot.gaussian`, `batchot.sparse`. 
        batchwise_matching: a boolean value that indicates whether each batch should be
            matched to the target distribution during training, or whether the trained
            source quantiles should be used instead (thus behaving as in eval mode). When true,
            the target distribution cannot exactly be enforced if the batch size is smaller than
            the number of quantiles.
        train_targets: a boolean value that indicates whether the target distribution
            will be trainable through gradient descent.
        lr: The learning rate used for internal training of the input quantile function.
            Default: 1e-1
        smoothing: int indicating the standard deviation of the additive noise used to
            avoid equal quantiles. The noise variance will be the average interquantile range
            divided by this number. None if no smoothing.

    Shape:
        - Input: :math:`(N, C ...)`
        - Output: :math:`(N, C, ...)` (same shape as input)

    """

    def __init__(
        self,
        num_features,
        num_quantiles,
        target_quantiles=None,
        train_targets=False,
        batchwise_matching=True,
        optimizer_class=torch.optim.RMSprop,
        lr=1e-1,
        smoothing = 10,
        device=None,
        dtype=None,
    ):
        super(BatchOT, self).__init__()

        # save parameters
        self.num_features = num_features
        self.num_quantiles = num_quantiles
        self.train_targets = train_targets
        self.batchwise_matching = batchwise_matching
        self.loss = 0
        self.optimizer_class = optimizer_class
        self.lr=lr
        self.smoothing = smoothing
        factory_kwargs = {'device': device, 'dtype': dtype}

        # check quantiles    
        assert num_quantiles >= 2, 'There must be at least 2 quantiles (min-max) to enforce'
        self.register_buffer('quantiles', torch.linspace(0, 1, num_quantiles, **factory_kwargs))

        if target_quantiles is None:
            # uniform distribution if not specified
            target_quantiles = torch.linspace(0, 1, num_quantiles, **factory_kwargs)
        elif callable(target_quantiles):
            # in case it's callable, just replace it by the outcome
            target_quantiles = target_quantiles(num_quantiles)
        

        # targets length must match the number of quantiles
        assert target_quantiles.numel() == num_quantiles,\
               'The number of target quantiles must match num_quantiles'

        # we check the target is non-increasing, and save it
        target_quantiles = torch.as_tensor(target_quantiles,  **factory_kwargs)

        assert torch.all(torch.diff(target_quantiles) >= 0), 'target quantiles must be nondecreasing'
        self.register_parameter('target_quantiles', nn.Parameter(target_quantiles, requires_grad=train_targets))

        # running estimate
        self.register_buffer('source_quantiles',
                             torch.zeros(num_quantiles, num_features, requires_grad=True,**factory_kwargs ))

        # initially we don't have an optimizer
        self.quantiles_optimizer = None

    def forward(self, x):
        """
        Applies batch transport

        Args:
            x : (batch_size, num_features,) + other_shape
                the data to process
        """

        # making x (batch_size,)+other_shape+(num_features,)
        x = torch.movedim(x, 1, -1)
        original_shape = x.shape
        # making x (num_samples, num_features) where num_samples is the total number of observations per feature.
        x = x.view(-1, self.num_features)

        if self.train_targets != self.target_quantiles.requires_grad:
            warnings.warn('`target_quantiles.requires_grad` and the `train_targets` flag do not match. please check this')

        # we initialize training if needs be
        if self.quantiles_optimizer is None and self.training:
            # we take the current batch quantiles as a rough estimate (probably bad because the batchsize
            # is smaller than num_quantiles)
            batch_quantiles = torch.quantile(input=x, q=self.quantiles, dim=0)
            self.source_quantiles = batch_quantiles.clone().detach()
            self.source_quantiles.requires_grad = True

            self.quantiles_optimizer = self.optimizer_class([self.source_quantiles], lr=self.lr)
    
        if self.training:
            # reajust the learning rate dynamically depending on current quantiles estimates
            self.quantiles_optimizer.lr = (self.source_quantiles.max()-self.source_quantiles.min()) * self.lr
            per_feature_std = torch.maximum(
                torch.tensor(1e-10).to(x.device),
                (self.source_quantiles[-1]-self.source_quantiles[0])/self.num_quantiles/10.
            ) # (num_features,)

            # preparing the additive noise for smoothing the quantiles
            if self.smoothing is None:
                noise = 0
            else:
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * per_feature_std[None]

            # training the quantiles with rotated hinge loss
            self.quantiles_optimizer.zero_grad()
            loss = qloss(self.quantiles, self.source_quantiles, x+noise).mean()
            loss.backward()
            self.quantiles_optimizer.step()
                        
            # remembering the loss for monitoring purposes
            self.loss = loss.item()

        # setting the source quantiles either as the empirical batch distribution in case
        # of batchwise matching in a training context, or as the trained distribution otherwise 
        if self.training and self.batchwise_matching:
            source_quantiles = torch.quantile(input=x, q=self.quantiles, dim=0)
        else:
            source_quantiles = self.source_quantiles

        ## sorting the quantiles
        source_quantiles = torch.sort(source_quantiles, dim=0)[0]

        """original_min = source_quantiles[0, None]
        original_max = source_quantiles[-1, None]
        diff = source_quantiles[1:] - source_quantiles[:-1] + 1e-3
        source_quantiles = torch.cumsum(
            torch.cat([source_quantiles[0, None], diff]),
            dim=0
        )"""

        per_feature_std = torch.maximum(
            torch.tensor(1e-10).to(x.device),
            (source_quantiles[-1]-source_quantiles[0])/self.num_quantiles/10.
        ) # (num_features,)

        # adding noise to smooth quantiles if desired
        if self.smoothing is not None:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * per_feature_std[None]
            x=x+noise

        # the target distribution is non-decreasing in all cases
        target_quantiles = torch.sort(self.target_quantiles, dim=0)[0]

        # increasing rearrangement to match source and target (doing the actual transportation)
        x = Interp1d()(x=source_quantiles.T, y=self.quantiles, xnew=x.T)
        x = torch.clip(x, 0, 1)
        x = Interp1d()(x=self.quantiles, y=target_quantiles, xnew=x).T

        # restoring shape (batch_size,)+other_shape + (num_features,)
        x = x.view(*original_shape)
        # restoring original shape (batch_size, num_features,)+other_shape
        x = torch.movedim(x, -1, 1)
        return x
