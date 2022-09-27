# Group Map: beyond mean and variance matching for deep learning

Define `GroupMap`, `InstanceMap`, `LayerMap` modules, to transform the input so that it follows a prescribed arbitrary distribution, like uniform, Gaussian, sparse, etc. 

> The main difference between the `GroupMap`, `InstanceMap` and `LayerMap` modules and their normalization-based counterparts `GroupNorm`, `InstanceNorm` and `LayerNorm` is they enforces the output to match _a whole distribution_ instead of just mean and variance.
> :warning: In this simplified implementation, there is no tracking of the input statistics and the module always uses the batch statistics for mapping, both at training and test time. 

<img src="groupmap.jpg" width="500">

> :warning: In this simplified implementation, there is no tracking of the input statistics and the module always uses the batch statistics for mapping, both at training and test time, similarly to `nn.LayerNorm`, `nn.GroupNorm`, or the default behaviour of `nn.InstanceNorm2d`. 

## What it does

Let $x$ be the input tensor, of arbitrary shape `(B, C, ...)` and let $x_{nc\boldsymbol{f}}$ be one of its entries for sample $n$, channel $c$ and a tuple of indices $\boldsymbol{f}$ corresponding to features. For instance, for images, we would have 2D features $\boldsymbol{f}=(i,j)$ for the row and column of a pixel. 

For each element of the input tensor, the following transformation is applied:

$y_{nc\boldsymbol{f}}=Q\left(F_{nc}\left(x_{nc\boldsymbol{f}}\right)\right) * \gamma_{c\boldsymbol{f}} + \beta_{c\boldsymbol{f}}$

Where:  
* $\forall q\in[0, 1],~Q(q)\in\mathbb{R}$ is the target _quantile function_. It describes what the distribution of the output should be and is provided by the user. The `GroupMap` module guarantees that the output will have a distribution that matches this target.
    > Typically, $Q$ is the quantile function for a classical distribution like uniform, Gaussian, Cauchy, etc.
    $Q(0)$ is the minimum of the target distribution, $Q(0.5)$ its median, $Q(1)$ its maximum, etc.
* $F_{nc}(v)=\mathbb{P}(x_{nc,\boldsymbol{f}}\leq v)\in[0, 1]$  is the input cumulative distribution function (cdf) for sample $n$ and channel $c$.
   It is estimated on data for sample $n$. Several behaviours are possible, depending on which part of $x_n$ it is computed from.
   * It can be the cdf for just a particular channel $x_{nc}$, then behaving like some optimal-transport version of `InstanceNorm`.
   * It can be computed and shared over all channels of sample $x_n$  (as in `LayerNorm`)
   * It can be computed and shared over groups of channels (as in `GroupNorm`).
    > $F_{nc}(v)=0$ if $v$ is the minimum of the input distribution, $0.5$ for the median, $1$ for the maximum, etc.).  
* $\gamma_{c\boldsymbol{f}}$ and $\beta_{c\boldsymbol{f}}$ are parameters for an affine transform that may or may not be activated. If it is activated, it matches classical behaviour, i.e. we have $\gamma_{c\boldsymbol{f}}=\gamma_{c}$ and $\beta_{c\boldsymbol{f}}=\beta_{c}$ for `InstanceMap` and `GroupMap`, while elementwise parameters for `LayerMap`.

This formula corresponds to the classical _increasing rearrangement_ method to optimally transport scalar input data distributed wrt a distribution to another scalar distribution, by mapping quantile to quantile (min to min, median to median, max to max, etc.)  

## Usage

### Specifics

The usage of the modules offered by `groupmap` purposefully matches that of classical normalization modules, so that they may be used as a drop-in replacement. There are two main subtleties with respect to the normalization-based ones.


**target quantiles**. All modules offer a `target_quantiles` parameter, which must be a callable taking a `Tensor` of numbers betweer 0 and 1 as inputs, and returning a `Tensor` of same shape as output with the corresponding quantiles for the target distribution.  

> The Module offers several default target quantiles functions:
>* `groupmap.uniform`: defines the uniform distribution as $Q(q)=q$ 
>* `groupmap.gaussian`: defines the Gaussian distribution as: $Q(q) = \sqrt{2}\text{erf}^{-1}(2q-1)$
    (also known as the probit function).
>* `groupmap.cauchy`: defines the Cauchy distribution as 
    $Q(q)=\tan(\pi(q-1/2))$.

Below is a quick description of the interface for quick reference. For a detailed description of the parameters to `GroupMap`, `LayerMap` and `InstanceMap`, please check the documentation for [`GroupNorm`](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html), [`LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) and [`InstanceNorm`](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html), respectively.

**eps**. Instead of being a constant added to a denominator to avoid a division by zero as in $\star\text{Norm}$ modules, `eps` serves as the standard deviation for an actual random Gaussian noise that is added to the input. The consequence of doing so is to avoid duplicate values in the input, so that computation of the input CDF is well behaved. However, having a value $\epsilon\neq 0$ is not mandatory in mapping-based transformations.
> :warning: all modules have `eps=0` by default.

### `GroupMap`
> input distribution is computed on groups of channels.
* `num_groups`: number of groups to separate the channels into
* `num_channels`: the number of channels expected in input, of shape (N, C, ...)
* `target_quantiles` as detailed above. default is `groupmap.gaussian`.
* `eps`: as detailed above

### `LayerMap`
> input distribution is computed over each whole sample.
* `normalized_shape`: shape of each sample
* `elementwise_affine`: whether or not to activate elementwise affine transformation of the output. If so, `weight` and `bias` parameters are created with shape `normalized_shape`.
* `target_quantiles` as detailed above. default is `groupmap.gaussian`.

### `InstanceMap`
> Input distribution is computed over each channel separately
* `num_features`: number of channels `C` for an input of shape `(N, C, ...)
* `affine`: whether or not to apply a channelwise affine transform.
* `track_running_stats`, `momentum`: in this implementation, these parameters are ignored. Statistics are computed from the input signal *anyways*, both at training and test times.
* `target_quantiles`: as detailed above. default is `groupmap.gaussian`.
* `eps`: as detailed above.
