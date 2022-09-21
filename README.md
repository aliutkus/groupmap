# Group Map: beyond mean and variance matching for deep learning

Define a `GroupMap` module, to apply an optimal transportation map over a multidimensional input. The objective is to transform the input to follow a prescribed arbitrary distribution at the output, like uniform, Gaussian, sparse, etc. 

> The main difference between `GroupMap` and normalization modules like `InstanceNorm`, `LayerNorm` or `GroupNorm` is it enforces the output to match _a whole distribution_ instead of just mean and variance.

<img src="groupmap.jpg" width="500">

> :warning: In this simplified implementation, there is no tracking of the input statistics and the module always uses the batch statistics for mapping, both at training and test time, similarly to `nn.LayerNorm`, `nn.GroupNorm`, or the default behaviour of `nn.InstanceNorm2d`. 

## What it does

Let $x$ be the input tensor, of arbitrary shape `(B, C, ...)` and let $x_{nc\boldsymbol{f}}$ be one of its entries for sample $n$, channel $c$ and a tuple of indices $\boldsymbol{f}$ corresponding to features. For instance, for images, we would have 2D features $\boldsymbol{f}=(i,j)$ for the row and column of a pixel. 

For each element of the input tensor, the following transformation is applied:

$y_{nc\boldsymbol{f}}=\mathit{Q}\left(\mathcal{F}_{nc}\left(x_{nc\boldsymbol{f}}\right)\right)$

Where:  
* $\forall q\in[0, 1],~\mathit{Q}(q)\in\mathbb{R}$ is the target _quantile function_. It describes what the distribution of the output should be and is provided by the user. The `GroupMap` module guarantees that the output will have a distribution that matches this target.
    > Typically, $\mathit{Q}$ is the quantile function for a classical distribution like uniform, Gaussian, Cauchy, etc.
    $\mathit{Q}(0)$ is the minimum of the target distribution, $\mathit{Q}(0.5)$ its median, $\mathit{Q}(1)$ its maximum, etc.
* $\mathcal{F}_{nc}(v)=\mathbb{P}(x_{nc,\boldsymbol{f}}\leq v)\in[0, 1]$  is the input cumulative distribution function (cdf) for sample $n$ and channel $c$.
   It is estimated on data for sample $n$. Several behaviours are possible, depending on which part of $x_n$ it is computed from.
   * It can be the cdf for just a particular channel $x_{nc}$, then behaving like some optimal-transport version of `InstanceNorm`.
   * It can be computed and shared over all channels of sample $x_n$  (as in `LayerNorm`)
   * It can be computed and shared over groups of channels (as in `GroupNorm`).
    > $\mathcal{F}_{nc}(v)=0$ if $v$ is the minimum of the input distribution, $0.5$ for the median, $1$ for the maximum, etc.).  


This formula corresponds to the classical _increasing rearrangement_ method to optimally transport scalar input data distributed wrt a distribution to another scalar distribution, by mapping quantile to quantile (min to min, median to median, max to max, etc.)  

## Interface

This repository defines a `groupmap.GroupMap`, with the following parameters:
* `num_groups`: number of groups to separate the channels into.
* `num_channels`: the number of channels expected in input, of shape (N, C, ...)
* `target_quantiles`: the target quantiles function. must be a callable that takes a Tensor with entries between 0 and 1, and returns a Tensor with same shape.  
You typically want to use the functions provided in this repo: `groupmap.gaussian` (default) `groupmap.uniform`, `groupmap.sparse`.

For more insights on how `num_groups` and `num_channels` interact, please check the [documentation for `nn.GroupNorm`](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html): `GroupMap` purposefully uses the same syntax.

## Shapes
* Input: `(N, C ...)`
* Output: `(N, C, ...)` (same shape as input)

## Example

```python
# import
import torch
from torch import nn
import groupmap

# create a dummy tensor with 4 channels and 4 features
v = torch.randn(1, 4, 4).to(device)
print('original data')
print(v[0].cpu().numpy())

# groupmap with each row as a group (as in instancenorm)
gm = groupmap.GroupMap(
    num_groups=4,
    num_channels=4,
    target_quantiles=groupmap.uniform
).to(device)

print('\ngroupmap with each row as a group')
print(gm(v)[0].cpu().numpy())

# groupmap with all rows together (as in layernorm)
gm = groupmap.GroupMap(
    num_groups=1,
    num_channels=4,
    target_quantiles=groupmap.uniform
).to(device)
print('\ngroupmap with all rows together as a group')
print(gm(v)[0].cpu().numpy())

# groupmap with groups of 2 consecutive rows (as in groupnorm)
gm = groupmap.GroupMap(
    num_groups=2,
    num_channels=4,
    target_quantiles=groupmap.uniform
).to(device)
print('\ngroupmap with 2 groups (two consecutive rows together)')
print(gm(v)[0].cpu().numpy())
```

This outputs:
```
original data
[[ 2.2675493  -0.5355932  -0.9594439  -0.526434  ]
 [-0.56300575  0.45603803 -0.23818    -0.2557832 ]
 [-0.8675442  -0.9519985  -1.6582233  -0.3447896 ]
 [-2.7685483   1.2761954   1.6266245  -1.3452631 ]]

groupmap with each row as a group
[[1.         0.33333334 0.         0.6666666 ]
 [0.         1.         0.6666666  0.33333334]
 [0.6666666  0.33333334 0.         1.        ]
 [0.         0.6666666  1.         0.33333334]]

groupmap with all rows together as a group
[[1.         0.4666667  0.20000002 0.5333333 ]
 [0.40000004 0.8        0.73333335 0.6666666 ]
 [0.33333334 0.26666668 0.06666667 0.59999996]
 [0.         0.8666667  0.93333334 0.13333334]]

groupmap with 2 groups (two consecutive rows together)
[[1.         0.2857143  0.         0.42857146]
 [0.14285715 0.85714287 0.71428573 0.57142854]
 [0.57142854 0.42857146 0.14285715 0.71428573]
 [0.         0.85714287 1.         0.2857143 ]] 
 ```
 
 ## Citation
 I don't have time to write a paper about GroupMap now. If you find this repository useful, please cite it this way:
 ```
 @software{Liutkus_GroupMap_beyond_mean_2022,
    author = {Liutkus, Antoine},
    month = {9},
    title = {{GroupMap: beyond mean and variance matching for deep learning}},
    url = {https://www.github.com/aliutkus/groupmap},
    year = {2022}
}
 ```
