# [Distributions](@id reference-distributions)

This page lists [Wren's built-in distributions](@ref wrendists).

Each distribution specifies whether it has a fast vectorised implementation of `logdensityof` when it occurs on the right-hand side of a tilde inside [a `@batch` loop](@ref batched).
If a distribution does _not_ have a custom overload, Wren will fall back to a generic implementation using `independent` (semantically equal to `Distributions.product_distribution`).
This is always correct, but may not necessarily be the most efficient implementation.
In general, vectorised implementations are most useful for distributions that are used as likelihoods in models, where the log-density is evaluated over a large dataset.
New implementations of vectorised log-densities are very much welcome!

## Univariate

### Continuous unbounded

```@docs
WrenDists.Normal
WrenDists.Laplace
WrenDists.Cauchy
WrenDists.Logistic
WrenDists.TDist
WrenDists.Flat
```

### Constrained flat prior

Use `WrenDists.truncated(WrenDists.Flat(); lower=0)` for an unnormalised flat
prior constrained to positive values.

```@docs
WrenDists.truncated
```

### Continuous $\mathbb{R}^+$

```@docs
WrenDists.LogNormal
WrenDists.Exponential
WrenDists.Gamma
WrenDists.InverseGamma
WrenDists.Weibull
WrenDists.HalfNormal
WrenDists.HalfCauchy
WrenDists.HalfTDist
```

### Continuous bounded

```@docs
WrenDists.Uniform
WrenDists.Beta
```

### Discrete

```@docs
WrenDists.Bernoulli
WrenDists.BernoulliLogit
WrenDists.Binomial
WrenDists.BinomialLogit
WrenDists.NegativeBinomial
WrenDists.Poisson
WrenDists.PoissonLog
WrenDists.Categorical
WrenDists.Geometric
```

## Multivariate

```@docs
WrenDists.MvNormalScale
WrenDists.Dirichlet
WrenDists.Multinomial
```
