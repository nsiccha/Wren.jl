# MCMC sampling

Wren currently provides two very simple samplers.

Results are stored as `FlexiChain{VarName}`; please see the [FlexiChains documentation](https://juliabayes.org/FlexiChains.jl) for more information.

```@example mcmc
using Wren, Wren.Dists

y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

@wren function eight_schools(y, sigma, J)
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(0, 5)
    theta ~ MvNormalScale(fill(mu, J), tau * I)
    for i in 1:J
        @observe y[i] ~ Normal(theta[i], sigma[i])
    end
    return (mu=mu, tau=tau)
end

model = eight_schools(y, sigma, J)
```

## Prior

This simply creates a chain with samples from the prior.

```@docs
Wren.prior
```

```@example mcmc
using Wren
prior(model, 1000, 4; progress=false)
```

## [NUTS](@id nuts)

```@docs
Wren.nuts
```

```@example mcmc
using Enzyme
adtype = AutoEnzyme(; mode=Reverse)

using Wren
chain = nuts(model, 1000, 4; adtype=adtype, progress=false)
```

### Gradient evaluations

Almost all of the cost of NUTS lies in evaluating the gradient of the log density, once per leapfrog step.
Each iteration of the chain stores two counts: `n_steps`, the leapfrog steps (and so gradient evaluations) of that transition, and `n_gradient_evaluations`, the running total of the chain's gradient evaluations up to and including that iteration.
The running total starts at the chain's first evaluation, so it includes the initial point, the initial step-size search and warm-up, which are not otherwise returned.
For example, to get the work spent on each chain before and during the retained draws:

```@example mcmc
n = chain[:n_gradient_evaluations]
s = chain[:n_steps]
before = n[1, :] .- s[1, :]  # initialisation and warm-up
during = sum(s; dims=1)      # retained draws
(before=collect(before), during=vec(collect(during)), total=collect(n[end, :]))
```

These counts are exact.
They do not include the gradient evaluations made once per call while preparing the [`VecModel`](@ref), before any chain starts.

### Initialisation

The [`nuts`](@ref) function accepts an optional `init` argument, which controls how the initial parameter values for the MCMC chains are controlled.

`init` should be a subtype of [`AbstractInitialization`](@ref).
Currently, Wren only provides a single `FindFiniteDensity`, which attempts to find a point with finite probability density.

However, you can specify how that point is searched for by passing an [`AbstractLatentSource`](@ref) to the `FindFiniteDensity` constructor.
As a shorthand, you can also pass a latent source (or a `VarNamedTuple` of values) directly as `init`; this is treated as `FindFiniteDensity(init)`.
For example, to replicate Stan's default behaviour of sampling from `[-2, 2]` in unconstrained space, you can pass:

```julia
init = FindFiniteDensity(FromUnconstrainedInterval(-2, 2))
nuts(model, 1000, 4; adtype, init, progress=false)
```

In the future, we hope to provide more initialisation strategies (e.g. using [Pathfinder](@ref pathfinder)).

!!! note "Separate initialisation for different chains"

    Currently, the same initialisation strategy is used for all chains.
    This does not necessarily mean that all chains will start at the same point; it just means that the way the starting point is searched for is the same.
    For example, the four starting points may all be drawn from the prior, but that does not mean that they will all be the same point.

    In the (near) future the API will be generalised to also accept a vector of initialisation strategies, one for each chain.

```@docs; canonical=false
AbstractInitialization
FindFiniteDensity
```
