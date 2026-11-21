# [Point estimates](@id point-estimates)

Load Optimization.jl and a solver package to find a maximum a posteriori (MAP) estimate.
The functions on this page are defined in WrenCore, so they can also be used with `using WrenCore` alone, without loading Wren's samplers.

```@example optimization; continued=true
using Wren
using Wren.Dists: Gamma, Poisson
using OptimizationOptimJL: LBFGS
import Enzyme, Optimization

@wren function poisson_rate(count)
    rate ~ Gamma(2.0, 1.0)
    @observe count ~ Poisson(rate)
end

model = poisson_rate(3)
fit = Wren.optimize(model, LBFGS())
```

The fitted parameters are returned as a `VarNamedTuple`; here the MAP estimate of `rate` is approximately 2:

```@example optimization
fit.params
```

Use `lp_function=loglikelihood` for maximum likelihood (MLE):

```@example optimization
fit = Wren.optimize(model, LBFGS(); lp_function=loglikelihood)
fit.params
```

Both methods optimise within the parameter support declared by the model's priors.
Parameters use unconstrained coordinates internally, but the objective is the original-space
density without a change-of-variables Jacobian. Solver keywords are passed to
`Optimization.solve`; check `fit.solution.retcode` before using an estimate.

## Convenience functions

[`MAP`](@ref) selects `lp_function=logjoint`, as in the first example.
[`MLE`](@ref) selects `lp_function=loglikelihood`, as in the second:

```@example optimization
map_fit = MAP(model, LBFGS())
mle_fit = MLE(model, LBFGS())
(map=map_fit.params, mle=mle_fit.params)
```

Both helpers fix the objective and forward all other keywords to [`optimize`](@ref),
including initialisation and solver options. Use `optimize` for a custom objective.

## Initialisation

By default, optimisation starts from a draw from the model's priors.
Use `latent_source` to provide an initial state, as for [`VecModel`](@ref):

```@example optimization
initial = FromValues(VarNamedTuple(rate=1.5))
fit = Wren.optimize(model, LBFGS(); latent_source=initial)
fit.params
```

## Direct log-density access

For direct control, construct a [`VecModel`](@ref) and use the LogDensityProblems.jl interface:

```@example optimization
import LogDensityProblems as LDP

density = VecModel(model, AutoEnzyme(; mode=Reverse); latent_source=initial)
x = vector_parameters(density, initial)
LDP.logdensity(density, x)
LDP.logdensity_and_gradient(density, x)
```

The default is the unnormalised posterior density in unconstrained coordinates, including
the Jacobian used by samplers. For a custom optimisation problem, pass `lp_function=logjoint`
for the original-space MAP objective or `lp_function=loglikelihood` for MLE. Pass `nothing`
as the AD backend when only density values are needed. This interface does not require Optimization.jl.
