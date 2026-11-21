"""
    optimize(model, solver; adtype=DEFAULT_AD, latent_source=FromPrior(),
             lp_function=logjoint, reduce_copies=true, kwargs...)

Find a maximum a posteriori (MAP) estimate with an Optimization.jl solver. Requires
Optimization.jl and the selected AD backend to be loaded, as for [`VecModel`](@ref).
Use `lp_function=loglikelihood` for maximum likelihood (MLE).
Both objectives use the latent support declared by the model's priors.

Returns `(params, solution)`: named model-space parameters and the native Optimization
solution, including its convergence status. The solver minimises the negative original-space
log-density in unconstrained coordinates. Initialisation and AD preparation use [`VecModel`](@ref).
Solver keywords are forwarded to `Optimization.solve`.
"""
function optimize end

"""
    MAP(model, solver; kwargs...)

Find the original-space MAP estimate with [`optimize`](@ref), fixing the objective to
`logjoint`. Forward all other keywords to `optimize`. Requires Optimization.jl.
"""
MAP(model::Model, solver; kwargs...) =
    optimize(model, solver; kwargs..., lp_function=logjoint)

"""
    MLE(model, solver; kwargs...)

Find the MLE with [`optimize`](@ref), fixing the objective to `loglikelihood`.
The model's priors still define the parameter support. Forward all other keywords
to `optimize`. Requires Optimization.jl.
"""
MLE(model::Model, solver; kwargs...) =
    optimize(model, solver; kwargs..., lp_function=loglikelihood)
