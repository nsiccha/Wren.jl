module WrenCoreFlexiChainsExt

using FlexiChains: FlexiChains, VNChain, FlexiChain, Parameter, Extra, ParameterOrExtra
using OrderedCollections: OrderedDict
using Random: AbstractRNG, default_rng
using StatsAPI: StatsAPI
using VarNames: VarNames, VarName, VarNamedTuple, parameters
using WrenCore: WrenCore

"""
    WrenCore.to_flexichain(
        results::AbstractMatrix{<:ModelResult},
        stats::AbstractMatrix{<:NamedTuple}=fill((;), size(results));
        iter_indices::AbstractVector{Int}=1:size(results, 1),
        chain_indices::AbstractVector{Int}=1:size(results, 2),
        sampling_time::AbstractVector{<:Union{Real,Missing}}=fill(missing, size(results, 2)),
        last_sampler_state::AbstractVector=fill(missing, size(results, 2)),
    ) -> FlexiChains.VNChain

Convert a matrix of `WrenCore.ModelResult`s, and optionally a matrix of `NamedTuple`
statistics, into a `FlexiChains.VNChain`.

The other keyword arguments are passed directly to the `FlexiChains.VNChain` constructor.
"""
function WrenCore.to_flexichain(
    results::AbstractMatrix{<:WrenCore.ModelResult},
    stats::AbstractMatrix{<:NamedTuple}=fill((;), size(results));
    iter_indices::AbstractVector{Int}=1:size(results, 1),
    chain_indices::AbstractVector{Int}=1:size(results, 2),
    sampling_time::AbstractVector{<:Union{Real,Missing}}=fill(missing, size(results, 2)),
    last_sampler_state::AbstractVector=fill(missing, size(results, 2)),
)
    niters, nchains = size(results)
    # The `FlexiChain` constructor checks the sizes of the other arguments.
    if size(stats) != size(results)
        msg = "expected `stats` to have size $(size(results)), but got $(size(stats))."
        throw(DimensionMismatch(msg))
    end

    dicts = Matrix{OrderedDict{ParameterOrExtra{<:VarName},Any}}(undef, niters, nchains)
    structures = Matrix{VarNamedTuple}(undef, niters, nchains)
    for (i, (result, stat)) in enumerate(zip(results, stats))
        # Parameters and tracked quantities. If a tracked quantity has the same name as
        # a parameter, the parameter takes precedence.
        params = VarNames.densify!!(
            merge(WrenCore.tracked_quantities(result), parameters(result)),
        )

        d = OrderedDict{ParameterOrExtra{<:VarName},Any}(
            Parameter(vn) => val for (vn, val) in pairs(params)
        )
        # Stats
        for (stat_vn, stat_val) in pairs(stat)
            d[Extra(stat_vn)] = stat_val
        end
        # Return value and logps
        d[Extra(:return_value)] = WrenCore.return_value(result)
        d[Extra(:logprior)] = WrenCore.logprior(result)
        d[Extra(:loglikelihood)] = WrenCore.loglikelihood(result)
        d[Extra(:logjoint)] = WrenCore.logjoint(result)
        dicts[i] = d
        structures[i] = VarNames.skeleton(params)
    end
    return FlexiChain{VarName}(
        niters,
        nchains,
        dicts;
        structures=structures,
        iter_indices=iter_indices,
        chain_indices=chain_indices,
        sampling_time=sampling_time,
        last_sampler_state=last_sampler_state,
    )
end

"""
    WrenCore.to_flexichain(
        vnts::AbstractMatrix{<:VarNamedTuple};
        iter_indices::AbstractVector{Int}=1:size(vnts, 1),
        chain_indices::AbstractVector{Int}=1:size(vnts, 2),
    ) -> FlexiChains.VNChain

Convert a matrix of `VarNamedTuple`s into a `FlexiChains.VNChain` holding each entry as
parameters, without any extras.
"""
function WrenCore.to_flexichain(
    vnts::AbstractMatrix{<:VarNamedTuple};
    iter_indices::AbstractVector{Int}=1:size(vnts, 1),
    chain_indices::AbstractVector{Int}=1:size(vnts, 2),
)
    dense_vnts = map(VarNames.densify!!, vnts)
    dicts = map(dense_vnts) do vnt
        OrderedDict{ParameterOrExtra{<:VarName},Any}(
            Parameter(vn) => val for (vn, val) in pairs(vnt)
        )
    end
    return FlexiChain{VarName}(
        size(vnts)...,
        dicts;
        structures=map(VarNames.skeleton, dense_vnts),
        iter_indices,
        chain_indices,
    )
end

"""
    simulate(
        model::WrenCore.Model,
        chain::FlexiChains.VNChain;
        rng::Random.AbstractRNG=Random.default_rng(),
        track::TrackedQuantitySpec=AllQuantities(),
    ) -> FlexiChains.VNChain

Simulate from `model` at every iteration in `chain`. Variables that `chain` does not hold
are drawn from the prior using `rng`.

The returned `FlexiChain` will contain all the simulated variables, the tracked quantities
selected by `track`, as well as the variables already present in the input `chain`, which
implies that:

- variables that are in the `chain` are fixed to their respective values;
- variables that arent in the `chain` are resampled according to the model;
- variables that are `@observe`d are never included.

To move a variable from the first into the second category, subset `chain` to exclude that
variable (more about subsetting chains
[here](https://juliabayes.org/FlexiChains.jl/stable/api#Subsetting-chains)).

The returned chain will also contain log-probabilities corresponding to the re-evaluation of
the model. In particular, the log probability for the newly simulated variables are now
considered as prior terms. However, note that the log-prior of the returned chain will also
contain the log-prior terms of the parameters already present in the input `chain`. Thus, if
you want to obtain the log-probability of the simulated variables only, you can subtract the
two log-prior terms.

See also: [`TrackedQuantitySpec`](@ref WrenCore.TrackedQuantitySpec)
"""
function WrenCore.simulate(
    model::WrenCore.Model,
    chain::VNChain;
    rng::AbstractRNG=default_rng(),
    track::WrenCore.TrackedQuantitySpec=WrenCore.AllQuantities(),
)
    fallback = WrenCore.FromPrior(rng)
    results = map(FlexiChains.parameters_at(chain)) do vnt
        WrenCore.simulate(model, WrenCore.FromValues(vnt, fallback); track)
    end
    simulated_chain = WrenCore.to_flexichain(
        results;
        iter_indices=FlexiChains.iter_indices(chain),
        chain_indices=FlexiChains.chain_indices(chain),
    )
    return merge(FlexiChains.subset_extras(chain), simulated_chain)
end

"""
    tracked_quantities(
        model::WrenCore.Model,
        chain::FlexiChains.VNChain;
        rng::Random.AbstractRNG=Random.default_rng(),
        track::TrackedQuantitySpec=AllQuantities(),
    ) -> FlexiChains.VNChain

Re-evaluate `model` at each iteration of `chain` and return a chain holding only the
tracked quantities requested with `track`. If a variable required by `model` is not stored
in `chain`, that variable will be drawn from the prior distribution.

See also: [`TrackedQuantitySpec`](@ref WrenCore.TrackedQuantitySpec)
"""
function WrenCore.tracked_quantities(
    model::WrenCore.Model,
    chain::VNChain;
    rng::AbstractRNG=default_rng(),
    track::WrenCore.TrackedQuantitySpec=WrenCore.AllQuantities(),
)
    fallback = WrenCore.FromPrior(rng)
    tq_matrix = map(FlexiChains.parameters_at(chain)) do vnt
        WrenCore.tracked_quantities(model, WrenCore.FromValues(vnt, fallback), track)
    end
    return WrenCore.to_flexichain(
        tq_matrix;
        iter_indices=FlexiChains.iter_indices(chain),
        chain_indices=FlexiChains.chain_indices(chain),
    )
end

"""
    logprior(
        model::WrenCore.Model,
        chain::FlexiChains.VNChain
    ) -> DimensionalData.DimMatrix

Return the log-prior of `model` at each iteration of `chain`. All variables required by
`model` must be stored in `chain`.
"""
function WrenCore.logprior(model::WrenCore.Model, chain::VNChain)
    return map(vnt -> WrenCore.logprior(model, vnt), FlexiChains.parameters_at(chain))
end

"""
    loglikelihood(
        model::WrenCore.Model,
        chain::FlexiChains.VNChain
    ) -> DimensionalData.DimMatrix

Return the log-likelihood of `model` at each iteration of `chain`. All variables required by
`model` must be stored in `chain`.
"""
function StatsAPI.loglikelihood(model::WrenCore.Model, chain::VNChain)
    return map(vnt -> StatsAPI.loglikelihood(model, vnt), FlexiChains.parameters_at(chain))
end

"""
    logjoint(
        model::WrenCore.Model,
        chain::FlexiChains.VNChain
    ) -> DimensionalData.DimMatrix

Return the log-joint (log-prior plus log-likelihood) of `model` at each iteration of
`chain`. All variables required by `model` must be stored in `chain`.
"""
function WrenCore.logjoint(model::WrenCore.Model, chain::VNChain)
    return map(vnt -> WrenCore.logjoint(model, vnt), FlexiChains.parameters_at(chain))
end

"""
    return_value(
        model::WrenCore.Model,
        chain::FlexiChains.VNChain
    ) -> DimensionalData.DimMatrix

Return the return value of `model` at each iteration of `chain`. All variables required by
`model` must be stored in `chain`.
"""
function WrenCore.return_value(model::WrenCore.Model, chain::VNChain)
    return map(vnt -> WrenCore.return_value(model, vnt), FlexiChains.parameters_at(chain))
end

"""
    parameters(
        model::WrenCore.Model,
        chain::FlexiChains.VNChain
    ) -> FlexiChains.VNChain

Return the parameters of `model` at each iteration of `chain`. If a variable required by
`model` is not stored in `chain`, that variable will be drawn from the prior distribution.

This function differs from [`simulate`](@ref) in that it contains _only_ parameters. That
makes it more efficient than `simulate`, which also computes the log-density terms.
"""
function VarNames.parameters(
    model::WrenCore.Model,
    chain::VNChain;
    rng::AbstractRNG=default_rng(),
)
    fallback = WrenCore.FromPrior(rng)
    param_matrix = map(FlexiChains.parameters_at(chain)) do vnt
        parameters(model, WrenCore.FromValues(vnt, fallback))
    end
    return WrenCore.to_flexichain(
        param_matrix;
        iter_indices=FlexiChains.iter_indices(chain),
        chain_indices=FlexiChains.chain_indices(chain),
    )
end

end # module WrenCoreFlexiChainsExt
