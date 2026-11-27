using Random: Random
import DifferentiationInterface as DI
using EnzymeCore: Reverse
using FlexiChains: VNChain
using DynamicHMC: DynamicHMC
using LogDensityProblems: LogDensityProblems
using MCMCProgress:
    MCMCProgress, RunProgress, Determinate, advance!, chain_at, close_phase!, open_phase!

mutable struct ChainReporter
    const run::RunProgress
    const chain_index::Int
    phase::Union{MCMCProgress.Phase,Nothing}
end

ChainReporter(run::RunProgress, chain_index::Integer) =
    ChainReporter(run, Int(chain_index), nothing)

function DynamicHMC.make_mcmc_reporter(
    reporter::ChainReporter,
    total_steps::Integer;
    currently_warmup::Bool=false,
    meta...,
)
    close_phases!(reporter)
    reporter.phase = open_phase!(
        chain_at(reporter.run, reporter.chain_index),
        currently_warmup ? "Warmup" : "Sampling",
        Determinate(total_steps),
    )
    return reporter
end

DynamicHMC.report(reporter::ChainReporter, step::Integer; meta...) =
    advance!(reporter.phase, step)

DynamicHMC.report(::ChainReporter, ::AbstractString; meta...) = nothing

function close_phases!(reporter::ChainReporter)
    reporter.phase === nothing || close_phase!(reporter.phase)
    reporter.phase = nothing
    return nothing
end
close_phases!(::DynamicHMC.NoProgressReport) = nothing

"""
    GradientCounter(ℓ)

Wrap the log density `ℓ` of one chain and count every call of
`LogDensityProblems.logdensity_and_gradient` on it, which is how DynamicHMC evaluates
gradients. The count includes the initial point, the initial step-size search and warm-up.
"""
mutable struct GradientCounter{L}
    const ℓ::L
    n::Int
end
GradientCounter(ℓ) = GradientCounter(ℓ, 0)

LogDensityProblems.dimension(c::GradientCounter) = LogDensityProblems.dimension(c.ℓ)
LogDensityProblems.capabilities(::Type{GradientCounter{L}}) where {L} =
    LogDensityProblems.capabilities(L)
LogDensityProblems.logdensity(c::GradientCounter, q) = LogDensityProblems.logdensity(c.ℓ, q)
function LogDensityProblems.logdensity_and_gradient(c::GradientCounter, q)
    c.n += 1
    return LogDensityProblems.logdensity_and_gradient(c.ℓ, q)
end

"""
    nuts(
        [rng::Random.AbstractRNG=default_rng()],
        model::Model,
        niters::Integer,
        nchains::Integer=1;
        adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Enzyme.Reverse),
        progress::Bool=true,
        reduce_copies::Bool=true,
        track=AllQuantities(),
        init=nothing,
        initial_stepsize::Union{Nothing,Real}=nothing,
        dynamichmc_kwargs...
    ) -> FlexiChains.VNChain

No U-Turn Sampler (NUTS) for a Wren model. All `dynamichmc_kwargs` are passed to
[`DynamicHMC.mcmc_with_warmup`](@extref), allowing you to customise the warmup and sampling
behaviour.

Each chain starts from an independent prior draw by default. Use `init` to select an
[`AbstractInitialization`](@ref) policy. Anything else passed as `init`, such as a
`VarNamedTuple` or a latent source, is treated as `FindFiniteDensity(init)`.
`initial_stepsize` replaces the default initial step-size search.

`reduce_copies` is passed to [`VecModel`](@ref): when sampling with Enzyme, it skips copies
of model arguments that turn out not to be needed (see [`reduce_copies`](@ref)).

By default, all tracked quantities will be included in the returned chain. To include only
one or a selection of tracked quantities, pass a Symbol or a tuple of Symbols; to include
none, pass `track=false` or an empty tuple. Note that tracked quantities are not evaluated
during the log-density step, but only once per iteration.

Each iteration also stores these sampler statistics as extras: `step_size`, `tree_depth`,
`n_steps`, `is_divergent`, `is_max_depth`, `acceptance_rate`, `log_density` and
`n_gradient_evaluations`. `n_steps` is the number of leapfrog steps of that transition, each
of which evaluates the gradient of the log density once. `n_gradient_evaluations` is the
running total of gradient evaluations of that chain up to and including that iteration,
counted from the chain's first evaluation, so it includes the initial point, the initial
step-size search and warm-up. The gradient evaluations made while preparing the
[`VecModel`](@ref), once per call and before any chain starts, are not included.
"""
function nuts(
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Reverse),
    progress::Bool=true,
    reduce_copies::Bool=true,
    track=AllQuantities(),
    init=nothing,
    initial_stepsize::Union{Nothing,Real}=nothing,
    dynamichmc_kwargs...,
)
    return nuts(
        Random.default_rng(),
        model,
        niters,
        nchains;
        adtype,
        progress,
        reduce_copies,
        track,
        init,
        initial_stepsize,
        dynamichmc_kwargs...,
    )
end
function nuts(
    rng::Random.AbstractRNG,
    model::Model,
    niters::Integer,
    nchains::Integer=1;
    adtype::DI.AbstractADType=DI.AutoEnzyme(; mode=Reverse),
    progress::Bool=true,
    reduce_copies::Bool=true,
    track=AllQuantities(),
    init=nothing,
    initial_stepsize::Union{Nothing,Real}=nothing,
    dynamichmc_kwargs...,
)
    haskey(dynamichmc_kwargs, :initialization) && throw(
        ArgumentError(
            "Use `init` for starting points and `initial_stepsize` for the step size.",
        ),
    )
    tq_spec = WrenCore.to_tq_spec(track)
    initial_stepsize = initial_stepsize === nothing ? nothing : Float64(initial_stepsize)
    init = init === nothing ? nothing : WrenCore.to_initialization(init)
    source = init === nothing ? WrenCore.FromPrior() : WrenCore._initial_source(init)
    # AD preparation
    if progress
        t = time()
        print("Doing AD preparation... ")
    end
    vm_single = WrenCore.VecModel(model, adtype; reduce_copies, latent_source=source)
    if progress
        println("done in $(round(time() - t, digits=2)) seconds.")
        t = time()
    end

    Random.seed!(Random.TaskLocalRNG(), rand(rng, UInt64))

    # Sources can contain mutable RNGs, so draw initial points before spawning tasks.
    starts =
        init === nothing ? nothing :
        [WrenCore._initial_position(vm_single, init) for _ in 1:nchains]

    t = time()
    label = "Sampling $(model.name)"
    outputs = with_progress(progress, label, nchains) do run
        tasks = @sync [
            Threads.@spawn _nuts_chain(
                vm_single,
                niters,
                run,
                j;
                track=tq_spec,
                q=starts === nothing ? nothing : starts[j],
                initial_stepsize,
                dynamichmc_kwargs...,
            ) for j in 1:nchains
        ]
        fetch.(tasks)
    end

    if progress
        println("Sampling done in $(round(time() - t, digits=2)) seconds.")
        t = time()
        print("Constructing chain... ")
    end
    model_results = stack(first.(outputs))
    stats = stack(last.(outputs))
    chn = WrenCore.to_flexichain(model_results, stats)
    if progress
        println("took $(round(time() - t, digits=2)) seconds")
    end
    return chn
end

function _nuts_chain(
    vm_single::WrenCore.VecModel,
    niters::Integer,
    run,
    j::Integer;
    track::WrenCore.TrackedQuantitySpec,
    q,
    initial_stepsize,
    warmup_stages=initial_stepsize === nothing ? DynamicHMC.default_warmup_stages() :
                  DynamicHMC.default_warmup_stages(; stepsize_search=nothing),
    dynamichmc_kwargs...,
)
    vm = WrenCore._copy_for_thread(vm_single)
    rng = Random.TaskLocalRNG()
    if q === nothing
        q = WrenCore.vector_parameters(vm, WrenCore.FromPrior(rng))
    end
    reporter = run === nothing ? DynamicHMC.NoProgressReport() : ChainReporter(run, j)
    counter = GradientCounter(vm)
    results = DynamicHMC.mcmc_with_warmup(
        rng,
        counter,
        niters;
        reporter,
        initialization=(; q, ϵ=initial_stepsize),
        warmup_stages,
        dynamichmc_kwargs...,
    )
    close_phases!(reporter)
    return _nuts_results_and_stats(results, vm, track, counter.n)
end

function _nuts_stats(tree_stat, step_size, n_gradient_evaluations)
    t = tree_stat.termination
    return (;
        step_size,
        tree_depth=tree_stat.depth,
        n_steps=tree_stat.steps,
        is_divergent=DynamicHMC.is_divergent(t),
        is_max_depth=(t == DynamicHMC.REACHED_MAX_DEPTH),
        acceptance_rate=tree_stat.acceptance_rate,
        log_density=tree_stat.π,
        n_gradient_evaluations,
    )
end
function _nuts_results_and_stats(
    results::NamedTuple,
    vecmodel::WrenCore.VecModel,
    track::WrenCore.TrackedQuantitySpec,
    n_total::Integer,
)
    ϵ = results.ϵ
    model_results = [
        WrenCore.simulate(vecmodel, col; track) for col in eachcol(results.posterior_matrix)
    ]
    # Each leapfrog step evaluates the gradient exactly once, so the retained draws used
    # `sum(steps)` of the `n_total` evaluations, and everything before them used the rest.
    steps = map(ts -> ts.steps, results.tree_statistics)
    cumulative = (n_total - sum(steps; init=0)) .+ cumsum(steps)
    stats = map(results.tree_statistics, cumulative) do ts, n
        _nuts_stats(ts, ϵ, n)
    end
    return model_results, stats
end
