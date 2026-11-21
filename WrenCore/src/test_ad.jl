using Chairmarks: Chairmarks
using LinearAlgebra: norm
using Random: Xoshiro

"""
    AbstractADCorrectnessTestSetting

Different ways to check the correctness of an AD backend in [`test_ad`](@ref).
"""
abstract type AbstractADCorrectnessTestSetting end

"""
    WithBackend(adtype=AutoForwardDiff())

Check correctness against the log density and gradient produced by another AD backend.
"""
struct WithBackend{A<:DI.AbstractADType} <: AbstractADCorrectnessTestSetting
    adtype::A
end
WithBackend() = WithBackend(DI.AutoForwardDiff())

"""
    WithExpectedResult(lp, grad)

Check correctness against a known log density and gradient. Either can be left as `nothing`
to skip testing.
"""
struct WithExpectedResult{V<:Union{Number,Nothing},G<:Union{AbstractVector,Nothing}} <:
       AbstractADCorrectnessTestSetting
    lp::V
    grad::G
end

# An expected value of `nothing` means that the check is skipped.
_check_passed(::Nothing, actual, atol, rtol) = true
_check_passed(expected, actual, atol, rtol) = isapprox(expected, actual; atol, rtol)

"""
    ADIncorrectException

Exception thrown when an AD backend returns an incorrect log-density or gradient.
"""
struct ADIncorrectException{VE,VA,GE,GA,T<:Real} <: Exception
    lp_expected::VE
    lp_actual::VA
    grad_expected::GE
    grad_actual::GA
    atol::T
    rtol::T
end
function Base.showerror(io::IO, exception::ADIncorrectException)
    lp_passed = _check_passed(
        exception.lp_expected,
        exception.lp_actual,
        exception.atol,
        exception.rtol,
    )
    grad_passed = _check_passed(
        exception.grad_expected,
        exception.grad_actual,
        exception.atol,
        exception.rtol,
    )
    failed = if !lp_passed && !grad_passed
        "lp and gradient"
    elseif !lp_passed
        "lp"
    else
        "gradient"
    end

    msg = """
    ADIncorrectException: incorrect $failed
      ├ atol     : $(exception.atol)
      ├ rtol     : $(exception.rtol)
      ├ expected : $((exception.lp_expected, exception.grad_expected))
      └ actual   : $((exception.lp_actual, exception.grad_actual))
    """
    print(io, msg)
    if !lp_passed
        min_atol = abs(exception.lp_expected - exception.lp_actual)
        min_rtol = min_atol / max(abs(exception.lp_expected), abs(exception.lp_actual))
        s = """
          lp check would have passed with:
               atol ≥ $min_atol
            or rtol ≥ $min_rtol
        """
        println(io)
        print(io, s)
    end
    if !grad_passed
        diff_norm = norm(exception.grad_expected - exception.grad_actual)
        max_norm = max(norm(exception.grad_expected), norm(exception.grad_actual))
        s = """
          Gradient check would have passed with:
               atol ≥ $diff_norm
            or rtol ≥ $(diff_norm / max_norm)
        """
        println(io)
        print(io, s)
    end
    return nothing
end

# Prints `title` followed by `lines` as a tree.
function _print_tree(io::IO, title::String, lines::Vector{Pair{String,String}})
    println(io, title)
    for (i, (key, val)) in enumerate(lines)
        connector = i == length(lines) ? "└" : "├"
        padded_key = rpad(key, 12)
        if i == length(lines)
            print(io, "  $connector $padded_key: $val")
        else
            println(io, "  $connector $padded_key: $val")
        end
    end
    return nothing
end

# Lines describing the model and vectorised model, shared by both result types.
function _model_lines(vecmodel::VecModel)
    lines = Pair{String,String}["model"=>string(vecmodel.model.name)]
    if vecmodel.adtype !== nothing
        push!(lines, "backend" => string(vecmodel.adtype))
    end
    push!(lines, "n_params" => string(length(vecmodel.sample)))
    return lines
end

"""
    ADTestResult

The correctness results returned by [`test_ad`](@ref).

$(TYPEDFIELDS)
"""
struct ADTestResult{M<:VecModel,VE,GE,VA,GA,T<:Real}
    "The vectorised model that was tested. The parameters used are in `vecmodel.sample`."
    vecmodel::M
    "The expected log-density value, or `nothing` if it was not checked."
    lp_expected::VE
    "The expected gradient of the log-density, or `nothing` if it was not checked."
    grad_expected::GE
    "The actual log-density value returned by the AD backend."
    lp_actual::VA
    "The actual gradient of the log-density returned by the AD backend."
    grad_actual::GA
    "The absolute tolerance used for correctness checking."
    atol::T
    "The relative tolerance used for correctness checking."
    rtol::T
end

function Base.show(io::IO, ::MIME"text/plain", result::ADTestResult)
    lines = _model_lines(result.vecmodel)
    push!(lines, "correctness" => "passed")
    _print_tree(io, "ADTestResult", lines)
    return nothing
end

"""
    test_ad(
        model::Model,
        adtype::DI.AbstractADType;
        reference::AbstractADCorrectnessTestSetting=WithBackend(),
        atol::Real=100 * eps(),
        rtol::Real=sqrt(eps()),
        lp_function=logjoint_transformed,
        latent_source::AbstractLatentSource=FromPrior(),
        parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
        verbose::Bool=(!isinteractive()),
    ) -> ADTestResult

Check that the log-density and gradient computed by `adtype` on `model` are correct, and
throw an [`ADIncorrectException`](@ref) if not.

Note that the corresponding backend package for `adtype` must be loaded first.

`reference` determines what the results are compared against: either another backend
([`WithBackend`](@ref)) or known values ([`WithExpectedResult`](@ref)).

The `lp_function`, `latent_source`, and `parameter_space` keywords are forwarded to
[`VecModel`](@ref), and have the same defaults.
"""
function test_ad(
    model::Model,
    adtype::DI.AbstractADType;
    reference::AbstractADCorrectnessTestSetting=WithBackend(),
    atol::Real=100 * eps(),
    rtol::Real=sqrt(eps()),
    lp_function=logjoint_transformed,
    latent_source::AbstractLatentSource=FromPrior(),
    parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
    verbose::Bool=(!isinteractive()),
)
    if reference isa WithBackend && adtype == reference.adtype
        @warn "Testing correctness of $adtype against itself — results will always pass. " *
              "Use a different reference backend or WithExpectedResult."
    end
    atol, rtol = promote(atol, rtol)

    vecmodel = VecModel(model, adtype; latent_source, lp_function, parameter_space)
    params = vecmodel.sample

    lp_actual, grad_actual = lp_and_grad(vecmodel, params)
    # standardise the array type returned. DI is agnostic towards this.
    # https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/754
    grad_actual = collect(grad_actual)

    lp_expected, grad_expected = if reference isa WithExpectedResult
        reference.lp, reference.grad === nothing ? nothing : collect(reference.grad)
    else
        # The expected lp is computed without any backend. This usually doesn't really
        # need to be tested, except for Reactant where the computation graph is different.
        no_ad_vecmodel = VecModel(
            vecmodel.model,
            nothing,
            vecmodel.lp_function,
            vecmodel.parameter_space,
            vecmodel.vecinfo,
            vecmodel.sample,
        )
        ref_vecmodel = VecModel(
            vecmodel.model,
            reference.adtype,
            vecmodel.lp_function,
            vecmodel.parameter_space,
            vecmodel.vecinfo,
            vecmodel.sample,
        )
        _, grad = lp_and_grad(ref_vecmodel, params)
        lp(no_ad_vecmodel, params), collect(grad)
    end

    lp_passed = _check_passed(lp_expected, lp_actual, atol, rtol)
    grad_passed = _check_passed(grad_expected, grad_actual, atol, rtol)
    if !lp_passed || !grad_passed
        throw(
            ADIncorrectException(
                lp_expected,
                lp_actual,
                grad_expected,
                grad_actual,
                atol,
                rtol,
            ),
        )
    end

    result = ADTestResult(
        vecmodel,
        lp_expected,
        grad_expected,
        lp_actual,
        grad_actual,
        atol,
        rtol,
    )
    if verbose
        show(stdout, MIME"text/plain"(), result)
        println()
    end
    return result
end

"""
    reduce_copies(
        model::Model,
        adtype::DI.AbstractADType;
        n_points::Int=10,
        rng::AbstractRNG=Xoshiro(468),
        atol::Real=100 * eps(),
        rtol::Real=sqrt(eps()),
        lp_function=logjoint_transformed,
        latent_source::AbstractLatentSource=FromPrior(),
        parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
        verbose::Bool=true,
    ) -> Model

Return a copy of `model` which skips copying any arguments whose copies turn out not to
affect the log density or gradient computed with `adtype`.

The copies are only made when differentiating with Enzyme (see [the docs](@ref
enzyme-copyargs)), so for any other `adtype` this returns `model` unchanged.
"""
function reduce_copies(
    model::Model{A,C,F,Val{S}},
    ::Union{DI.AbstractADType,Nothing};
    kwargs...,
) where {A,C,F,S}
    model
end
function reduce_copies(
    model::Model{A,C,F,Val{S}},
    adtype::DI.AutoEnzyme;
    n_points::Int=10,
    rng::AbstractRNG=Xoshiro(468),
    atol::Real=100 * eps(),
    rtol::Real=sqrt(eps()),
    lp_function=logjoint_transformed,
    latent_source::AbstractLatentSource=FromPrior(),
    parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
    verbose::Bool=true,
) where {A,C,F,S}
    atol, rtol = promote(atol, rtol)
    function lp_and_collected_grad(vecmodel, params)
        lp, grad = lp_and_grad(vecmodel, params)
        # standardise the array type returned, as in `test_ad`.
        return lp, collect(grad)
    end

    ref_vecmodel = VecModel(
        model,
        adtype;
        latent_source,
        lp_function,
        parameter_space,
        # `reduce_copies` must be false here, otherwise infinite recursion!
        reduce_copies=false,
    )
    points = [vector_parameters(ref_vecmodel, rng) for _ in 1:n_points]
    expected = [lp_and_collected_grad(ref_vecmodel, p) for p in points]

    # Arguments in the copy set are only really copied if `_needs_copy` says so, so
    # skipping the others would make no difference.
    candidates = filter(k -> _needs_copy(typeof(model.args[k])), S)

    # Try removing each candidate in turn. Each attempt keeps the removals that already
    # passed, so this doesn't rely on arguments being independent of each other (for
    # example, two arguments might share memory).
    #
    # TODO(penelopeysm): Each attempt is a new model type, so Enzyme has to compile a new
    # gradient for it (roughly 1 second each on the posteriordb models). This could be made
    # cheaper by first trying to remove all candidates at once, and if that fails, splitting
    # the candidates in half and recursing (like a binary search). Usually almost all copies
    # can be removed, so the first attempt would often be the only one needed.
    copy_argnames = S
    removed = Symbol[]
    for k in candidates
        trial_argnames = filter(!=(k), copy_argnames)
        trial_vecmodel = VecModel(
            _with_copied_args(model, trial_argnames),
            adtype,
            ref_vecmodel.lp_function,
            ref_vecmodel.parameter_space,
            ref_vecmodel.vecinfo,
            ref_vecmodel.sample,
        )
        passed = all(zip(points, expected)) do (params, (lp_expected, grad_expected))
            lp_actual, grad_actual = lp_and_collected_grad(trial_vecmodel, params)
            return _check_passed(lp_expected, lp_actual, atol, rtol) &&
                   _check_passed(grad_expected, grad_actual, atol, rtol)
        end
        if passed
            copy_argnames = trial_argnames
            push!(removed, k)
        end
    end

    if verbose
        if isempty(removed)
            @info "reduce_copies: no copies could be skipped"
        else
            @info "reduce_copies: these arguments no longer need to be copied: $(join(removed, ", "))"
        end
    end
    return _with_copied_args(model, copy_argnames)
end

"""
    BenchmarkResult

The benchmark results returned by [`benchmark_model`](@ref) and [`@bmodel`](@ref). Each
benchmark is the fastest `Chairmarks.Sample` for the function it is named after.

$(TYPEDFIELDS)
"""
struct BenchmarkResult{M<:VecModel}
    "The vectorised model that was benchmarked. The parameters used are in `vecmodel.sample`."
    vecmodel::M
    "Benchmark for drawing a sample from the model with [`parameters`](@ref)."
    parameters_bench::Chairmarks.Sample
    "Benchmark for evaluating the log-joint density with [`logjoint`](@ref)."
    logjoint_bench::Chairmarks.Sample
    "Benchmark for evaluating the vectorised log-density with [`lp`](@ref)."
    lp_bench::Chairmarks.Sample
    "Benchmark for [`lp_and_grad`](@ref), or `nothing` if no AD backend was given."
    lp_and_grad_bench::Union{Nothing,Chairmarks.Sample}
end

_show_sample(s::Chairmarks.Sample) = sprint(show, MIME"text/plain"(), s)

# Benchmarks `f()` for roughly `seconds` and returns the fastest sample. The fastest sample
# is the one least affected by other things running on the machine, so it varies less
# between runs than the median. The garbage collector handling follows DynamicPPL's `run_ad`
# (which in turn follows Mooncake's benchmarks).
function _min_bench(f, seconds::Real)
    f() # warmup
    GC.gc(true)
    # Incremental collection after every sample.
    return minimum(Chairmarks.benchmark(nothing, f, _ -> GC.gc(false); seconds))
end

function Base.show(io::IO, ::MIME"text/plain", result::BenchmarkResult)
    lines = _model_lines(result.vecmodel)
    push!(lines, "parameters" => _show_sample(result.parameters_bench))
    push!(lines, "logjoint" => _show_sample(result.logjoint_bench))
    push!(lines, "lp" => _show_sample(result.lp_bench))
    if result.lp_and_grad_bench !== nothing
        push!(lines, "lp_and_grad" => _show_sample(result.lp_and_grad_bench))
        ratio = result.lp_and_grad_bench.time / result.lp_bench.time
        push!(lines, "grad / lp" => string(round(ratio; sigdigits=4)))
    end
    _print_tree(io, "BenchmarkResult", lines)
    return nothing
end

"""
    benchmark_model(
        model::Model,
        adtype::Union{Nothing,DI.AbstractADType}=nothing;
        lp_function=logjoint_transformed,
        latent_source::AbstractLatentSource=FromPrior(Xoshiro(468)),
        parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
        reduce_copies::Bool=true,
        seconds::Real=1,
        verbose::Bool=(!isinteractive()),
    ) -> BenchmarkResult

    benchmark_model(
        vecmodel::VecModel;
        seconds::Real=1,
        verbose::Bool=(!isinteractive()),
    ) -> BenchmarkResult

Benchmark a model at multiple levels:

- **Model level**: [`parameters`](@ref) (sampling from the prior) and [`logjoint`](@ref)
  (log-joint density evaluation).
- **VecModel level**: [`lp`](@ref) (log-density of the vectorised model) and, if `adtype`
  is given, [`lp_and_grad`](@ref) (log-density and its gradient).

Note that if `adtype` is given, the corresponding backend package must be loaded first.

The `lp_function`, `latent_source`, `parameter_space`, and `reduce_copies` keywords are
forwarded to [`VecModel`](@ref). `lp_function`, `parameter_space`, and `reduce_copies` have
the same defaults as for `VecModel`. `latent_source` instead defaults to drawing from the
prior with a fixed seed, so that every call benchmarks at the same parameters. (The time
taken can depend on the parameter values, so random parameters make results vary between
calls.)

`latent_source` is evaluated once, directly in `parameter_space`, and the resulting vector
is the point used for density and gradient evaluation (`result.vecmodel.sample`). Values that
`latent_source` already supplies in that space, such as a
`TransformedValue(v, AsUnconstrainedVec())` in the default `UnconstrainedSpace`, are used
exactly as given rather than converted to the model's original space and back.

The second method benchmarks an existing `VecModel` at `vecmodel.sample`, with the AD
backend it was constructed with. The model-level log-joint benchmark uses the model-space parameters
`parameters(vecmodel, vecmodel.sample)`.

`seconds` is the approximate time spent on each of the benchmarks.

See also [`@bmodel`](@ref).
"""
function benchmark_model(
    model::Model,
    adtype::Union{Nothing,DI.AbstractADType}=nothing;
    lp_function=logjoint_transformed,
    latent_source::AbstractLatentSource=FromPrior(Xoshiro(468)),
    parameter_space::AbstractParameterSpace=UnconstrainedSpace(),
    reduce_copies::Bool=true,
    seconds::Real=1,
    verbose::Bool=(!isinteractive()),
)
    # Draw the benchmark point once and pass it on as a `FromVector`, so that a stateful
    # source such as `FromPrior(rng)` is not drawn from again while `VecModel` is prepared
    # (e.g. by `reduce_copies`). Every AD backend therefore benchmarks the same vector.
    vecinfo, sample = _prepare_vecmodel_info(model, latent_source, parameter_space)
    vecmodel = VecModel(
        model,
        adtype;
        latent_source=FromVector(sample, vecinfo),
        lp_function,
        parameter_space,
        reduce_copies,
    )
    return benchmark_model(vecmodel; seconds, verbose)
end
function benchmark_model(
    vecmodel::VecModel;
    seconds::Real=1,
    verbose::Bool=(!isinteractive()),
)
    model = vecmodel.model
    params = vecmodel.sample
    vnt_params = parameters(vecmodel, params)

    # Model level
    rng = Xoshiro(468) # avoid mutating global RNG
    parameters_bench = _min_bench(() -> parameters(model, rng), seconds)
    logjoint_bench = _min_bench(() -> logjoint(model, vnt_params), seconds)

    # VecModel level
    lp_bench = _min_bench(() -> lp(vecmodel, params), seconds)
    lp_and_grad_bench = if vecmodel.adtype === nothing
        nothing
    else
        _min_bench(() -> lp_and_grad(vecmodel, params), seconds)
    end

    result = BenchmarkResult(
        vecmodel,
        parameters_bench,
        logjoint_bench,
        lp_bench,
        lp_and_grad_bench,
    )
    if verbose
        show(stdout, MIME"text/plain"(), result)
        println()
    end
    return result
end

"""
    @bmodel model
    @bmodel model adtype

Benchmark a model; shorthand for [`benchmark_model`](@ref). The one-argument form
benchmarks `parameters`, `logjoint`, and `lp` only. The two-argument form also benchmarks
`lp_and_grad` with the given AD backend.
"""
macro bmodel(model)
    return :(benchmark_model($(esc(model))))
end
macro bmodel(model, adtype)
    return :(benchmark_model($(esc(model)), $(esc(adtype))))
end
