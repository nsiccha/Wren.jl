using Wren: Wren, VecModel, lp, lp_and_grad, benchmark_model, posteriordb_model
# Loaded so that WrenCore can build the PosteriorDB models.
using PosteriorDB: PosteriorDB
using WrenDists: WrenDists
using DynamicPPL: DynamicPPL
using ADTypes: AutoEnzyme, AutoMooncake, AutoReactant
using Enzyme: Reverse, set_runtime_activity
using Mooncake: Mooncake
using Reactant: Reactant
using Printf: @sprintf
using LogDensityProblems: logdensity, logdensity_and_gradient
using Random: Xoshiro

# BLAS's own threads make small matrix operations (as in GPRegr) much slower in some Julia
# sessions but not others, e.g. a gradient that usually takes 5 μs taking 5–460 μs. One
# BLAS thread makes the results consistent, and the matrices here are too small to benefit
# from more. (Same as in `bench/suite/run.jl`.)
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

# Each entry is either a time in seconds, or an exception that was caught.
const BenchEntry = Union{Real,Exception}

struct BenchResult
    name::String
    # Primals
    wren_primal::BenchEntry
    turing_primal::BenchEntry
    # Gradients
    wren_enzyme::BenchEntry
    wren_mooncake::BenchEntry
    turing_enzyme::BenchEntry
    turing_mooncake::BenchEntry
    # Reactant
    wren_reactant_primal::BenchEntry
    wren_reactant_grad::BenchEntry
end

function _try(f)
    try
        return f()
    catch e
        return e
    end
end

# Errors are shown in the tables as `ERR [k]`, and listed in full after the tables.
const ERRORS = Tuple{String,String,Exception}[]

function _format(x::Exception, base, name, column)
    push!(ERRORS, (name, column, x))
    return "ERR [$(length(ERRORS))]"
end
# `base` is `nothing` if there is nothing to compare against, or an exception if the
# baseline failed; in both cases the ratio is left out.
function _format(x::Real, base::Union{BenchEntry,Nothing}, _name, _column)
    ratio = base isa Real ? " ($(_sigfigs(x / base))x)" : ""
    # Round before picking the unit, so that e.g. 999.7 ns is shown as 1 μs, not 1000 ns.
    x = round(x; sigdigits=SIGFIGS)
    absolute = if x < 1e-6
        "$(_sigfigs(x * 1e9)) ns"
    elseif x < 1e-3
        "$(_sigfigs(x * 1e6)) μs"
    elseif x < 1.0
        "$(_sigfigs(x * 1e3)) ms"
    else
        "$(_sigfigs(x)) s"
    end
    return absolute * ratio
end

# Anything beyond 3 significant figures is just noise.
const SIGFIGS = 3
# `%g` drops trailing zeros, so that e.g. 120.0 is shown as 120.
_sigfigs(x::Real) = @sprintf("%g", round(x; sigdigits=SIGFIGS))

# Each column is a (header, field, baseline field) triple. A baseline of `nothing` means
# the column is not compared against anything.
const PRIMAL_COLUMNS = [
    ("Wren", :wren_primal, nothing),
    ("Turing", :turing_primal, :wren_primal),
    ("Wren+R", :wren_reactant_primal, :wren_primal),
]
const GRAD_COLUMNS = [
    ("Wren+E", :wren_enzyme, nothing),
    ("Wren+M", :wren_mooncake, :wren_enzyme),
    ("Wren+R", :wren_reactant_grad, :wren_enzyme),
    ("Turing+E", :turing_enzyme, :wren_enzyme),
    ("Turing+M", :turing_mooncake, :wren_enzyme),
]

function print_table(io::IO, title, results, columns)
    header = ["Model"; [c[1] for c in columns]]
    rows = map(results) do r
        cells = map(columns) do (column, field, base)
            base_value = base === nothing ? nothing : getfield(r, base)
            _format(getfield(r, field), base_value, r.name, column)
        end
        [r.name; cells]
    end
    widths = [
        maximum(textwidth, [header[j]; [row[j] for row in rows]]) for j in eachindex(header)
    ]
    # Model names are left-aligned, timings right-aligned.
    _pad(s, j) = j == 1 ? rpad(s, widths[j]) : lpad(s, widths[j])
    _line(cells) = "| " * join((_pad(c, j) for (j, c) in enumerate(cells)), " | ") * " |"
    println(io, title)
    println(io, _line(header))
    println(io, "|" * join(("-"^(w + 2) for w in widths), "|") * "|")
    foreach(row -> println(io, _line(row)), rows)
    return println(io)
end

function print_errors(io::IO)
    isempty(ERRORS) && return nothing
    println(io, "Errors:")
    for (k, (name, column, e)) in enumerate(ERRORS)
        msg = join(Base.split(sprint(showerror, e), "\n"), "; ")
        # Truncate long messages
        MAX_LEN = 200
        if length(msg) > MAX_LEN
            msg = first(msg, MAX_LEN) * "…"
        end
        println(io, "  [$k] $name, $column: $msg")
    end
end

# The same log density and parameters that DynamicPPL's `run_ad` uses: the log joint in
# unconstrained space, at a point drawn from the prior.
function _turing_ldf(model, adtype)
    ldf = DynamicPPL.LogDensityFunction(
        model,
        DynamicPPL.getlogjoint_internal,
        DynamicPPL.LinkAll();
        adtype=adtype,
    )
    params = collect(rand(Xoshiro(468), ldf, DynamicPPL.InitFromPrior()))
    return ldf, params
end

# The Wren models come from WrenCore's PosteriorDB extension, so that they are the same as
# the ones in the main benchmark suite (`bench/suite`). The Turing models in `models/` are
# written to match them.
function bench(mod, pdb_name)
    name = string(nameof(mod))
    wren_model = posteriordb_model(pdb_name)
    turing_model = mod.turing_model

    enzyme = AutoEnzyme(; mode=Reverse)
    mooncake = AutoMooncake(; config=nothing)
    reactant = AutoReactant(enzyme)

    # -- Wren primals + gradients --
    wren_enzyme_result = _try() do
        benchmark_model(wren_model, enzyme; verbose=false)
    end
    wren_primal =
        wren_enzyme_result isa Exception ? wren_enzyme_result :
        wren_enzyme_result.lp_bench.time
    wren_enzyme =
        wren_enzyme_result isa Exception ? wren_enzyme_result :
        wren_enzyme_result.lp_and_grad_bench.time
    wren_mooncake = _try() do
        r = benchmark_model(wren_model, mooncake; verbose=false)
        r.lp_and_grad_bench.time
    end

    # -- Turing (DynamicPPL) --
    # Timed with `_min_bench`, the same as Wren in `benchmark_model`, so that both report
    # the fastest sample. (DynamicPPL's `run_ad` reports the median instead.)
    turing_enzyme_adtype = AutoEnzyme(; mode=set_runtime_activity(Reverse))
    turing_enzyme_result = _try() do
        ldf, params = _turing_ldf(turing_model, turing_enzyme_adtype)
        primal = Wren.WrenCore._min_bench(() -> logdensity(ldf, params), 1).time
        grad =
            Wren.WrenCore._min_bench(() -> logdensity_and_gradient(ldf, params), 1).time
        (; primal, grad)
    end
    turing_primal =
        turing_enzyme_result isa Exception ? turing_enzyme_result :
        turing_enzyme_result.primal
    turing_enzyme =
        turing_enzyme_result isa Exception ? turing_enzyme_result :
        turing_enzyme_result.grad
    turing_mooncake = _try() do
        ldf, params = _turing_ldf(turing_model, mooncake)
        Wren.WrenCore._min_bench(() -> logdensity_and_gradient(ldf, params), 1).time
    end

    # -- Wren + Reactant --
    wren_reactant_primal = _try() do
        # Benchmarked separately from the gradient, so that we still get a primal time
        # even if the gradient fails.
        vecmodel = VecModel(wren_model, reactant)
        params = vecmodel.sample
        # Timed the same way as in `benchmark_model`.
        Wren.WrenCore._min_bench(() -> lp(vecmodel, params), 1).time
    end
    wren_reactant_grad = _try() do
        r = benchmark_model(wren_model, reactant; verbose=false)
        r.lp_and_grad_bench.time
    end

    return BenchResult(
        name,
        wren_primal,
        turing_primal,
        wren_enzyme,
        wren_mooncake,
        turing_enzyme,
        turing_mooncake,
        wren_reactant_primal,
        wren_reactant_grad,
    )
end

include("models/ark.jl")
include("models/eight_schools_centered.jl")
include("models/glm_binomial.jl")
include("models/glm_poisson.jl")
include("models/glmm_poisson.jl")
include("models/gp_regr.jl")
include("models/kidscore_interaction.jl")
include("models/mh.jl")
include("models/rate1.jl")
include("models/sblri_blr.jl")

# Maps each module in `models/` to its PosteriorDB posterior name.
const MODELS = (
    ArK => "arK-arK",
    EightSchoolsCentered => "eight_schools-eight_schools_centered",
    GLMBinomial => "GLM_Binomial_data-GLM_Binomial_model",
    GLMPoisson => "GLM_Poisson_Data-GLM_Poisson_model",
    GLMMPoisson => "GLMM_Poisson_data-GLMM_Poisson_model",
    GPRegr => "gp_pois_regr-gp_regr",
    KidscoreInteraction => "kidiq-kidscore_interaction",
    Mh => "Mh_data-Mh_model",
    Rate1 => "Rate_1_data-Rate_1_model",
    SblriBLR => "sblri-blr",
)

results = map(MODELS) do (mod, pdb_name)
    println("Benchmarking ", nameof(mod), "...")
    bench(mod, pdb_name)
end
println()
print_table(stdout, "Primals (R = Reactant; ratios vs Wren):", results, PRIMAL_COLUMNS)
print_table(
    stdout,
    "Gradients (E = Enzyme, M = Mooncake, R = Reactant; ratios vs Wren+E):",
    results,
    GRAD_COLUMNS,
)
print_errors(stdout)
