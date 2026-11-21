module WrenCore

using DensityInterface: DensityInterface, logdensityof
import DifferentiationInterface as DI
using DocStringExtensions: TYPEDFIELDS
using FillArrays: FillArrays
using LogDensityProblems: LogDensityProblems
using Plaice: Plaice
using Random: AbstractRNG, default_rng
using VarNames: VarNames, varname, VarName, VarNamedTuple, vnt_pretty_print
using StatsAPI: StatsAPI

function _issue_error(str)
    error(
        "$str.\n\nPlease report this issue at https://github.com/JuliaBayes/Wren.jl/issues. Thank you!\n",
    )
end

function _docs_link(str)
    return "https://juliabayes.org/Wren.jl/$str"
end

const WREN_COMPILATION_DEBUG = Ref(false)

# Parameter spaces
include("space.jl")

# Latent sources
include("latents/latents.jl")
include("latents/void.jl")
include("latents/prior.jl")
include("latents/values.jl")
include("latents/interval.jl")

# Accumulators
include("accs/accs.jl")
include("accs/logp.jl")
include("accs/vnt.jl")
include("accs/params.jl")
include("accs/sitename_tracker.jl")
include("accs/tq.jl")
include("accs/accessors.jl")

# Model definition
include("model/expr.jl")
using .Expressions:
    AbstractWrenExpr,
    AddLogLikelihood,
    TrackedQuantity,
    IID,
    Observe,
    OtherStatement,
    Batch,
    ReturnStatement,
    Sample,
    WrenExpr,
    _show_wren_exprs,
    _wren_expr_to_display_expr,
    get_top_level_symbol,
    to_wren_exprs

include("model/dataflow.jl")
include("model/model.jl")
include("model/submodel.jl")
include("model/emit.jl")

# Vectorisation
include("latents/vector.jl")
include("accs/vectorised_params.jl")
include("accs/vec_sample.jl")
include("vecmodel.jl")

# User-facing interface
include("evaluate.jl")
include("condition.jl")
include("test_ad.jl")
include("test_utils.jl")
include("optimization.jl")

# Overloaded in extensions
function to_flexichain end
function posteriordb_model end
function available_posteriordb_models end

export @wren,
    @addloglikelihood,
    @iid,
    @sample,
    @observe,
    @batch,
    @track,
    Model,
    AbstractLatentSource,
    AbstractTransform,
    AbstractParameterSpace,
    AbstractADCorrectnessTestSetting,
    AbstractAccumulator,
    AccumulatorTuple,
    LogPriorAccumulator,
    LogLikelihoodAccumulator,
    LogJacobianAccumulator,
    ParamsAccumulator,
    TrackedQuantityAccumulator,
    AsUnconstrainedVec,
    AsVec,
    ADIncorrectException,
    ADTestResult,
    BenchmarkResult,
    FromValues,
    FromPrior,
    FromUnconstrainedInterval,
    FromVector,
    UnconstrainedSpace,
    AsOriginal,
    TransformedValue,
    VecModel,
    DEFAULT_AD,
    VectorisedParamsAccumulator,
    vector_parameters,
    WithBackend,
    WithExpectedResult,
    OriginalSpace,
    apply_parameter_space,
    condition,
    conditioned,
    decondition,
    iid,
    independent,
    simulate,
    ModelResult,
    return_value,
    tracked_quantities,
    TrackedQuantitySpec,
    AllQuantities,
    SomeQuantities,
    NoQuantities,
    evaluate,
    EvalContext,
    get_latent,
    logjac,
    logjoint,
    logjoint_transformed,
    logprior,
    internal_values_as_vector,
    lp,
    lp_and_grad,
    lp_and_grad!,
    @bmodel,
    benchmark_model,
    test_ad,
    reduce_copies,
    target_transform,
    # Optimisation
    optimize,
    MAP,
    MLE,
    # FlexiChains
    to_flexichain,
    # PosteriorDB
    posteriordb_model,
    available_posteriordb_models

# Reexports
using VarNames: @vn, @vnt, VarNamedTuple, VarName, parameters
export @vn, @vnt, VarNamedTuple, VarName, parameters

using DensityInterface: logdensityof
export logdensityof

using StatsAPI: loglikelihood
export loglikelihood

using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme
export AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme

using EnzymeCore: Forward, Reverse
export Forward, Reverse

using LinearAlgebra: I, Diagonal, Symmetric
export I, Diagonal, Symmetric

end # module WrenCore
