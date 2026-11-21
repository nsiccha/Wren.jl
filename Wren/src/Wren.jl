module Wren

using WrenCore

# Samplers
include("samplers/progress.jl")
include("samplers/prior.jl")
include("samplers/nuts.jl")
export prior, nuts

# Distributions
module Dists
using Reexport: @reexport
@reexport using WrenDists
end

# Re-exports
using WrenCore:
    WrenCore,
    # DSL
    @wren,
    @addloglikelihood,
    @iid,
    @sample,
    @observe,
    @batch,
    @track,
    # Models
    Model,
    condition,
    decondition,
    simulate,
    ModelResult,
    return_value,
    tracked_quantities,
    TrackedQuantitySpec,
    AllQuantities,
    SomeQuantities,
    NoQuantities,
    logjoint,
    logprior,
    # Vectorisation
    VecModel,
    vector_parameters,
    lp,
    lp_and_grad,
    lp_and_grad!,
    DEFAULT_AD,
    reduce_copies,
    # Benchmark and test
    @bmodel,
    benchmark_model,
    test_ad,
    WithBackend,
    WithExpectedResult,
    # Initialisations for MCMC
    FromPrior,
    FromValues,
    # Optimisation
    optimize,
    MAP,
    MLE,
    # PosteriorDB
    posteriordb_model,
    available_posteriordb_models

export WrenCore,
    # DSL
    @wren,
    @addloglikelihood,
    @iid,
    @sample,
    @observe,
    @batch,
    @track,
    # Models
    Model,
    condition,
    decondition,
    simulate,
    ModelResult,
    return_value,
    tracked_quantities,
    TrackedQuantitySpec,
    AllQuantities,
    SomeQuantities,
    NoQuantities,
    logjoint,
    logprior,
    # Vectorisation
    VecModel,
    vector_parameters,
    lp,
    lp_and_grad,
    lp_and_grad!,
    DEFAULT_AD,
    reduce_copies,
    # Benchmark and test
    @bmodel,
    benchmark_model,
    test_ad,
    WithBackend,
    WithExpectedResult,
    # Initialisations for MCMC
    FromPrior,
    FromValues,
    # Optimisation
    optimize,
    MAP,
    MLE,
    # PosteriorDB
    posteriordb_model,
    available_posteriordb_models

using VarNames: @vn, @vnt, VarNamedTuple, VarName, parameters
export @vn, @vnt, VarNamedTuple, VarName, parameters

using DensityInterface: logdensityof
export logdensityof

using StatsAPI: loglikelihood
export loglikelihood

using DifferentiationInterface:
    AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoMooncakeForward, AutoEnzyme
export AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoMooncakeForward, AutoEnzyme
using ADTypes: AutoReactant
export AutoReactant

using EnzymeCore: Forward, Reverse
export Forward, Reverse

using LinearAlgebra: I, Diagonal, Symmetric
export I, Diagonal, Symmetric

using FlexiChains:
    Wide,
    Long,
    At,
    Near,
    Contains,
    (..),
    Touches,
    Where,
    Not,
    DimArray,
    Begin,
    End,
    mean,
    median,
    std,
    var,
    quantile,
    summarystats,
    mad,
    ess,
    rhat,
    mcse,
    gelmandiag,
    gelmandiag_multivariate,
    discretediag,
    bfmi
export Wide,
    Long,
    At,
    Near,
    Contains,
    (..),
    Touches,
    Where,
    Not,
    DimArray,
    Begin,
    End,
    mean,
    median,
    std,
    var,
    quantile,
    summarystats,
    mad,
    ess,
    rhat,
    mcse,
    gelmandiag,
    gelmandiag_multivariate,
    discretediag,
    bfmi

end # module Wren
