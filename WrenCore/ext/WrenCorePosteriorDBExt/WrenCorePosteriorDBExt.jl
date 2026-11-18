module WrenCorePosteriorDBExt

using WrenCore: WrenCore

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
include("models/eight_schools_noncentered.jl")
include("models/flat_gaussian_regression.jl")
include("models/earnings.jl")
include("models/kidscore_flat.jl")
include("models/kidscore_momhs.jl")
include("models/kidscore_momhsiq.jl")
include("models/kidscore_momiq.jl")
include("models/kilpisjarvi.jl")
include("models/mesquite.jl")
include("models/nes_years.jl")
include("models/pilots.jl")
include("models/radon_pooled_all.jl")
include("models/radon_pooled_mn.jl")
include("models/rate2.jl")
include("models/rate3.jl")
include("models/rate4.jl")
include("models/rate5.jl")
include("models/rats_model.jl")
include("models/wells_dist.jl")
include("models/wells_dist100.jl")
include("models/wells_dist100ars.jl")
include("models/wells_dae.jl")
include("models/wells_dae_c.jl")
include("models/wells_dae_inter.jl")
include("models/wells_daae_c.jl")
include("models/wells_interaction.jl")
include("models/wells_interaction_c.jl")
include("models/radon_county_intercept_all.jl")
include("models/radon_county_intercept_mn.jl")
include("models/radon_hierarchical_intercept_noncentered_all.jl")
include("models/radon_hierarchical_intercept_noncentered_mn.jl")
include("models/radon_partially_pooled_noncentered_all.jl")
include("models/radon_partially_pooled_noncentered_mn.jl")
include("models/radon_variable_intercept_noncentered_all.jl")
include("models/radon_variable_intercept_noncentered_mn.jl")
include("models/radon_variable_intercept_slope_noncentered_all.jl")
include("models/radon_variable_intercept_slope_noncentered_mn.jl")
include("models/radon_variable_slope_noncentered_all.jl")
include("models/radon_variable_slope_noncentered_mn.jl")

# Maps PosteriorDB posterior names to containers with a `wren_model` property.
const MODELS = Dict(
    "arK-arK" => ArK,
    "eight_schools-eight_schools_centered" => EightSchoolsCentered,
    "GLM_Binomial_data-GLM_Binomial_model" => GLMBinomial,
    "GLM_Poisson_Data-GLM_Poisson_model" => GLMPoisson,
    "GLMM_Poisson_data-GLMM_Poisson_model" => GLMMPoisson,
    "gp_pois_regr-gp_regr" => GPRegr,
    "kidiq-kidscore_interaction" => KidscoreInteraction,
    "Mh_data-Mh_model" => Mh,
    "Rate_1_data-Rate_1_model" => Rate1,
    "sblri-blr" => SblriBLR,
    "mesquite-mesquite" => MesquiteRaw,
    "mesquite-logmesquite" => LogMesquite,
    "mesquite-logmesquite_logva" => LogMesquiteLogVA,
    "mesquite-logmesquite_logvas" => LogMesquiteLogVAS,
    "mesquite-logmesquite_logvash" => LogMesquiteLogVASH,
    "mesquite-logmesquite_logvolume" => LogMesquiteLogVolume,
    "nes1972-nes" => NES1972,
    "nes1976-nes" => NES1976,
    "nes1980-nes" => NES1980,
    "nes1984-nes" => NES1984,
    "nes1988-nes" => NES1988,
    "nes1992-nes" => NES1992,
    "nes1996-nes" => NES1996,
    "nes2000-nes" => NES2000,
    "pilots-pilots" => Pilots,
    "Rate_2_data-Rate_2_model" => Rate2,
    "Rate_3_data-Rate_3_model" => Rate3,
    "Rate_4_data-Rate_4_model" => Rate4,
    "Rate_5_data-Rate_5_model" => Rate5,
    "rats_data-rats_model" => RatsModel,
    "eight_schools-eight_schools_noncentered" => EightSchoolsNoncentered,
    "earnings-earn_height" => EarnHeight,
    "earnings-log10earn_height" => Log10EarnHeight,
    "earnings-logearn_height" => LogEarnHeight,
    "earnings-logearn_height_male" => LogEarnHeightMale,
    "earnings-logearn_interaction" => LogEarnInteraction,
    "earnings-logearn_interaction_z" => LogEarnInteractionZ,
    "earnings-logearn_logheight_male" => LogEarnLogheightMale,
    "kidiq_with_mom_work-kidscore_interaction_c" => KidscoreInteractionC,
    "kidiq_with_mom_work-kidscore_interaction_c2" => KidscoreInteractionC2,
    "kidiq_with_mom_work-kidscore_interaction_z" => KidscoreInteractionZ,
    "kidiq_with_mom_work-kidscore_mom_work" => KidscoreMomWork,
    "kidiq-kidscore_momhs" => KidscoreMomhs,
    "kidiq-kidscore_momhsiq" => KidscoreMomhsiq,
    "kidiq-kidscore_momiq" => KidscoreMomiq,
    "kilpisjarvi_mod-kilpisjarvi" => Kilpisjarvi,
    "radon_all-radon_pooled" => RadonPooledAll,
    "radon_mn-radon_pooled" => RadonPooledMN,
    "wells_data-wells_dist" => WellsDist,
    "wells_data-wells_dist100_model" => WellsDist100,
    "wells_data-wells_dist100ars_model" => WellsDist100Ars,
    "wells_data-wells_dae_model" => WellsDAE,
    "wells_data-wells_dae_c_model" => WellsDAEC,
    "wells_data-wells_dae_inter_model" => WellsDAEInter,
    "wells_data-wells_daae_c_model" => WellsDAAEC,
    "wells_data-wells_interaction_c_model" => WellsInteractionC,
    "wells_data-wells_interaction_model" => WellsInteraction,
    "radon_all-radon_county_intercept" => RadonCountyInterceptAll,
    "radon_all-radon_hierarchical_intercept_noncentered" => RadonHierarchicalInterceptNoncenteredAll,
    "radon_all-radon_partially_pooled_noncentered" => RadonPartiallyPooledNoncenteredAll,
    "radon_all-radon_variable_intercept_noncentered" => RadonVariableInterceptNoncenteredAll,
    "radon_all-radon_variable_intercept_slope_noncentered" => RadonVariableInterceptSlopeNoncenteredAll,
    "radon_all-radon_variable_slope_noncentered" => RadonVariableSlopeNoncenteredAll,
    "radon_mn-radon_county_intercept" => RadonCountyInterceptMn,
    "radon_mn-radon_hierarchical_intercept_noncentered" => RadonHierarchicalInterceptNoncenteredMn,
    "radon_mn-radon_partially_pooled_noncentered" => RadonPartiallyPooledNoncenteredMn,
    "radon_mn-radon_variable_intercept_noncentered" => RadonVariableInterceptNoncenteredMn,
    "radon_mn-radon_variable_intercept_slope_noncentered" => RadonVariableInterceptSlopeNoncenteredMn,
    "radon_mn-radon_variable_slope_noncentered" => RadonVariableSlopeNoncenteredMn,
)
const MODEL_NAMES = sort!(collect(keys(MODELS)))

"""
    posteriordb_model(name::AbstractString) -> WrenCore.Model

Return a Wren model for the PosteriorDB posterior called `name`, e.g. `"arK-arK"`, using the
data from PosteriorDB.

This requires PosteriorDB.jl and WrenDists.jl to be loaded. Only some posteriors are
available; an error is thrown for any others, which lists the available ones.
"""
function WrenCore.posteriordb_model(name::AbstractString)
    haskey(MODELS, name) || throw(
        ArgumentError(
            "no Wren model for the PosteriorDB posterior \"$name\". Available posteriors " *
            "are: " *
            join(MODEL_NAMES, ", "),
        ),
    )
    return MODELS[name].wren_model
end

"""
    available_posteriordb_models() -> Vector{String}

Return a vector of PosteriorDB model names which WrenCore has implementations for.
"""
WrenCore.available_posteriordb_models() = MODEL_NAMES

end # module
