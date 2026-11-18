module WrenCorePosteriorDBExt

using WrenCore: WrenCore

include("models/logaddexp.jl")
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
include("models/radon_hierarchical_intercept_centered_all.jl")
include("models/radon_hierarchical_intercept_centered_mn.jl")
include("models/radon_county.jl")
include("models/radon_partially_pooled_centered_all.jl")
include("models/radon_partially_pooled_centered_mn.jl")
include("models/radon_variable_intercept_centered_all.jl")
include("models/radon_variable_intercept_centered_mn.jl")
include("models/radon_variable_intercept_slope_centered_all.jl")
include("models/radon_variable_intercept_slope_centered_mn.jl")
include("models/radon_variable_slope_centered_all.jl")
include("models/radon_variable_slope_centered_mn.jl")
include("models/bones.jl")
include("models/capture_common.jl")
include("models/capture_m0.jl")
include("models/capture_mb.jl")
include("models/capture_mt.jl")
include("models/capture_mth.jl")
include("models/capture_mtbh.jl")
include("models/dogs.jl")
include("models/dogs_hierarchical.jl")
include("models/dogs_log.jl")
include("models/election88.jl")
include("models/hmm_example.jl")
include("models/hmm_gaussian.jl")
include("models/bball_drive_0.jl")
include("models/bball_drive_1.jl")
include("models/iohmm_reg.jl")
include("models/butterfly_occupancy.jl")
include("models/dogs_nonhierarchical.jl")
include("models/sat_2pl.jl")
include("models/irt_common.jl")
include("models/fims_2pl.jl")
include("models/science_grsm.jl")
include("models/timss_gpcm.jl")
include("models/lda_three_docs.jl")
include("models/lda_three_men.jl")
include("models/lda_pride_prejudice.jl")
include("models/traffic_bym2.jl")
include("models/mnist_rbm10.jl")
include("models/glmm1.jl")
include("models/irt_2pl.jl")
include("models/lsat.jl")
include("models/nes_logit.jl")
include("models/survey.jl")
include("models/accel_splines.jl")
include("models/arma11.jl")
include("models/diamonds.jl")
include("models/dugongs.jl")
include("models/garch11.jl")
include("models/gp_pois_regr.jl")
include("models/low_dim_gauss_mix_collapse.jl")
include("models/low_dim_gauss_mix.jl")
include("models/losscurve_sislob.jl")
include("models/normal_mixture.jl")
include("models/normal_mixture_k.jl")
include("models/sblrc_blr.jl")
include("models/seeds.jl")
include("models/seeds_centered.jl")
include("models/seeds_stanified.jl")
include("models/sesame_one_pred.jl")
include("models/surgical_model.jl")
include("models/soil_incubation.jl")
include("models/covid_imperial.jl")
include("models/presidential_gp.jl")
include("models/prophet.jl")
include("models/mcycle_gp.jl")
include("models/uk_drivers.jl")
include("models/logistic_rhs.jl")

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
    "radon_all-radon_hierarchical_intercept_centered" => RadonHierarchicalInterceptCenteredAll,
    "radon_all-radon_partially_pooled_centered" => RadonPartiallyPooledCenteredAll,
    "radon_all-radon_variable_intercept_centered" => RadonVariableInterceptCenteredAll,
    "radon_all-radon_variable_intercept_slope_centered" => RadonVariableInterceptSlopeCenteredAll,
    "radon_all-radon_variable_slope_centered" => RadonVariableSlopeCenteredAll,
    "radon_mn-radon_hierarchical_intercept_centered" => RadonHierarchicalInterceptCenteredMn,
    "radon_mn-radon_partially_pooled_centered" => RadonPartiallyPooledCenteredMn,
    "radon_mn-radon_variable_intercept_centered" => RadonVariableInterceptCenteredMn,
    "radon_mn-radon_variable_intercept_slope_centered" => RadonVariableInterceptSlopeCenteredMn,
    "radon_mn-radon_variable_slope_centered" => RadonVariableSlopeCenteredMn,
    "radon_mod-radon_county" => RadonCounty,
    "bones_data-bones_model" => Bones,
    "M0_data-M0_model" => CaptureM0,
    "Mb_data-Mb_model" => CaptureMb,
    "Mt_data-Mt_model" => CaptureMt,
    "Mth_data-Mth_model" => CaptureMth,
    "Mtbh_data-Mtbh_model" => CaptureMtbh,
    "dogs-dogs" => Dogs,
    "dogs-dogs_hierarchical" => DogsHierarchical,
    "dogs-dogs_log" => DogsLog,
    "GLMM_data-GLMM1_model" => GLMM1,
    "election88-election88_full" => Election88,
    "hmm_example-hmm_example" => HMMExample,
    "hmm_gaussian_simulated-hmm_gaussian" => HMMGaussian,
    "bball_drive_event_0-hmm_drive_0" => BballDrive0,
    "bball_drive_event_1-hmm_drive_1" => BballDrive1,
    "iohmm_reg_simulated-iohmm_reg" => IOHMMReg,
    "butterfly-multi_occupancy" => ButterflyOccupancy,
    "dogs-dogs_nonhierarchical" => DogsNonhierarchical,
    "sat-hier_2pl" => SAT2PL,
    "fims_Aus_Jpn_irt-2pl_latent_reg_irt" => FIMS2PL,
    "science_irt-grsm_latent_reg_irt" => ScienceGRSM,
    "timssAusTwn_irt-gpcm_latent_reg_irt" => TIMSSGPCM,
    "three_docs1200-ldaK2" => LDAThreeDocs,
    "three_men1-ldaK2" => LDAThreeMen.case1,
    "three_men2-ldaK2" => LDAThreeMen.case2,
    "three_men3-ldaK2" => LDAThreeMen.case3,
    "prideprejudice_chapter-ldaK5" => LDAPridePrejudice.chapter,
    "prideprejudice_paragraph-ldaK5" => LDAPridePrejudice.paragraph,
    "traffic_accident_nyc-bym2_offset_only" => TrafficBYM2,
    "mnist_100-nn_rbm1bJ10" => MNIST100RBM,
    "irt_2pl-irt_2pl" => IRT2PL,
    "lsat_data-lsat_model" => LSAT,
    "nes_logit_data-nes_logit_model" => NESLogit,
    "Survey_data-Survey_model" => Survey,
    "mcycle_splines-accel_splines" => AccelSplines,
    "arma-arma11" => ARMA11,
    "diamonds-diamonds" => Diamonds,
    "dugongs_data-dugongs_model" => Dugongs,
    "garch-garch11" => Garch11,
    "gp_pois_regr-gp_pois_regr" => GPPoissonRegression,
    "low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse" => LowDimGaussMixCollapse,
    "low_dim_gauss_mix-low_dim_gauss_mix" => LowDimGaussMix,
    "loss_curves-losscurve_sislob" => LossCurveSislob,
    "normal_2-normal_mixture" => NormalMixture,
    "normal_5-normal_mixture_k" => NormalMixtureK,
    "sblrc-blr" => SblrcBLR,
    "seeds_data-seeds_model" => Seeds,
    "seeds_data-seeds_centered_model" => SeedsCentered,
    "seeds_data-seeds_stanified_model" => SeedsStanified,
    "sesame_data-sesame_one_pred_a" => SesameOnePred,
    "surgical_data-surgical_model" => SurgicalModel,
    "soil_carbon-soil_incubation" => SoilIncubation,
    "ecdc0401-covid19imperial_v2" => CovidImperial.ecdc0401_v2,
    "ecdc0401-covid19imperial_v3" => CovidImperial.ecdc0401_v3,
    "ecdc0501-covid19imperial_v2" => CovidImperial.ecdc0501_v2,
    "ecdc0501-covid19imperial_v3" => CovidImperial.ecdc0501_v3,
    "state_wide_presidential_votes-hierarchical_gp" => PresidentialGP,
    "rstan_downloads-prophet" => Prophet,
    "mcycle_gp-accel_gp" => McycleGP,
    "uk_drivers-state_space_stochastic_level_stochastic_seasonal" => UKDrivers,
    "ovarian-logistic_regression_rhs" => LogisticRHS.ovarian,
    "prostate-logistic_regression_rhs" => LogisticRHS.prostate,
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
