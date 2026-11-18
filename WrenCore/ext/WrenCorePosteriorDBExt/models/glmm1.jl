# PosteriorDB: GLMM_data-GLMM1_model
module GLMM1

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "GLMM_data-GLMM1_model"),
    ))
end

@wren function glmm1_model(nsite, obs, obssite)
    @iid alpha ~ WD.Flat() size=nsite
    mu_alpha ~ WD.Normal(0.0, 10.0)
    sd_alpha ~ WD.Uniform(0.0, 5.0)
    @observe @iid alpha ~ WD.Normal(mu_alpha, sd_alpha) size=nsite
    @batch for i in eachindex(obs)
        @observe obs[i] ~ WD.PoissonLog(alpha[obssite[i]])
    end
    return nothing
end

const wren_model = glmm1_model(data["nsite"], data["obs"], data["obssite"])

end
