# PosteriorDB: election88-election88_full
module Election88

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "election88-election88_full",
    )))
end

@wren function election88(age, edu, age_edu, state, region, black, female,
    v_prev_full, y, n_age, n_edu, n_age_edu, n_state, n_region)
    @iid beta ~ WD.Normal(0.0, 100.0) size=5
    sigma_a ~ WD.Uniform(0.0, 100.0)
    sigma_b ~ WD.Uniform(0.0, 100.0)
    sigma_c ~ WD.Uniform(0.0, 100.0)
    sigma_d ~ WD.Uniform(0.0, 100.0)
    sigma_e ~ WD.Uniform(0.0, 100.0)
    @iid a ~ WD.Normal(0.0, sigma_a) size=n_age
    @iid b ~ WD.Normal(0.0, sigma_b) size=n_edu
    @iid c ~ WD.Normal(0.0, sigma_c) size=n_age_edu
    @iid d ~ WD.Normal(0.0, sigma_d) size=n_state
    @iid e ~ WD.Normal(0.0, sigma_e) size=n_region
    eta = beta[1] .+ beta[2] .* black .+ beta[3] .* female .+
        beta[5] .* (female .* black) .+ beta[4] .* v_prev_full .+
        a[age] .+ b[edu] .+ c[age_edu] .+ d[state] .+ e[region]
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.BernoulliLogit(eta[i])
    end
    return nothing
end

const wren_model = election88(data["age"], data["edu"], data["age_edu"],
    data["state"], data["region_full"], data["black"], data["female"],
    data["v_prev_full"], data["y"], data["n_age"], data["n_edu"],
    data["n_age_edu"], data["n_state"], data["n_region_full"])

end
