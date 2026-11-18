# PosteriorDB: pilots-pilots
module Pilots

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "pilots-pilots")))
end

@wren function pilots_model(group_id, scenario_id, y, n_groups, n_scenarios)
    mu_a ~ WD.Normal(0.0, 1.0)
    mu_b ~ WD.Normal(0.0, 1.0)
    sigma_a ~ WD.Uniform(0.0, 100.0)
    sigma_b ~ WD.Uniform(0.0, 100.0)
    sigma_y ~ WD.Uniform(0.0, 100.0)
    @iid a ~ WD.Normal(10 * mu_a, sigma_a) size=n_groups
    @iid b ~ WD.Normal(10 * mu_b, sigma_b) size=n_scenarios
    y_hat = a[group_id] .+ b[scenario_id]
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(y_hat[i], sigma_y)
    end
    return nothing
end

const wren_model = pilots_model(data["group_id"], data["scenario_id"], data["y"],
    data["n_groups"], data["n_scenarios"])

end
