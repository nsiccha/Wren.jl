# PosteriorDB: eight_schools-eight_schools_noncentered
module EightSchoolsNoncentered

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "eight_schools-eight_schools_noncentered"),
        ),
    )
end

@wren function eight_schools_noncentered(y, sigma, J)
    @iid theta_trans ~ WD.Normal(0.0, 1.0) size=J
    mu ~ WD.Normal(0.0, 5.0)
    tau ~ WD.HalfCauchy(0.0, 5.0)
    theta = theta_trans .* tau .+ mu
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Normal(theta[i], sigma[i])
    end
    return nothing
end

const wren_model = eight_schools_noncentered(data["y"], data["sigma"], data["J"])

end
