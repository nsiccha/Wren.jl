# PosteriorDB: eight_schools-eight_schools_centered
module EightSchoolsCentered

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "eight_schools-eight_schools_centered"),
        ),
    )
end

@model function dppl_eight_schools_centered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ product_distribution(Fill(Normal(mu, tau), J))
    for i in eachindex(y)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

const turing_model = dppl_eight_schools_centered(data["J"], data["y"], data["sigma"])

end
