# PosteriorDB: GLM_Binomial_data-GLM_Binomial_model
module GLMBinomial

using PosteriorDB
using DynamicPPL, Distributions

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "GLM_Binomial_data-GLM_Binomial_model"),
        ),
    )
end

@model function pdb_GLM_Binomial_model(nyears, C, N, year, year_squared)
    alpha ~ Normal(0, 100)
    beta1 ~ Normal(0, 100)
    beta2 ~ Normal(0, 100)
    logit_p = alpha .+ beta1 .* year .+ beta2 .* year_squared
    for i in eachindex(C)
        C[i] ~ BinomialLogit(N[i], logit_p[i])
    end
end

const turing_model = let
    year = data["year"]
    pdb_GLM_Binomial_model(data["nyears"], data["C"], data["N"], year, year .^ 2)
end

end
