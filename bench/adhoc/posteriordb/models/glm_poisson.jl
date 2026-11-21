# PosteriorDB: GLM_Poisson_Data-GLM_Poisson_model
module GLMPoisson

using PosteriorDB
using DynamicPPL, Distributions

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "GLM_Poisson_Data-GLM_Poisson_model"),
        ),
    )
end

@model function pdb_GLM_Poisson_model(n, C, year, year_squared, year_cubed)
    alpha ~ Uniform(-20, 20)
    beta1 ~ Uniform(-10, 10)
    beta2 ~ Uniform(-10, 10)
    beta3 ~ Uniform(-10, 10)
    log_lambda = alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed
    for i in eachindex(C)
        C[i] ~ LogPoisson(log_lambda[i])
    end
end

const turing_model = let
    year = data["year"]
    pdb_GLM_Poisson_model(data["n"], data["C"], year, year .^ 2, year .^ 3)
end

end
