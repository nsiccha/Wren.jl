# PosteriorDB: GLMM_Poisson_data-GLMM_Poisson_model
module GLMMPoisson

using PosteriorDB
using DynamicPPL, Distributions
using FillArrays: Fill

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "GLMM_Poisson_data-GLMM_Poisson_model"),
        ),
    )
end

@model function pdb_GLMM_Poisson_model(n, C, year, year_squared, year_cubed)
    alpha ~ Uniform(-20, 20)
    beta1 ~ Uniform(-10, 10)
    beta2 ~ Uniform(-10, 10)
    beta3 ~ Uniform(-10, 10)
    eps ~ product_distribution(Fill(Flat(), n))
    sigma ~ Uniform(0, 5)
    log_lambda =
        alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed .+ eps
    for i in eachindex(C)
        C[i] ~ LogPoisson(log_lambda[i])
    end
    # `eps` is already a parameter, so it can't go on the left of another tilde. (Wren
    # allows this with `@observe`.)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(eps))),
        eps,
    )
end

const turing_model = let
    year = data["year"]
    pdb_GLMM_Poisson_model(data["n"], data["C"], year, year .^ 2, year .^ 3)
end

end
