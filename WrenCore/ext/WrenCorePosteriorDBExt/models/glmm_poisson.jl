# PosteriorDB: GLMM_Poisson_data-GLMM_Poisson_model
module GLMMPoisson

using PosteriorDB
using WrenCore: @wren
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(
        PosteriorDB.dataset(
            PosteriorDB.posterior(pdb, "GLMM_Poisson_data-GLMM_Poisson_model"),
        ),
    )
end

@wren function glmm_poisson_model(counts, year, year_squared, year_cubed, n)
    alpha ~ WD.Uniform(-20.0, 20.0)
    beta1 ~ WD.Uniform(-10.0, 10.0)
    beta2 ~ WD.Uniform(-10.0, 10.0)
    beta3 ~ WD.Uniform(-10.0, 10.0)
    @iid eps ~ WD.Flat() size=n
    sigma ~ WD.Uniform(0.0, 5.0)
    log_lambda =
        alpha .+ beta1 .* year .+ beta2 .* year_squared .+ beta3 .* year_cubed .+ eps
    @batch for i in eachindex(counts)
        @observe counts[i] ~ WD.PoissonLog(log_lambda[i])
    end
    @observe @iid eps ~ WD.Normal(0.0, sigma) size=n
    return nothing
end

const wren_model = let
    counts = data["C"]
    year = data["year"]
    glmm_poisson_model(counts, year, year .^ 2, year .^ 3, data["n"])
end

end
