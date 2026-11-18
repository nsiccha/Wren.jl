# PosteriorDB: seeds_data-seeds_model
module Seeds

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "seeds_data-seeds_model")))
end

@wren function seeds_model(n, N, x1, x2)
    alpha0 ~ WD.Normal(0.0, 1000.0)
    alpha1 ~ WD.Normal(0.0, 1000.0)
    alpha12 ~ WD.Normal(0.0, 1000.0)
    alpha2 ~ WD.Normal(0.0, 1000.0)
    # Stan Gamma uses a rate; WrenDists Gamma uses a scale.
    tau ~ WD.Gamma(0.001, 1000.0)
    sigma = inv(sqrt(tau))
    @iid b ~ WD.Normal(0.0, sigma) size=length(n)
    eta = alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+
        alpha12 .* (x1 .* x2) .+ b
    @batch for i in eachindex(n)
        @observe n[i] ~ WD.BinomialLogit(N[i], eta[i])
    end
    return nothing
end

const wren_model = seeds_model(data["n"], data["N"], data["x1"], data["x2"])

end
