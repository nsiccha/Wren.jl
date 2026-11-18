# PosteriorDB: seeds_data-seeds_centered_model
module SeedsCentered

using PosteriorDB
using Statistics: mean
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "seeds_data-seeds_centered_model"),
    ))
end

@wren function seeds_centered(n, N, x1, x2)
    alpha0 ~ WD.Normal(0.0, 1.0)
    alpha1 ~ WD.Normal(0.0, 1.0)
    alpha12 ~ WD.Normal(0.0, 1.0)
    alpha2 ~ WD.Normal(0.0, 1.0)
    @iid c ~ WD.Flat() size=length(n)
    sigma ~ WD.HalfCauchy(0.0, 1.0)
    @observe @iid c ~ WD.Normal(0.0, sigma) size=length(n)
    b = c .- mean(c)
    eta = alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+
        alpha12 .* (x1 .* x2) .+ b
    @batch for i in eachindex(n)
        @observe n[i] ~ WD.BinomialLogit(N[i], eta[i])
    end
    return nothing
end

const wren_model = seeds_centered(data["n"], data["N"], data["x1"], data["x2"])

end
