# PosteriorDB: irt_2pl-irt_2pl
module IRT2PL

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "irt_2pl-irt_2pl")))
end

@wren function irt_2pl(y, I, J)
    sigma_theta ~ WD.HalfCauchy(0.0, 2.0)
    @iid theta ~ WD.Normal(0.0, sigma_theta) size=J
    sigma_a ~ WD.HalfCauchy(0.0, 2.0)
    @iid a ~ WD.LogNormal(0.0, sigma_a) size=I
    mu_b ~ WD.Normal(0.0, 5.0)
    sigma_b ~ WD.HalfCauchy(0.0, 2.0)
    @iid b ~ WD.Normal(mu_b, sigma_b) size=I
    logits = Vector{typeof(a[1] + theta[1] + b[1])}(undef, length(y))
    for n in eachindex(y)
        i = mod1(n, I)
        j = cld(n, I)
        logits[n] = a[i] * (theta[j] - b[i])
    end
    @batch for n in eachindex(y)
        @observe y[n] ~ WD.BernoulliLogit(logits[n])
    end
    return nothing
end

const wren_model = irt_2pl(data["y"], data["I"], data["J"])

end
