# PosteriorDB: dogs-dogs_nonhierarchical
module DogsNonhierarchical

using PosteriorDB
using LogExpFunctions: log1pexp
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "dogs-dogs_nonhierarchical",
    )))
end

function dog_loglikelihood(y, mu1, mu2, sigma1, sigma2, rho, z1, z2)
    total = zero(mu1 + mu2 + sigma1 + sigma2 + rho)
    l22 = sqrt(1 - rho^2)
    for j in axes(y, 1)
        # Preserve the source's z * diag_pre_multiply(sigma, L) product.
        a = inv(1 + exp(-(mu1 + z1[j] * sigma1 + z2[j] * sigma2 * rho)))
        b = inv(1 + exp(-(mu2 + z2[j] * sigma2 * l22)))
        shocks = 0
        avoids = 0
        for t in axes(y, 2)
            p = a^shocks * b^avoids
            total += logdensityof(WD.Bernoulli(p), y[j, t])
            shocks += y[j, t]
            avoids += 1 - y[j, t]
        end
    end
    return total
end

@wren function dogs_nonhierarchical(y)
    mu1 ~ WD.Flat()
    mu2 ~ WD.Flat()
    sigma1 ~ WD.HalfNormal(0.0, 1.0)
    sigma2 ~ WD.HalfNormal(0.0, 1.0)
    rho_unit ~ WD.Beta(2.0, 2.0)
    @iid z1 ~ WD.Normal(0.0, 1.0) size=size(y, 1)
    @iid z2 ~ WD.Normal(0.0, 1.0) size=size(y, 1)

    rho = 2 * rho_unit - 1
    # Logistic priors on the two unconstrained locations are tracked in #208.
    @addloglikelihood -log1pexp(-mu1) - log1pexp(mu1) -
        log1pexp(-mu2) - log1pexp(mu2)
    @addloglikelihood dog_loglikelihood(y, mu1, mu2, sigma1, sigma2,
        rho, z1, z2)
    return nothing
end

const wren_model = let
    data["n_dogs"] == 30 && data["n_trials"] == 25 ||
        error("unexpected public dogs fixture")
    dogs_nonhierarchical(data["y"])
end

end
