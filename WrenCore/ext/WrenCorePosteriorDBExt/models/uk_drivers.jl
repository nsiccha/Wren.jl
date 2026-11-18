# PosteriorDB: uk_drivers-state_space_stochastic_level_stochastic_seasonal
module UKDrivers

using PosteriorDB
using Statistics: mean, std
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "uk_drivers-state_space_stochastic_level_stochastic_seasonal",
    )))
end

function state_space_loglikelihood(y, x, w, mu, seasonal, beta, lambda,
        sigma1, sigma2, sigma3)
    total = zero(beta + lambda + sigma1)
    previous_mu = zero(total)
    for t in eachindex(y)
        if t >= 2
            total += logdensityof(WD.Normal(previous_mu, sigma2), mu[t])
        end
        if t >= 12
            previous_season = zero(total)
            for lag in 1:11
                previous_season += seasonal[t - lag]
            end
            total += logdensityof(WD.Normal(-previous_season, sigma1),
                seasonal[t])
        end
        total += logdensityof(WD.Normal(mu[t] + beta * x[t] + lambda * w[t] +
            seasonal[t], sigma3), y[t])
        previous_mu = mu[t]
    end
    return total
end

@wren function uk_drivers(y, x, w, lower, width)
    @iid mu ~ WD.Uniform(lower, lower + width) size=length(y)
    @iid seasonal ~ WD.Flat() size=length(y)
    beta ~ WD.Flat()
    lambda ~ WD.Flat()
    raw_sigma1 ~ WD.Flat()
    raw_sigma_gap2 ~ WD.Flat()
    raw_sigma_gap3 ~ WD.Flat()

    sigma1 = exp(raw_sigma1)
    sigma2 = sigma1 + exp(raw_sigma_gap2)
    sigma3 = sigma2 + exp(raw_sigma_gap3)
    # Stan's positive_ordered[3] transform and the three untruncated
    # Student-t prior factors on its positive components.
    # Native positive-ordered prior support is tracked in #207.
    @addloglikelihood raw_sigma1 + raw_sigma_gap2 + raw_sigma_gap3 +
        logdensityof(WD.TDist(4.0, 0.0, 1.0), sigma1) +
        logdensityof(WD.TDist(4.0, 0.0, 1.0), sigma2) +
        logdensityof(WD.TDist(4.0, 0.0, 1.0), sigma3)
    @addloglikelihood state_space_loglikelihood(y, x, w, mu, seasonal,
        beta, lambda, sigma1, sigma2, sigma3)
    return nothing
end

const wren_model = let
    y = data["y"]
    data["n"] == 192 || error("unexpected UK drivers fixture")
    lower = mean(y) - 3 * std(y)
    width = 6 * std(y)
    uk_drivers(y, data["x"], data["w"], lower, width)
end

end
