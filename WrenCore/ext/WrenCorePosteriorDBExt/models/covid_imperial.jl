# PosteriorDB: ecdc{0401,0501}-covid19imperial_v{2,3}
module CovidImperial

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

function covid_logdensity(deaths, f, X, N, epidemic_start, pop, SI, N0,
        raw_mu, raw_alpha_hier, raw_kappa, raw_infections, raw_phi,
        raw_tau, raw_ifr_noise)
    mu = exp.(raw_mu)
    alpha_hier = exp.(raw_alpha_hier)
    kappa, phi, tau = exp(raw_kappa), exp(raw_phi), exp(raw_tau)
    infections = exp.(raw_infections)
    ifr_noise = exp.(raw_ifr_noise)
    alpha = alpha_hier .- log(1.05) / 6
    M, N2, P = length(N), size(deaths, 1), length(alpha)

    total = logdensityof(WD.Exponential(inv(0.03)), tau) + raw_tau +
        logdensityof(WD.Normal(0.0, 5.0), phi) + raw_phi +
        logdensityof(WD.Normal(0.0, 0.5), kappa) + raw_kappa
    for j in 1:P
        total += logdensityof(WD.Gamma(0.1667, 1.0), alpha_hier[j]) +
            raw_alpha_hier[j]
    end
    for m in 1:M
        total += logdensityof(WD.Normal(3.28, kappa), mu[m]) + raw_mu[m] +
            logdensityof(WD.Exponential(tau), infections[m]) +
            raw_infections[m] +
            logdensityof(WD.Normal(1.0, 0.1), ifr_noise[m]) + raw_ifr_noise[m]

        prediction = Vector{typeof(total)}(undef, N2)
        for day in 1:N0
            prediction[day] = infections[m]
        end
        cumulative = (N0 - 1) * infections[m]
        for day in (N0 + 1):N2
            cumulative += prediction[day - 1]
            convolution = zero(total)
            for previous in 1:(day - 1)
                convolution += prediction[previous] * SI[day - previous]
            end
            linear = zero(total)
            for j in 1:P
                linear += X[m, day, j] * alpha[j]
            end
            rt = mu[m] * exp(-linear)
            prediction[day] = ((pop[m] - cumulative) / pop[m]) * rt *
                convolution
        end
        for day in epidemic_start[m]:N[m]
            expected = zero(total)
            for previous in 1:(day - 1)
                expected += prediction[previous] * f[day - previous, m]
            end
            expected *= ifr_noise[m]
            expected > 0 || return oftype(total, -Inf)
            total += logdensityof(WD.NegativeBinomial(phi,
                phi / (phi + expected)), deaths[day, m])
        end
    end
    return total
end

@wren function covid_imperial(deaths, f, X, N, epidemic_start, pop, SI, N0, M, P)
    @iid raw_mu ~ WD.Flat() size=M
    @iid raw_alpha_hier ~ WD.Flat() size=P
    raw_kappa ~ WD.Flat()
    @iid raw_infections ~ WD.Flat() size=M
    raw_phi ~ WD.Flat()
    raw_tau ~ WD.Flat()
    @iid raw_ifr_noise ~ WD.Flat() size=M
    @addloglikelihood covid_logdensity(deaths, f, X, N, epidemic_start, pop,
        SI, N0, raw_mu, raw_alpha_hier, raw_kappa, raw_infections, raw_phi,
        raw_tau, raw_ifr_noise)
    return nothing
end

function load_case(name)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    data["M"] == 14 && data["P"] == 6 && data["N0"] == 6 &&
        data["N2"] == 100 && size(data["X"]) == (14, 100, 6) ||
        error("unexpected public Imperial COVID-19 fixture: $name")
    return covid_imperial(data["deaths"], data["f"], data["X"], data["N"],
        data["EpidemicStart"], data["pop"], data["SI"], data["N0"],
        data["M"], data["P"])
end

const ecdc0401_v2 = (wren_model=load_case("ecdc0401-covid19imperial_v2"),)
const ecdc0401_v3 = (wren_model=load_case("ecdc0401-covid19imperial_v3"),)
const ecdc0501_v2 = (wren_model=load_case("ecdc0501-covid19imperial_v2"),)
const ecdc0501_v3 = (wren_model=load_case("ecdc0501-covid19imperial_v3"),)

end
