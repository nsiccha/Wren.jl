# PosteriorDB: state_wide_presidential_votes-hierarchical_gp
module PresidentialGP

using PosteriorDB
using LinearAlgebra: Symmetric, cholesky, issuccess
using WrenCore: @wren, @addloglikelihood, @iid, logdensityof
using WrenDists: WrenDists as WD

function year_covariance(n_years, sigma_long, sigma_short,
        length_long, length_short)
    T = typeof(sigma_long + sigma_short + length_long + length_short)
    covariance = Matrix{T}(undef, n_years, n_years)
    for j in 1:n_years, i in 1:n_years
        distance2 = (i - j)^2
        covariance[i, j] = sigma_long^2 *
            exp(-0.5 * distance2 / length_long^2) + sigma_short^2 *
            exp(-0.5 * distance2 / length_short^2) +
            (i == j ? 1e-6 : 0.0)
    end
    return covariance
end

function gp_logdensity(y, state_region_ind, state_ind, region_ind, year_ind,
        gp_region_std, gp_state_std, year_std, state_std, region_std,
        tot_var, prop_var, mu, length_gp_region_long,
        length_gp_state_long, length_gp_region_short, length_gp_state_short)
    variances = 17 .* prop_var .* tot_var
    sigma_year = sqrt(variances[1])
    sigma_region = sqrt(variances[2])
    sigma_state = sqrt.(view(variances, 3:12))
    sigma_gp_region_long = sqrt(variances[13])
    sigma_gp_state_long = sqrt(variances[14])
    sigma_gp_region_short = sqrt(variances[15])
    sigma_gp_state_short = sqrt(variances[16])
    sigma_error = sqrt(variances[17])

    n_years = size(gp_region_std, 1)
    region_cov = year_covariance(n_years, sigma_gp_region_long,
        sigma_gp_region_short, length_gp_region_long, length_gp_region_short)
    state_cov = year_covariance(n_years, sigma_gp_state_long,
        sigma_gp_state_short, length_gp_state_long, length_gp_state_short)
    region_chol = cholesky(Symmetric(region_cov); check=false)
    state_chol = cholesky(Symmetric(state_cov); check=false)
    if !(issuccess(region_chol) && issuccess(state_chol))
        return oftype(mu, -Inf)
    end
    gp_region = region_chol.L * gp_region_std
    gp_state = state_chol.L * gp_state_std
    year_re = sigma_year .* year_std
    region_re = sigma_region .* region_std
    state_re = sigma_state[state_region_ind] .* state_std
    total = zero(mu)
    for n in eachindex(y)
        state, region, year = state_ind[n], region_ind[n], year_ind[n]
        expected = mu + year_re[year] + state_re[state] +
            region_re[region] + gp_region[year, region] + gp_state[year, state]
        total += logdensityof(WD.Normal(expected, sigma_error), y[n])
    end
    return total
end

@wren function hierarchical_gp(y, state_region_ind, state_ind, region_ind,
        year_ind, n_years, n_regions, n_states, n_years_obs)
    @iid gp_region_std ~ WD.Normal() size=(n_years, n_regions)
    @iid gp_state_std ~ WD.Normal() size=(n_years, n_states)
    @iid year_std ~ WD.Normal() size=n_years_obs
    @iid state_std ~ WD.Normal() size=n_states
    @iid region_std ~ WD.Normal() size=n_regions
    tot_var ~ WD.Gamma(3.0, 1 / 3)
    prop_var ~ WD.Dirichlet(fill(2.0, 17))
    mu ~ WD.Normal(0.5, 0.5)
    length_gp_region_long ~ WD.Weibull(30.0, 8.0)
    length_gp_state_long ~ WD.Weibull(30.0, 8.0)
    length_gp_region_short ~ WD.Weibull(30.0, 3.0)
    length_gp_state_short ~ WD.Weibull(30.0, 3.0)
    @addloglikelihood gp_logdensity(y, state_region_ind, state_ind,
        region_ind, year_ind, gp_region_std, gp_state_std, year_std,
        state_std, region_std, tot_var, prop_var, mu,
        length_gp_region_long, length_gp_state_long,
        length_gp_region_short, length_gp_state_short)
    return nothing
end

const wren_model = let
    name = "state_wide_presidential_votes-hierarchical_gp"
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    data["N"] == 550 && data["N_states"] == 50 &&
        data["N_regions"] == 10 && data["N_years"] == 14 &&
        data["N_years_obs"] == 11 ||
        error("unexpected public presidential-vote GP fixture")
    hierarchical_gp(data["y"], data["state_region_ind"], data["state_ind"],
        data["region_ind"], data["year_ind"], data["N_years"],
        data["N_regions"], data["N_states"], data["N_years_obs"])
end

end
