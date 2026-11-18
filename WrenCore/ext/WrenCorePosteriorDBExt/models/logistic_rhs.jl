# PosteriorDB: ovarian/prostate-logistic_regression_rhs
module LogisticRHS

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

@wren function logistic_rhs(y, x, n, d, scale_icept, scale_global,
        nu_global, nu_local, slab_scale, slab_df)
    beta0 ~ WD.Normal(0.0, scale_icept)
    @iid z ~ WD.Normal(0.0, 1.0) size=d
    tau ~ WD.HalfTDist(nu_global, 0.0, 2 * scale_global)
    @iid lambda ~ WD.HalfTDist(nu_local, 0.0, 1.0) size=d
    caux ~ WD.InverseGamma(0.5 * slab_df, 0.5 * slab_df)
    c2 = slab_scale^2 * caux
    beta = Vector{typeof(beta0 + tau + caux + z[1] + lambda[1])}(undef, d)
    for j in eachindex(z)
        lambda_j = lambda[j]
        beta[j] = z[j] * tau * sqrt(c2 * lambda_j^2 /
            (c2 + tau^2 * lambda_j^2))
    end
    eta = beta0 .+ x * beta
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.BernoulliLogit(eta[i])
    end
    return nothing
end

function load_case(name, n, d)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    data["n"] == n && data["d"] == d ||
        error("unexpected public regularized horseshoe fixture: $name")
    return logistic_rhs(data["y"], Float64.(data["x"]), n, d,
        data["scale_icept"], data["scale_global"],
        data["nu_global"], data["nu_local"],
        data["slab_scale"], data["slab_df"])
end

const ovarian = (wren_model=load_case("ovarian-logistic_regression_rhs", 54, 1536),)
const prostate = (wren_model=load_case("prostate-logistic_regression_rhs", 102, 5966),)

end
