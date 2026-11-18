module Issue192

using Test, WrenCore
using WrenDists: WrenDists as WD
import Enzyme
import ForwardDiff

@wren function positive_flat_scale(y)
    sigma ~ WD.truncated(WD.Flat(); lower=0.0)
    @observe y ~ WD.Normal(0.0, sigma)
end

@testset "#192 positive flat scale" begin
    vm = VecModel(positive_flat_scale(1.0), AutoEnzyme(; mode=Reverse);
        latent_source=FromValues(VarNamedTuple(sigma=2.0)), reduce_copies=false)
    u = log(2.0)
    value, gradient = lp_and_grad(vm, [u])
    @test value ≈ logdensityof(WD.Normal(0.0, 2.0), 1.0) + u
    @test gradient ≈ [0.25]

    prior_vm = VecModel(positive_flat_scale(1.0), AutoEnzyme(; mode=Reverse))
    @test isfinite(lp(prior_vm, vector_parameters(prior_vm)))
end

# garch11's `beta1 ∈ (0, 1 - alpha1)` written directly, with a bound that depends on
# another parameter.
@wren function dependent_flat_bound()
    alpha ~ WD.Beta(1.0, 2.0)
    beta ~ WD.truncated(WD.Flat(); lower=0.0, upper=1 - alpha)
end

@testset "#192 parameter-dependent flat bound" begin
    logistic(x) = inv(1 + exp(-x))
    function reference(v)
        alpha = logistic(v[1])
        s = logistic(v[2])
        # Beta(1, 2) density, the logit Jacobian of alpha, and the Jacobian of
        # beta = (1 - alpha) * s; the flat density itself is zero.
        return log(2) + log1p(-alpha) + log(alpha) + log1p(-alpha) +
            log1p(-alpha) + log(s) + log1p(-s)
    end
    v = [0.3, -0.4]
    native = VecModel(dependent_flat_bound(), AutoEnzyme(; mode=Reverse))
    value, gradient = lp_and_grad(native, v)
    @test value ≈ reference(v)
    @test gradient ≈ ForwardDiff.gradient(reference, v)
end

end
