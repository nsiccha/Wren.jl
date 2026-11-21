module WrenCoreOptimizationTests

using WrenCore, Distributions, Optimization, OptimizationOptimJL, Test
import Enzyme

@wren function poisson_rate(count)
    rate ~ Gamma(2.0, 1.0)
    @observe count ~ Poisson(rate)
end

@testset "MAP and MLE" begin
    initial = FromValues(VarNamedTuple(rate=1.5))
    map_fit = MAP(poisson_rate(3), LBFGS(); latent_source=initial, abstol=1e-10)
    mle_fit = MLE(poisson_rate(3), LBFGS(); latent_source=initial, abstol=1e-10)
    @test map_fit.params[@vn(rate)] ≈ 2.0 rtol=1e-6
    @test mle_fit.params[@vn(rate)] ≈ 3.0 rtol=1e-6
    @test Optimization.SciMLBase.successful_retcode(map_fit.solution)
    @test Optimization.SciMLBase.successful_retcode(mle_fit.solution)
end

end
