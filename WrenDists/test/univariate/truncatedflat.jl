using WrenDists: WrenDists as WD
using Plaice
using Random: Xoshiro
using Test

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

@testset "truncated Flat" begin
    for (bounds, u, x, logjac) in (
        ((lower=0.0,), log(2.0), 2.0, log(2.0)),
        ((upper=5.0,), log(2.0), 3.0, log(2.0)),
        ((lower=-1.0, upper=3.0), log(2.0), 5 / 3, log(8 / 9)),
    )
        d = WD.truncated(WD.Flat(); bounds...)
        @test WD.logdensityof(d, x) == 0.0
        physical, jac = Plaice.with_logabsdet_jacobian(
            Plaice.from_unconstrained_vec(d), [u])
        @test physical ≈ x
        @test jac ≈ logjac
        @test all(z -> d.lower < z < d.upper,
            [rand(Xoshiro(seed), d) for seed in 1:20])
    end

    lower = WD.truncated(WD.Flat(); lower=0.0)
    @test WD.logdensityof(lower, -1.0) == -Inf
    @test WD.logdensityof(lower, 0.0) == 0.0

    bounded = WD.truncated(WD.Flat(); lower=-1.0, upper=3.0)
    @test WD.logdensityof(bounded, 3.1) == -Inf
    @test WD.logdensityof(bounded, 3.0) == 0.0
    @test WD.logdensityof(bounded, 1.0) == 0.0 # No Uniform normaliser.

    @test rand(Xoshiro(1), WD.truncated(WD.Flat{Float32}(); lower=0)) isa Float32

    # An empty interval, e.g. a dependent bound that rounds to `lower == upper`,
    # rejects every point instead of throwing.
    empty = WD.truncated(WD.Flat(); lower=1.0, upper=1.0)
    @test WD.logdensityof(empty, 1.0) == -Inf
    @test WD.logdensityof(WD.truncated(WD.Flat(); lower=2.0, upper=1.0), 1.5) == -Inf
end

if RUN_REACTANT
    @testset "truncated Flat with a traced bound" begin
        # A bound that depends on another parameter is a traced number when the
        # density is compiled. An empty interval cannot throw there.
        density(alpha, x) =
            WD.logdensityof(WD.truncated(WD.Flat(); lower=0.0, upper=1 - alpha), x)
        for (alpha, x) in ((0.3, 0.5), (0.3, 0.7), (0.3, 0.9), (1.2, 0.1))
            alpha_ra = Reactant.to_rarray(alpha; track_numbers=true)
            x_ra = Reactant.to_rarray(x; track_numbers=true)
            @test Float64(Reactant.@jit density(alpha_ra, x_ra)) == density(alpha, x)
        end
    end
end
