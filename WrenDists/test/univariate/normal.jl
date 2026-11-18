using WrenDists
using WrenDists: logdensityof, test_dist, ADTestCase
import DifferentiationInterface as DI
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using FillArrays: Fill
using Plaice
using Random: Xoshiro
using Test

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

test_dist(
    "Normal",
    WrenDists.Normal(1.0, 2.0),
    Distributions.Normal(1.0, 2.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Normal(x[1], x[2]), 1.0)
        ADTestCase(f, [1.0, 2.0], (), ())
    end,),
)

test_dist(
    "iid Normal",
    WrenDists.Independent(WrenDists.Repeated(WrenDists.Normal(1.0, 2.0), (3, 4))),
    Distributions.product_distribution(fill(Distributions.Normal(1.0, 2.0), 3, 4));
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.Independent(
                    WrenDists.Repeated(WrenDists.Normal(x[1], x[2]), (3, 4)),
                ),
                ones(3, 4),
            )
            ADTestCase(f, [1.0, 2.0], (), ())
        end,
    ),
)

test_dist(
    "BatchedNormal",
    WrenDists.BatchedNormal([1.0, 2.0, 3.0], [0.5, 1.5, 1.0]),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 0.5),
        Distributions.Normal(2.0, 1.5),
        Distributions.Normal(3.0, 1.0),
    ]);
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedNormal(x[1:3], x[4:6]), [0.5, 0.5, 0.5])
            ADTestCase(f, [1.0, 2.0, 3.0, 0.5, 1.5, 1.0], (), ())
        end,
    ),
)

@testset "BatchedNormal sampling precision: $T" for T in (Float16, Float32, Float64)
    mu = T[1 2 3 4; 5 6 7 8]
    sigma = fill(T(0.5), size(mu))
    d = WrenDists.BatchedNormal(mu, sigma)
    x = rand(Xoshiro(8675309), d)
    noise = T.(randn(Xoshiro(8675309), size(mu)))
    @test eltype(x) === T
    @test x == noise .* sigma .+ mu
    draw_sum(s) = sum(rand(Xoshiro(8675309), WrenDists.BatchedNormal(mu, s)))
    @test ReverseDiff.gradient(draw_sum, sigma) ≈ noise
end

test_dist(
    "BatchedNormal (constant sigma)",
    WrenDists.BatchedNormal([1.0, 2.0, 3.0], Fill(1.0, 3)),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 1.0),
        Distributions.Normal(2.0, 1.0),
        Distributions.Normal(3.0, 1.0),
    ]);
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedNormal(x[1:3], Fill(x[4], 3)),
                [0.5, 0.5, 0.5],
            )
            ADTestCase(f, [1.0, 2.0, 3.0, 1.0], (), ())
        end,
    ),
)

test_dist(
    "BatchedNormal (constant mu)",
    WrenDists.BatchedNormal(Fill(1.0, 3), [0.5, 1.5, 1.0]),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 0.5),
        Distributions.Normal(1.0, 1.5),
        Distributions.Normal(1.0, 1.0),
    ]);
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedNormal(Fill(x[1], 3), x[2:4]),
                [0.5, 0.5, 0.5],
            )
            ADTestCase(f, [1.0, 0.5, 1.5, 1.0], (), ())
        end,
    ),
)

@testset "BatchedNormal input shape" begin
    for sigma in (ones(3), Fill(1.0, 3))
        d = WrenDists.BatchedNormal(zeros(3), sigma)
        for x in (zeros(2), zeros(4), zeros(3, 1))
            @test_throws DimensionMismatch logdensityof(d, x)
        end
    end
end
