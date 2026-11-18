module WrenADTests

@info "testing ad.jl"

using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff
using Distributions
using Enzyme: Enzyme
using ForwardDiff
using ReverseDiff
using Random
using Test
using WrenCore
using WrenDists: WrenDists

@wren function ad_model()
    x ~ Normal()
    observation = 0.5
    @observe observation ~ Normal(x)
    return x
end

@testset "AD testing utilities" begin
    @testset "test_ad" begin
        result = test_ad(
            ad_model(),
            AutoReverseDiff();
            latent_source=FromPrior(Xoshiro(473)),
            verbose=false,
        )
        @test result isa ADTestResult
        @test result.lp_actual ≈ result.lp_expected
        @test result.grad_actual ≈ result.grad_expected
        @test occursin("correctness : passed", sprint(show, MIME"text/plain"(), result))
    end

    @testset "warns if testing AD backend against itself" begin
        @test_logs (:warn, r"against itself") test_ad(
            ad_model(),
            AutoForwardDiff();
            reference=WithBackend(AutoForwardDiff()),
            verbose=false,
        )
    end

    @testset "WithExpectedResult with `nothing` skips that check" begin
        result = test_ad(
            ad_model(),
            AutoForwardDiff();
            reference=WithExpectedResult(nothing, nothing),
            verbose=false,
        )
        @test result.lp_expected === nothing
        @test result.grad_expected === nothing
    end

    @testset "ADIncorrectException" begin
        @test_throws ADIncorrectException test_ad(
            ad_model(),
            AutoForwardDiff();
            reference=WithExpectedResult(0.0, [0.0]),
            latent_source=FromPrior(Xoshiro(475)),
            verbose=false,
        )
        # Only the gradient is checked here, and it is wrong.
        e = try
            test_ad(
                ad_model(),
                AutoForwardDiff();
                reference=WithExpectedResult(nothing, [1e6]),
                verbose=false,
            )
        catch e
            e
        end
        @test e isa ADIncorrectException
        @test occursin("incorrect gradient", sprint(showerror, e))
    end

    @testset "benchmark_model" begin
        result = benchmark_model(
            ad_model();
            latent_source=FromPrior(Xoshiro(474)),
            seconds=0.1,
            verbose=false,
        )
        @test result isa BenchmarkResult
        @test result.parameters_bench.time > 0
        @test result.logjoint_bench.time > 0
        @test result.lp_bench.time > 0
        @test result.lp_and_grad_bench === nothing

        result = benchmark_model(ad_model(), AutoForwardDiff(); seconds=0.1, verbose=false)
        @test result.lp_and_grad_bench.time > 0
        @test occursin("grad / lp", sprint(show, MIME"text/plain"(), result))

        @testset "vector-space start is benchmarked exactly" begin
            # Converting this start to the original space and back again does not recover
            # the zero vector exactly, so the point must not take that round trip.
            @wren function simplex_model()
                x ~ WrenDists.Dirichlet(fill(0.1, 1495))
            end
            start = zeros(1494)
            source = FromValues(
                VarNamedTuple(x=TransformedValue(start, AsUnconstrainedVec())),
            )
            for adtype in (nothing, AutoEnzyme(; mode=Reverse))
                result = benchmark_model(
                    simplex_model(),
                    adtype;
                    latent_source=source,
                    reduce_copies=false,
                    seconds=0.1,
                    verbose=false,
                )
                @test result.vecmodel.sample == start
                @test (result.lp_and_grad_bench === nothing) == (adtype === nothing)
            end
        end

        @testset "all backends benchmark the same prior draw" begin
            # With Enzyme, `VecModel` evaluates its latent source once more in
            # `reduce_copies`, which would advance a `FromPrior` RNG.
            @wren function args_model(y)
                x ~ WrenDists.Normal()
                @observe y ~ WrenDists.Normal(x)
            end
            primal = benchmark_model(args_model(0.5); seconds=0.1, verbose=false)
            enzyme = benchmark_model(
                args_model(0.5),
                AutoEnzyme(; mode=Reverse);
                seconds=0.1,
                verbose=false,
            )
            @test enzyme.vecmodel.sample == primal.vecmodel.sample
        end

        @testset "VecModel method" begin
            vecmodel = VecModel(
                ad_model(),
                AutoForwardDiff();
                latent_source=FromPrior(Xoshiro(474)),
            )
            result = benchmark_model(vecmodel; seconds=0.1, verbose=false)
            @test result.vecmodel === vecmodel
            @test result.lp_and_grad_bench.time > 0
            @test result.logjoint_bench.time > 0
        end
    end

    @testset "@bmodel" begin
        @test (@bmodel ad_model()) isa BenchmarkResult
        @test (@bmodel ad_model() AutoForwardDiff()).lp_and_grad_bench.time > 0
    end

    @testset "reduce_copies" begin
        # `x` is written to and then read, so skipping its copy gives wrong gradients with
        # Enzyme (see #69). `y` is only read, but is copied because of `eachindex(y)`.
        @wren function reduce_model(x, y)
            @batch for i in eachindex(y)
                x[i] ~ WrenDists.Normal()
                @observe y[i] ~ WrenDists.Normal(x[i])
            end
        end
        model = reduce_model(zeros(5), randn(Xoshiro(468), 5))
        @test WrenCore._copied_args(model) == (:x, :y)

        reduced = reduce_copies(model, AutoEnzyme(; mode=Reverse); verbose=false)
        @test WrenCore._copied_args(reduced) == (:x,)
        @test reduced.evaluation_fn === model.evaluation_fn
    end
end

end # module WrenADTests
