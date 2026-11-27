module WrenNUTSTests

using Wren
using Wren.Dists
using FlexiChains
using Test
using Random
import Enzyme
import ForwardDiff

@wren function f()
    x ~ Normal()
end
model = f()

struct RecordStarts
    starts::Channel{Vector{Float64}}
end
function Wren.DynamicHMC.warmup(_, recorder::RecordStarts, state)
    put!(recorder.starts, copy(state.Q.q))
    return nothing, state
end

struct RecordInitialization
    states::Channel{NamedTuple}
end
function Wren.DynamicHMC.warmup(_, recorder::RecordInitialization, state)
    put!(recorder.states, (; q=copy(state.Q.q), state.ϵ))
    return nothing, state
end

mutable struct SequenceValues <: WrenCore.AbstractLatentSource
    values::Vector{Float64}
    next::Int
end
SequenceValues(values) = SequenceValues(values, 1)
function WrenCore.get_latent(source::SequenceValues, ::VarName, dist, ::Val)
    index = source.next
    source.next += 1
    value = source.values[index]
    return WrenCore.TransformedValue(value, WrenCore.AsOriginal())
end
WrenCore.scope(source::SequenceValues, ::VarName) = source

# Count gradient evaluations from inside the model: with ForwardDiff, a one-dimensional
# model's gradient is one evaluation with a dual number, and plain evaluations see a Float64.
const DUAL_EVALUATIONS = Ref(0)
count_dual(::Real) = 0.0
function count_dual(::ForwardDiff.Dual)
    DUAL_EVALUATIONS[] += 1
    return 0.0
end

# A warmup stage that changes nothing and records the count when warmup ends.
struct RecordDualCount
    count::Base.RefValue{Int}
end
function Wren.DynamicHMC.warmup(_, recorder::RecordDualCount, state)
    recorder.count[] = DUAL_EVALUATIONS[]
    return nothing, state
end

@testset "nuts.jl" begin
    @testset "interface" begin
        chain = nuts(model, 1000, 4; progress=false)
        @test chain isa VNChain
        @test FlexiChains.niters(chain) == 1000
        @test FlexiChains.nchains(chain) == 4
        @test only(FlexiChains.parameters(chain)) == @vn(x)
    end

    @testset "gradient counts" begin
        @wren function counted_normal()
            x ~ Normal()
            @addloglikelihood count_dual(x)
        end
        adtype = AutoForwardDiff()
        # Preparing the VecModel may evaluate the gradient; that is not part of any chain.
        DUAL_EVALUATIONS[] = 0
        WrenCore.VecModel(counted_normal(), adtype)
        preparation = DUAL_EVALUATIONS[]

        at_warmup_end = Ref(-1)
        stages =
            (Wren.DynamicHMC.default_warmup_stages()..., RecordDualCount(at_warmup_end))
        DUAL_EVALUATIONS[] = 0
        chn = nuts(
            Xoshiro(468),
            counted_normal(),
            200;
            adtype,
            progress=false,
            warmup_stages=stages,
        )
        n = vec(chn[:n_gradient_evaluations])
        s = vec(chn[:n_steps])
        @test eltype(n) <: Integer && eltype(s) <: Integer
        @test all(>=(1), s)
        @test diff(n) == s[2:end]
        # Initial point and step-size search, then at least one step per warmup transition.
        @test n[1] - s[1] == at_warmup_end[] - preparation
        @test n[1] - s[1] > sum(stage -> stage.N, stages[2:(end-1)])
        @test sum(s) == DUAL_EVALUATIONS[] - at_warmup_end[]
        @test n[end] == DUAL_EVALUATIONS[] - preparation

        # Each chain counts its own evaluations.
        chn4 = nuts(Xoshiro(468), model, 100, 4; progress=false)
        n4 = chn4[:n_gradient_evaluations]
        s4 = chn4[:n_steps]
        for j in 1:4
            @test diff(vec(n4[:, j])) == vec(s4[2:end, j])
            @test n4[1, j] > s4[1, j]
        end
    end

    @testset "initialization" begin
        @wren function bounded_posterior()
            x ~ Normal(10.0, 0.1)
            @addloglikelihood (9.0 < x < 11.0 ? 0.0 : -Inf)
        end
        starts = Channel{Vector{Float64}}(2)
        kwargs = (;
            progress=false,
            warmup_stages=(RecordStarts(starts), Wren.DynamicHMC.InitialStepsizeSearch()),
        )
        chn1 = nuts(Xoshiro(468), bounded_posterior(), 3, 2; kwargs...)
        q1, q2 = take!(starts), take!(starts)
        chn2 = nuts(Xoshiro(468), bounded_posterior(), 3, 2; kwargs...)
        @test all(x -> 9.0 < x < 11.0, chn1[@vn(x)])
        @test FlexiChains.has_same_data(chn1, chn2)
        @test q1 != q2
    end

    @testset "rng" begin
        chn1 = nuts(Xoshiro(468), model, 100, 3; progress=false)
        chn2 = nuts(Xoshiro(468), model, 100, 3; progress=false)
        @test FlexiChains.has_same_data(chn1, chn2)
        chn3 = nuts(Xoshiro(469), model, 100, 3; progress=false)
        @test !FlexiChains.has_same_data(chn1, chn3)

        # reusing RNG should give new chains
        s = Xoshiro(468)
        chn1 = nuts(s, model, 100, 3; progress=false)
        chn2 = nuts(s, model, 100, 3; progress=false)
        @test !FlexiChains.has_same_data(chn1, chn2)
    end

    @testset "model-space initialization" begin
        @wren function positive_parameter()
            x ~ LogNormal()
        end
        states = Channel{NamedTuple}(2)
        kwargs = (;
            progress=false,
            initial_stepsize=0.125f0,
            warmup_stages=(RecordInitialization(states),),
            algorithm=Wren.DynamicHMC.NUTS(; max_depth=1),
        )
        initial = FindFiniteDensity(VarNamedTuple(; x=exp(2.0)))
        nuts(Xoshiro(468), positive_parameter(), 3, 2; init=initial, kwargs...)
        s1, s2 = take!(states), take!(states)
        @test s1.q ≈ s2.q ≈ [2.0]
        @test s1.ϵ == s2.ϵ == 0.125
        # A plain VarNamedTuple is shorthand for `FindFiniteDensity(...)`.
        nuts(
            Xoshiro(468),
            positive_parameter(),
            3,
            2;
            init=VarNamedTuple(; x=exp(2.0)),
            kwargs...,
        )
        @test take!(states).q ≈ take!(states).q ≈ [2.0]
        chn = nuts(
            Xoshiro(468),
            positive_parameter(),
            3;
            init=initial,
            progress=false,
            initial_stepsize=0.01,
        )
        @test all(>(0), chn[@vn(x)])

        nuts(
            Xoshiro(468),
            positive_parameter(),
            3,
            2;
            init=FindFiniteDensity(FromPrior(Xoshiro(123))),
            kwargs...,
        )
        q1, q2 = take!(states).q, take!(states).q
        nuts(
            Xoshiro(468),
            positive_parameter(),
            3,
            2;
            init=FindFiniteDensity(FromPrior(Xoshiro(123))),
            kwargs...,
        )
        @test q1 != q2
        @test sort([q1, q2]) == sort([take!(states).q, take!(states).q])

        source = SequenceValues(Float64[])
        @test_throws ArgumentError nuts(
            positive_parameter(),
            3;
            init=FindFiniteDensity(source),
            initialization=(; q=[0.0]),
            progress=false,
        )
        @test source.next == 1
    end

    @testset "finite-density search" begin
        @wren function finite_region()
            x ~ Normal()
            @addloglikelihood (x > 1 ? 0.0 : -Inf)
        end
        @wren function flat_parameter()
            x ~ Flat()
        end
        starts = Channel{Vector{Float64}}(1)
        kwargs = (;
            progress=false,
            reduce_copies=false,
            initial_stepsize=0.01,
            warmup_stages=(RecordStarts(starts),),
            algorithm=Wren.DynamicHMC.NUTS(; max_depth=1),
        )

        # Preparation consumes the first value, separately from the candidate budget.
        source = SequenceValues([0.0, 2.0])
        nuts(Xoshiro(468), finite_region(), 3; init=FindFiniteDensity(source), kwargs...)
        @test take!(starts) == [2.0]
        @test source.next == 3

        source = SequenceValues([0.0, 0.0, 2.0])
        nuts(
            Xoshiro(468),
            finite_region(),
            3;
            init=FindFiniteDensity(source; max_attempts=2),
            kwargs...,
        )
        @test take!(starts) == [2.0]
        @test source.next == 4

        source = SequenceValues([0.0, 0.0, 2.0])
        @test_throws ArgumentError nuts(
            Xoshiro(468),
            finite_region(),
            3;
            init=FindFiniteDensity(source; max_attempts=1),
            kwargs...,
        )
        @test source.next == 3
        @test !isready(starts)

        source = SequenceValues([0.0, NaN, 2.0])
        nuts(
            Xoshiro(468),
            flat_parameter(),
            3;
            init=FindFiniteDensity(source; max_attempts=2),
            kwargs...,
        )
        @test take!(starts) == [2.0]
        @test_throws ArgumentError FindFiniteDensity(FromPrior(); max_attempts=0)
    end
end

end
