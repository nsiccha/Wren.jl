"""
    BernoulliLogit(logitp)
    BernoulliLogit()

Bernoulli distribution with success probability `logistic(logitp)` (default `logitp = 0.0`,
i.e. probability 0.5).

**Customised batched implementations**: Yes
"""
struct BernoulliLogit{P<:Number} <: DiscreteUnivariateDist
    logitp::P
end

BernoulliLogit() = BernoulliLogit(0.0)

Base.eltype(::Type{BernoulliLogit{P}}) where {P} = float(P)

# `log(logistic(η)) = -log1pexp(-η)` and `log(1 - logistic(η)) = -log1pexp(η)`.
@inline function bernoullilogit_logmass(η::T, x::Number) where {T<:Number}
    return select(
        iszero(x) | isone(x),
        () -> -log1pexp(ifelse(isone(x), -η, η)),
        () -> convert(T, -Inf),
    )
end

@inline function DensityInterface.logdensityof(d::BernoulliLogit, x::Number)
    T = masstype(d, x)
    η = convert(T, d.logitp)
    return bernoullilogit_logmass(η, x)
end

@inline Base.rand(rng::AbstractRNG, d::BernoulliLogit) =
    rand(rng, Bernoulli(logistic(d.logitp)))

# --- Batched BernoulliLogit ---

struct BatchedBernoulliLogit{P<:AbstractArray} <: DiscreteBatchedUnivariate
    logitp::P
end
Base.size(d::BatchedBernoulliLogit) = size(d.logitp)
function Base.rand(rng::AbstractRNG, d::BatchedBernoulliLogit)
    return rand(rng, BatchedBernoulli(logistic.(d.logitp)))
end
function DensityInterface.logdensityof(d::BatchedBernoulliLogit, x)
    η = d.logitp
    return if within_compile()
        # Promote as the scalar and native paths do, so integer logits have a float
        # mass and Float32 logits keep the precision of Float64 outcomes.
        η = convert.(float(promote_type(eltype(x), eltype(η))), η)
        sum(bernoullilogit_logmass.(η, x))
    else
        T = float(promote_type(eltype(x), eltype(η)))
        s = zero(T)
        for i in eachindex(x, η)
            ηi = convert(T, η[i])
            s += bernoullilogit_logmass(ηi, x[i])
        end
        s
    end
end
