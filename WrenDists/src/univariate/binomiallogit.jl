"""
    BinomialLogit(n, logitp)
    BinomialLogit(n)
    BinomialLogit()

Binomial distribution for `n` (default 1) trials with success probability
`logistic(logitp)` (default `logitp = 0.0`, i.e. probability 0.5).

**Customised batched implementations**: Yes
"""
struct BinomialLogit{N<:Integer,P<:Number} <: DiscreteUnivariateDist
    n::N
    logitp::P
end

BinomialLogit(n::Integer) = BinomialLogit(n, 0.0)
BinomialLogit() = BinomialLogit(1, 0.0)

Base.eltype(::Type{BinomialLogit{N,P}}) where {N,P} = float(P)

@inline function insupport(d::BinomialLogit, x::Number)
    return isinteger(x) & (x >= zero(x)) & (x <= d.n)
end

# For η ≥ 0: -n·log1pexp(-η) - (n - k)·η; for η < 0: k·η - n·log1pexp(η).
# Taking log1pexp of -|η| avoids overflow; masking c·a avoids 0·Inf at the endpoints.
@inline function binomiallogit_logmass(n::Number, k::Number, η::AbstractFloat)
    positive = η >= zero(η)
    a = ifelse(positive, -η, η)
    c = ifelse(positive, n - k, k)
    return logbinom(n, k) + c * ifelse(iszero(c), zero(a), a) - n * log1pexp(a)
end

# The logit terms use -k·log1pexp(-η) - (n-k)·log1pexp(η).
# Count-only masks keep cached AD tapes valid when the logit changes sign.
@inline function binomiallogit_logmass(n::Number, k::Number, η::Number)
    η = float(η)
    m = n - k
    # Mask before softplus so endpoint counts also handle infinite logits.
    a = ifelse(iszero(k), zero(η), -η)
    b = ifelse(iszero(m), zero(η), η)
    return logbinom(n, k) - k * log1pexp(a) - m * log1pexp(b)
end

@inline function DensityInterface.logdensityof(d::BinomialLogit, x::Number)
    T = masstype(d, x)
    n, k, η = convert(T, d.n), convert(T, x), convert(T, d.logitp)
    value = binomiallogit_logmass(n, k, η)
    return select(insupport(d, x), () -> value, () -> convert(T, -Inf))
end

@inline function Base.rand(rng::AbstractRNG, d::BinomialLogit)
    return rand(rng, Binomial(d.n, logistic(d.logitp)))
end

# --- Batched BinomialLogit ---

struct BatchedBinomialLogit{N<:AbstractArray,P<:AbstractArray} <: DiscreteBatchedUnivariate
    n::N
    logitp::P
end
Base.size(d::BatchedBinomialLogit) = size(d.logitp)
function Base.rand(rng::AbstractRNG, d::BatchedBinomialLogit)
    return rand.(Ref(rng), BinomialLogit.(d.n, d.logitp))
end
function DensityInterface.logdensityof(d::BatchedBinomialLogit, x)
    n, η = d.n, d.logitp
    return if within_compile()
        valid = @. isinteger(x) & (x >= zero(x)) & (x <= n)
        k = @. ifelse(valid, x, zero(x))
        η = @. ifelse(valid, η, zero(η))
        logp = binomiallogit_logmass.(n, k, η)
        sum(@. ifelse(valid, logp, oftype(logp, -Inf)))
    else
        T = float(promote_type(eltype(x), eltype(n), eltype(η)))
        s = zero(T)
        valid = true
        for i in eachindex(x, n, η)
            # don't return -Inf inside the loop: see #317
            valid &= isinteger(x[i]) & (0 <= x[i]) & (x[i] <= n[i])
            ni, ki, ηi = convert(T, n[i]), convert(T, x[i]), convert(T, η[i])
            s += binomiallogit_logmass(ni, ki, ηi)
        end
        ifelse(valid, s, convert(T, -Inf))
    end
end
