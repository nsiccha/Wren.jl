"""
    Binomial(n, p)
    Binomial(n)
    Binomial()

Binomial distribution for `n` (default 1) trials with success probability `p` (default 0.5).

**Customised batched implementations**: Yes
"""
struct Binomial{N<:Integer,P<:Number} <: DiscreteUnivariateDist
    n::N
    p::P
end

Binomial(n::Integer) = Binomial(n, 0.5)
Binomial() = Binomial(1, 0.5)

Base.eltype(::Type{Binomial{N,P}}) where {N,P} = float(P)

@inline function insupport(d::Binomial, x::Number)
    return isinteger(x) & (x >= zero(x)) & (x <= d.n)
end

@inline _clamp_to_nonneg(n::N) where {N<:Number} = max(n, zero(N))
@inline _add_one(n::N) where {N<:Number} = n + one(N)
@inline function logbinom(n::Number, k::Number)
    nc, kc, mc = _clamp_to_nonneg(n), _clamp_to_nonneg(k), _clamp_to_nonneg(n - k)
    return loggamma(_add_one(nc)) - loggamma(_add_one(kc)) - loggamma(_add_one(mc))
end

@inline function DensityInterface.logdensityof(d::Binomial, x::Number)
    T = masstype(d, x)
    n, k, p = convert(T, d.n), convert(T, x), convert(T, d.p)
    # Skip zero terms so `p = 0` and `p = 1` work.
    a = select(k == zero(T), () -> zero(T), () -> k * log(p))
    b = select(k == n, () -> zero(T), () -> (n - k) * log1p(-p))
    return select(insupport(d, x), () -> logbinom(n, k) + a + b, () -> convert(T, -Inf))
end

# Add `n` Bernoulli samples. This fixed-length loop also works with tracing tools and
# on GPUs.
@inline function Base.rand(rng::AbstractRNG, d::Binomial)
    T = eltype(d)
    s = zero(T)
    for _ in 1:(d.n)
        u = rand(rng, noisetype(d))
        s += select(u < d.p, () -> one(T), () -> zero(T))
    end
    return s
end

# --- Batched Binomial ---

struct BatchedBinomial{N<:AbstractArray,P<:AbstractArray} <: DiscreteBatchedUnivariate
    n::N
    p::P
end
Base.size(d::BatchedBinomial) = size(d.p)
Base.rand(rng::AbstractRNG, d::BatchedBinomial) = rand.(Ref(rng), Binomial.(d.n, d.p))
function DensityInterface.logdensityof(d::BatchedBinomial, x)
    # Benchmarks @ https://github.com/JuliaBayes/Wren.jl/issues/103
    n, p = d.n, d.p
    return if within_compile()
        valid = @. isinteger(x) & (x >= zero(x)) & (x <= n)
        k = @. ifelse(valid, x, zero(x))
        p = @. ifelse(valid, p, one(p) / 2)
        logp = @. logbinom(n, k) + xlogy(k, p) + xlogy(n - k, one(k) - p)
        sum(@. ifelse(valid, logp, oftype(logp, -Inf)))
    else
        T = float(promote_type(eltype(x), eltype(n), eltype(p)))
        s = zero(T)
        valid = true
        for i in eachindex(x)
            # don't return -Inf inside the loop: see #317
            valid &= isinteger(x[i]) & (0 <= x[i]) & (x[i] <= n[i])
            ni, ki, pi = convert(T, n[i]), convert(T, x[i]), convert(T, p[i])
            s += logbinom(ni, ki) + xlogy(ki, pi) + xlogy(ni - ki, one(T) - pi)
        end
        ifelse(valid, s, convert(T, -Inf))
    end
end
