"""
    Categorical(p)

Categorical distribution over `1:length(p)` with probability vector `p`.

**Customised batched implementations**: Yes
"""
struct Categorical{V<:AbstractVector{<:Number}} <: DiscreteUnivariateDist
    p::V
end

Base.eltype(::Type{Categorical{V}}) where {V} = float(eltype(V))

@inline function massat(d::Categorical, x::Number)
    T = masstype(d, x)
    return if within_compile()
        convert(T, sum(d.p .* ((1:length(d.p)) .== x)))
    else
        m = zero(T)
        for (i, pᵢ) in enumerate(d.p)
            m += select(x == i, () -> convert(T, pᵢ), () -> zero(T))
        end
        m
    end
end

@inline function DensityInterface.logdensityof(d::Categorical, x::Number)
    return log(massat(d, x))
end

# Some tools cannot stop a loop based on `q`, so count every partial sum below it.
@inline function _categorical_quantile(d::Categorical, q::Number)
    T = masstype(d, q)
    total = zero(T)
    i = one(T)
    for pᵢ in d.p
        total += convert(T, pᵢ)
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    return min(i, convert(T, length(d.p)))
end

# A sample changes in steps as the probabilities change, so its derivative is zero
# almost everywhere.
@inline function Base.rand(rng::AbstractRNG, d::Categorical)
    return _categorical_quantile(d, rand(rng, noisetype(d)))
end

# --- Batched Categorical ---
# P is a (K, N) matrix: K categories × N batch elements, each column a probability vector.

struct BatchedCategorical{P<:AbstractMatrix} <: DiscreteBatchedUnivariate
    P::P
end
Base.size(d::BatchedCategorical) = (size(d.P, 2),)
function Base.rand(rng::AbstractRNG, d::BatchedCategorical)
    K, N = size(d.P)
    x = Vector{Int}(undef, N)
    for i in 1:N
        x[i] = _categorical_quantile(Categorical(@view(d.P[:, i])), rand(rng, noisetype(d)))
    end
    return x
end
function DensityInterface.logdensityof(d::BatchedCategorical, x)
    P = d.P
    return if within_compile()
        K = size(P, 1)
        T = eltype(P)
        H = T.(1:K) .== reshape(x, 1, :)
        sum(log.(sum(P .* H; dims=1)))
    else
        T = float(eltype(P))
        s = zero(T)
        valid = true
        for i in eachindex(x)
            # don't return -Inf inside the loop: see #317
            k = x[i]
            validi = isinteger(k) & (1 <= k) & (k <= size(P, 1))
            valid &= validi
            # if `k` is invalid, we will return -Inf at the end of the loop; so the
            # one(T) is just a placeholder.
            s += log(validi ? convert(T, P[Int(k), i]) : one(T))
        end
        ifelse(valid, s, convert(T, -Inf))
    end
end
