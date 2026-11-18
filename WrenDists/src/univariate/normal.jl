"""
    Normal(μ, σ)
    Normal(μ)
    Normal()

Normal distribution with mean `μ` (default 0.0) and standard deviation `σ` (default 1.0).

**Customised batched implementations**: Yes
"""
struct Normal{M<:Number,S<:Number} <: UnconstrainedUnivariateDist
    μ::M
    σ::S
end

Normal(μ) = Normal(μ, one(μ))
Normal() = Normal(0.0, 1.0)

@inline zval(d::Normal, x::Number) = (x - d.μ) / d.σ
@inline xval(d::Normal, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Normal, x::Number)
    z = zval(d, x)
    # Match `σ` to `z` so integer parameters do not reduce `BigFloat` precision.
    σ = oftype(z, d.σ)
    return -(z^2 + log2π) / 2 - log(σ)
end

# Apply `μ` and `σ` after drawing noise so automatic differentiation can follow them.
@inline function Base.rand(rng::AbstractRNG, d::Normal)
    return xval(d, randn(rng, rngtype(d)))
end

# --- Batched Normal distributions ---

# Compute the shared normalising constant once.
function _iid_logdensityof(dist::Normal, x::AbstractArray)
    μ = dist.μ
    _, _, σ = promote(zero(eltype(x)), μ, dist.σ)
    inv_σ = inv(float(σ))
    z2 = sum(xi -> abs2((xi - μ) * inv_σ), x; init=zero(inv_σ))
    lognorm = logdensityof(dist, oftype(z2, μ))
    return length(x) * lognorm - z2 / 2
end

# One `randn` call for the whole array is faster than drawing element by element.
function Base.rand(rng::AbstractRNG, d::IID{<:Normal})
    dist = _dist(d)
    return xval.(Ref(dist), randn(rng, rngtype(dist), _dims(d)))
end

using FillArrays: AbstractFill, getindex_value

struct BatchedNormal{M<:AbstractArray,S<:AbstractArray} <: BatchedUnivariate
    mu::M
    sigma::S
end
Base.size(d::BatchedNormal) = size(d.mu)
# Convert noise so numeric wrappers need no typed randn method.
Base.rand(rng::Random.AbstractRNG, d::BatchedNormal) =
    convert.(noisetype(d), randn(rng, size(d.mu))) .* d.sigma .+ d.mu

function DensityInterface.logdensityof(d::BatchedNormal, x::AbstractArray{<:Number})
    _check_size(d, x)
    # Benchmarks @ https://github.com/JuliaBayes/Wren.jl/issues/99
    mu, sigma = d.mu, d.sigma
    # Promote so that integer data or parameters still give a float result.
    T = float(promote_type(eltype(x), eltype(mu), eltype(sigma)))
    lognorm = length(x) * convert(T, log2π) / 2
    return if within_compile()
        z = (x .- mu) ./ sigma
        -sum(abs2, z) / 2 - sum(log, sigma) - lognorm
    else
        s = zero(T)
        for i in eachindex(x)
            z = (x[i] - mu[i]) / sigma[i]
            s -= abs2(z) / 2 + log(sigma[i])
        end
        s - lognorm
    end
end

# A shared `sigma` needs one `log` and one division for the whole batch.
function DensityInterface.logdensityof(
    d::BatchedNormal{<:AbstractArray,<:AbstractFill},
    x::AbstractArray{<:Number},
)
    _check_size(d, x)
    mu, sigma = d.mu, getindex_value(d.sigma)
    T = float(promote_type(eltype(x), eltype(mu), typeof(sigma)))
    sqdist = if within_compile()
        sum(abs2, x .- mu)
    else
        s = zero(T)
        for i in eachindex(x, mu)
            s += abs2(x[i] - mu[i])
        end
        s
    end
    return -sqdist / (2 * abs2(sigma)) - length(x) * (log(sigma) + convert(T, log2π) / 2)
end

# The Plaice interface is shared with the other batched distributions; see batched.jl.
