"""
    Laplace(μ, b)
    Laplace(μ)
    Laplace()

Laplace distribution with location `μ` (default 0.0) and scale `b` (default 1.0).

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct Laplace{M<:Number,B<:Number} <: UnconstrainedUnivariateDist
    μ::M
    b::B
end

Laplace(μ) = Laplace(μ, one(μ))
Laplace() = Laplace(0.0, 1.0)

@inline zval(d::Laplace, x::Number) = (x - d.μ) / d.b
@inline xval(d::Laplace, z::Number) = muladd(d.b, z, d.μ)

@inline function DensityInterface.logdensityof(d::Laplace, x::Number)
    z = zval(d, x)
    # Match `b` to `z` so integer parameters do not reduce `BigFloat` precision.
    b = oftype(z, d.b)
    return -abs(z) - logtwo - log(b)
end

# The difference of two unit exponential samples follows Laplace(0, 1).
# Applying `μ` and `b` afterward lets automatic differentiation pass through the sample.
@inline function Base.rand(rng::AbstractRNG, d::Laplace)
    T = noisetype(d)
    e₁ = -log1p(-rand(rng, T))
    e₂ = -log1p(-rand(rng, T))
    return xval(d, e₁ - e₂)
end
