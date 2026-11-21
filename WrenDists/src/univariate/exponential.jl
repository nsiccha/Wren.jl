"""
    Exponential(θ)
    Exponential()

Exponential distribution with scale `θ` (default 1.0).

**Customised batched implementations**: Yes
"""
struct Exponential{R<:Number} <: PositiveUnivariateDist
    θ::R
end

Exponential() = Exponential(1.0)

@inline sval(d::Exponential, x::Number) = x / d.θ

@inline function DensityInterface.logdensityof(d::Exponential, x::Number)
    s = sval(d, x)
    θ = oftype(s, d.θ)
    return select(x >= zero(x), () -> -(s + log(θ)), () -> oftype(float(s), -Inf))
end

# Apply `θ` after drawing noise so automatic differentiation can follow it.
@inline function Base.rand(rng::AbstractRNG, d::Exponential)
    e = -log(rand(rng, noisetype(d)))
    return d.θ * e
end

# --- Batched Exponential ---

struct BatchedExponential{R<:AbstractArray} <: BatchedUnivariate
    θ::R
end
Base.size(d::BatchedExponential) = size(d.θ)
function Base.rand(rng::AbstractRNG, d::BatchedExponential)
    return .-log.(rand(rng, rngtype(d), size(d))) .* d.θ
end
function DensityInterface.logdensityof(d::BatchedExponential, x)
    # Benchmarks @ https://github.com/JuliaBayes/Wren.jl/issues/105
    θ = d.θ
    return if within_compile()
        value = sum(@. -(x / θ) - log(θ))
        ifelse(all(x .>= 0), value, oftype(value, -Inf))
    else
        T = float(promote_type(eltype(x), eltype(θ)))
        s = zero(T)
        valid = true
        for i in eachindex(x, θ)
            # don't return -Inf inside the loop: see #317
            valid &= x[i] >= zero(x[i])
            s -= convert(T, x[i]) / convert(T, θ[i]) + log(convert(T, θ[i]))
        end
        ifelse(valid, s, convert(T, -Inf))
    end
end
Plaice.scalar_to_scalar_bijector(::BatchedExponential) = Plaice.Log(0.0, 1)
