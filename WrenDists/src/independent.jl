"""
    Repeated{D,N}

A single measure `dist` repeated over `N` extra dimensions `dims`. Used as the component
collection of an [`Independent`](@ref) built by `WrenCore.iid`.
"""
struct Repeated{D<:WrenDist,N}
    dist::D
    dims::Dims{N}
end
Base.size(r::Repeated) = (size(r.dist)..., r.dims...)
Base.:(==)(a::Repeated, b::Repeated) = (a.dist == b.dist) & (a.dims == b.dims)

"""
    Independent{C}

The product measure of a collection `components` of measures. Construct it with
`WrenCore.independent(dists::AbstractArray)` or `WrenCore.iid(dist, dims)`.
"""
struct Independent{C}
    components::C
end

const IID{D,N} = Independent{Repeated{D,N}}
const IIDUnivariate = Independent{<:Repeated{<:UnivariateDist}}
const IIDMultivariate = Independent{<:Repeated{<:MultivariateDist}}
const IndependentArray = Independent{<:AbstractArray{<:UnivariateDist}}

_dist(d::IID) = d.components.dist
_dims(d::IID) = d.components.dims
Base.show(io::IO, d::IID) = print(io, "iid(", _dist(d), ", ", _dims(d), ")")

Base.size(::UnivariateDist) = ()
Base.eltype(::Type{D}) where {D<:UnivariateDist} = basefloat(_promoted_paramtype(D))
Base.size(d::Independent) = size(d.components)
Base.:(==)(a::Independent, b::Independent) = a.components == b.components
Base.show(io::IO, d::Independent) = print(io, "independent(", d.components, ")")

Base.rand(rng::Random.AbstractRNG, d::IIDUnivariate) = rand(rng, _dist(d), _dims(d))
# Stack vector draws along the trailing dimensions.
function Base.rand(rng::Random.AbstractRNG, d::IIDMultivariate)
    draws = rand(rng, _dist(d), _dims(d))
    isempty(draws) && return Array{noisetype(_dist(d))}(undef, size(d))
    return stack(draws)
end
Base.rand(rng::Random.AbstractRNG, d::IndependentArray) =
    map(di -> rand(rng, di), d.components)
Base.rand(d::Independent) = rand(Random.default_rng(), d)

DensityInterface.DensityKind(::Independent) = DensityInterface.HasDensity()
function _check_size(d, x::AbstractArray)
    size(x) == size(d) ||
        throw(DimensionMismatch("expected size $(size(d)), got $(size(x))"))
    return nothing
end
function DensityInterface.logdensityof(d::IID, x::AbstractArray)
    _check_size(d, x)
    return _iid_logdensityof(_dist(d), x)
end
function DensityInterface.logdensityof(d::IndependentArray, x::AbstractArray)
    _check_size(d, x)
    return sum(logdensityof.(d.components, x))
end

"""
    _iid_logdensityof(dist, x::AbstractArray)

Sum of `logdensityof(dist, xᵢ)` over every draw `xᵢ` stored in `x`. Add methods to this
function to provide closed-form expressions for specific measures.
"""
function _iid_logdensityof(dist::UnivariateDist, x::AbstractArray)
    return sum(Base.Fix1(logdensityof, dist), x; init=_zero_logdensity(dist, x))
end

function _zero_logdensity(dist::WrenDist, x::AbstractArray)
    return zero(float(promote_type(eltype(x), noisetype(dist))))
end

# Each slice is one draw. This fallback is CPU-only.
function _iid_logdensityof(dist::MultivariateDist, x::AbstractArray{<:Any,M}) where {M}
    slices = eachslice(x; dims=ntuple(i -> i + 1, Val(M - 1)))
    return sum(Base.Fix1(logdensityof, dist), slices; init=_zero_logdensity(dist, x))
end

# --- ElementwiseLink / ElementwiseInvLink ---

# Apply one bijector or an array of bijectors, flattening on link and reshaping on inverse.
struct ElementwiseLink{B,N} <: Plaice.AbstractBijector
    bijector::B
    dims::Dims{N}
end
struct ElementwiseInvLink{B,N} <: Plaice.AbstractBijector
    bijector::B
    dims::Dims{N}
end

_map_inverse(b) = Plaice.inverse(b)
_map_inverse(b::AbstractArray) = Plaice.inverse.(b)
Plaice.inverse(t::ElementwiseLink) = ElementwiseInvLink(_map_inverse(t.bijector), t.dims)
Plaice.inverse(t::ElementwiseInvLink) = ElementwiseLink(_map_inverse(t.bijector), t.dims)

for T in (:ElementwiseLink, :ElementwiseInvLink)
    @eval begin
        Base.:(==)(a::$T, b::$T) = (a.bijector == b.bijector) & (a.dims == b.dims)
    end
end

_bcast(b) = Ref(b)
_bcast(b::AbstractArray) = b

# Compute Jacobians separately to avoid broadcasting tuples.
_ladj(b, x) = Plaice.logabsdet_jacobian(b, x)
_ladj(::Plaice.TypedIdentity, x) = zero(x)
_ladj(::Plaice.Exp, y) = y
_ladj(b::Plaice.Log, x) = -log(b.sign * (x - b.bound))
_ladj(b::Plaice.Truncate, y) = last(@inline Plaice.with_logabsdet_jacobian(b, y))
_apply(b, x) = b(x)

const IdentityLike = Union{Plaice.TypedIdentity,AbstractArray{Plaice.TypedIdentity}}

function Plaice.with_logabsdet_jacobian(t::ElementwiseLink, x::AbstractArray)
    b = _bcast(t.bijector)
    return vec(_apply.(b, x)), sum(_ladj.(b, x))
end
(t::ElementwiseLink)(x::AbstractArray) = vec(_apply.(_bcast(t.bijector), x))

function Plaice.with_logabsdet_jacobian(t::ElementwiseInvLink, y::AbstractVector)
    b = _bcast(t.bijector)
    yr = reshape(y, t.dims)
    return _apply.(b, yr), sum(_ladj.(b, yr))
end
(t::ElementwiseInvLink)(y::AbstractVector) = _apply.(_bcast(t.bijector), reshape(y, t.dims))

# Identity transforms only change the shape.
function Plaice.with_logabsdet_jacobian(
    ::ElementwiseLink{<:IdentityLike},
    x::AbstractArray{T},
) where {T<:Number}
    return vec(x), zero(T)
end
(::ElementwiseLink{<:IdentityLike})(x::AbstractArray) = vec(x)
function Plaice.with_logabsdet_jacobian(
    t::ElementwiseInvLink{<:IdentityLike},
    y::AbstractVector{T},
) where {T<:Number}
    return reshape(y, t.dims), zero(T)
end
(t::ElementwiseInvLink{<:IdentityLike})(y::AbstractVector) = reshape(y, t.dims)

# For Exp, the log-Jacobian is sum(y).
function Plaice.with_logabsdet_jacobian(
    t::ElementwiseInvLink{<:Plaice.Exp},
    y::AbstractVector,
)
    yr = reshape(y, t.dims)
    return _apply.(Ref(t.bijector), yr), sum(y)
end

# --- Plaice interface for Independent ---

Plaice.from_vec(d::Independent) = Plaice.Reshape(size(d))
Plaice.to_vec(d::Independent) = Plaice.Vec(size(d))

Base.length(d::Independent) = prod(size(d))
Plaice.vec_length(d::Independent) = length(d)
Plaice.unconstrained_vec_length(d::Independent) = length(d)
Plaice.unconstrained_vec_length(d::IIDMultivariate) =
    Plaice.unconstrained_vec_length(_dist(d)) * prod(_dims(d))

# Scalar transforms preserve the coordinate optics.
_index_optics(d) = [VarNames.Index(Tuple(I), (;)) for I in vec(CartesianIndices(size(d)))]
Plaice.optic_vec(d::Independent) = _index_optics(d)
Plaice.unconstrained_optic_vec(d::Independent) = _index_optics(d)
function Plaice.unconstrained_optic_vec(d::IIDMultivariate)
    return vec([
        foldl(Plaice.append_index, Tuple(I); init=optic) for
        optic in Plaice.unconstrained_optic_vec(_dist(d)), I in CartesianIndices(_dims(d))
    ])
end

function Plaice.from_unconstrained_vec(d::IIDUnivariate)
    return ElementwiseInvLink(
        Plaice.inverse(Plaice.scalar_to_scalar_bijector(_dist(d))),
        _dims(d),
    )
end
function Plaice.to_unconstrained_vec(d::IIDUnivariate)
    return ElementwiseLink(Plaice.scalar_to_scalar_bijector(_dist(d)), _dims(d))
end

Plaice.from_unconstrained_vec(d::IIDMultivariate) =
    Plaice.inverse(Plaice.to_unconstrained_vec(d))
function Plaice.to_unconstrained_vec(d::IIDMultivariate)
    bijector = Plaice.to_unconstrained_vec(_dist(d))
    bijector isa Plaice.TypedIdentity && return Plaice.Vec(size(d))
    iszero(prod(_dims(d))) && throw(
        ArgumentError(
            "empty constrained multivariate iid transforms are not supported by Plaice",
        ),
    )
    return Plaice.ProductVecTransform(
        Plaice.Elementwise(bijector, _dims(d)),
        nothing,
        size(_dist(d)),
    )
end

function Plaice.from_unconstrained_vec(d::IndependentArray)
    return ElementwiseInvLink(
        Plaice.inverse.(Plaice.scalar_to_scalar_bijector.(d.components)),
        size(d),
    )
end
function Plaice.to_unconstrained_vec(d::IndependentArray)
    return ElementwiseLink(Plaice.scalar_to_scalar_bijector.(d.components), size(d))
end
