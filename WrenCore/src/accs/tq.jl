const TQ_ACCNAME = :TrackedQuantities

"""
    TrackedQuantitySpec

Selects which tracked quantities (`@track` statements) are evaluated and recorded. One of
[`AllQuantities`](@ref), [`SomeQuantities`](@ref) or [`NoQuantities`](@ref).
"""
abstract type TrackedQuantitySpec end

"""
    AllQuantities()

Select every tracked quantity.

See also: [`SomeQuantities`](@ref), [`NoQuantities`](@ref)
"""
struct AllQuantities <: TrackedQuantitySpec end

"""
    NoQuantities()

Select no tracked quantities.

See also: [`AllQuantities`](@ref), [`SomeQuantities`](@ref)
"""
struct NoQuantities <: TrackedQuantitySpec end

"""
    SomeQuantities(names::Tuple{Vararg{Symbol}})

Select the tracked quantities named in `names`. The name of a submodel, such as `:a` in
`a ~ sub(...)`, selects every tracked quantity of that submodel.

See also: [`AllQuantities`](@ref), [`NoQuantities`](@ref)
"""
struct SomeQuantities{names} <: TrackedQuantitySpec
    SomeQuantities(names::Tuple{Vararg{Symbol}}) = new{names}()
end

_tq_requested(::AllQuantities, ::Val) = true
_tq_requested(::NoQuantities, ::Val) = false
_tq_requested(::SomeQuantities{names}, ::Val{sym}) where {names,sym} = sym in names

"""
    TrackedQuantityAccumulator(spec::TrackedQuantitySpec=AllQuantities())

Record the values of the `@track` statements selected by `spec` in a `VarNamedTuple`.

See also: [`TrackedQuantitySpec`](@ref)
"""
struct TrackedQuantityAccumulator{S<:TrackedQuantitySpec,V<:VarNamedTuple} <:
       AbstractAccumulator
    spec::S
    values::V
end
function TrackedQuantityAccumulator(spec::TrackedQuantitySpec=AllQuantities())
    return TrackedQuantityAccumulator(spec, VarNamedTuple())
end

function _tq_requested(acc::TrackedQuantityAccumulator, sym::Val)
    return _tq_requested(acc.spec, sym)
end

"""
    is_accumulating_tq(accs::AccumulatorTuple, ::Val{sym})

Return `true` if `accs` contains a [`TrackedQuantityAccumulator`](@ref) that selects `sym`.
"""
function is_accumulating_tq(accs::AccumulatorTuple, sym::Val)
    hasacc(accs, Val(TQ_ACCNAME)) || return false
    return _tq_requested(getacc(accs, Val(TQ_ACCNAME)), sym)
end

accumulator_name(::Type{<:TrackedQuantityAccumulator}) = TQ_ACCNAME

function Base.copy(acc::TrackedQuantityAccumulator)
    return TrackedQuantityAccumulator(acc.spec, copy(acc.values))
end

function accumulate_assume!!(
    acc::TrackedQuantityAccumulator,
    val,
    tval,
    vn,
    right,
    logjac,
    template,
    @nospecialize(sitename::Val),
)
    return acc
end
function accumulate_observe!!(
    acc::TrackedQuantityAccumulator,
    val,
    vn,
    right,
    template,
    @nospecialize(sitename::Val),
)
    return acc
end

"""
    accumulate_tq!!(acc::AbstractAccumulator, value, vn::VarName)

Update and then return `acc` when a `@track` statement records `value` under `vn`.
"""
accumulate_tq!!(acc::AbstractAccumulator, value, vn::VarName) = acc
function accumulate_tq!!(acc::TrackedQuantityAccumulator, value, vn::VarName)
    new_values = VarNames.templated_setindex!!(acc.values, value, vn, VarNames.NoTemplate())
    return TrackedQuantityAccumulator(acc.spec, new_values)
end

"""
    accumulate_tq!!(at::AccumulatorTuple, value, vn::VarName)

Update all the accumulators of `at` via `accumulate_tq!!`.
"""
@generated function accumulate_tq!!(
    at::AccumulatorTuple{N,T},
    value,
    vn::VarName,
) where {N,T}
    names = fieldnames(T)
    calls = [:(accumulate_tq!!(at.nt.$(names[i]), value, vn)) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        AccumulatorTuple(NamedTuple{$names}(($(calls...),)))
    end
end

function _zero(acc::TrackedQuantityAccumulator)
    return TrackedQuantityAccumulator(acc.spec, empty(acc.values))
end
reset(acc::TrackedQuantityAccumulator) = _zero(acc)

function scope(acc::TrackedQuantityAccumulator, ::VarName{sym}) where {sym}
    submodel_spec = _tq_requested(acc, Val{sym}()) ? AllQuantities() : NoQuantities()
    return TrackedQuantityAccumulator(submodel_spec, acc.values)
end

function combine_from_scope(
    parent::TrackedQuantityAccumulator,
    child::TrackedQuantityAccumulator,
    vn::VarName,
    template,
)
    isempty(child.values) && return parent
    nested_values =
        VarNames.templated_setindex!!(VarNamedTuple(), child.values, vn, template)
    return TrackedQuantityAccumulator(parent.spec, merge(parent.values, nested_values))
end
