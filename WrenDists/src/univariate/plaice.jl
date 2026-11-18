# Plaice interface for univariate distributions.
# See Plaice.jl/src/univariate/univariate.jl for the OnlyWrap/VectWrap pattern.

Plaice.from_unconstrained_vec(d::UnivariateDist) =
    Plaice.OnlyWrap(Plaice.inverse(Plaice.scalar_to_scalar_bijector(d)))
Plaice.to_unconstrained_vec(d::UnivariateDist) =
    Plaice.VectWrap(Plaice.scalar_to_scalar_bijector(d))
Plaice.from_vec(::UnivariateDist) = Plaice.OnlyWrap(Plaice.TypedIdentity())
Plaice.to_vec(::UnivariateDist) = Plaice.VectWrap(Plaice.TypedIdentity())

Plaice.vec_length(::UnivariateDist) = 1
Plaice.unconstrained_vec_length(::UnivariateDist) = 1

Plaice.optic_vec(::UnivariateDist) = [VarNames.Iden()]
Plaice.unconstrained_optic_vec(::UnivariateDist) = [VarNames.Iden()]

# scalar_to_scalar_bijector by abstract type
Plaice.scalar_to_scalar_bijector(::UnconstrainedUnivariateDist) = Plaice.TypedIdentity()
Plaice.scalar_to_scalar_bijector(::DiscreteUnivariateDist) = Plaice.TypedIdentity()
Plaice.scalar_to_scalar_bijector(::PositiveUnivariateDist) = Plaice.Log(0.0, 1)

# Bounded distributions use Untruncate
Plaice.scalar_to_scalar_bijector(d::Beta) =
    Plaice.Untruncate(zero(noisetype(d)), one(noisetype(d)))
Plaice.scalar_to_scalar_bijector(d::Uniform) = Plaice.Untruncate(d.a, d.b)
Plaice.scalar_to_scalar_bijector(d::TruncatedFlat) =
    Plaice.Untruncate(d.lower, d.upper)
