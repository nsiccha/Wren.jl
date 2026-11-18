# [Distributions](@id distributions)

To define Wren models, you will (probably...) need to use a few probability distributions.

## [Built-in distributions](@id wrendists)

Wren ships its own set of distributions, which is currently not very extensive, but is designed to satisfy most common use cases for probabilistic programming
(Help with expanding this set is **very** welcome!)
You can see a full list of the built-in distributions [on this page](@ref reference-distributions).

To access these, you need to add `using Wren.Dists` to your code.
(The reason Wren doesn't automatically re-export them is to avoid name clashes if you want to load e.g. Distributions.jl.)

**In the first instance, we very much recommend trying these first.**
Their constructors are designed to be as close as possible to the equivalent constructors in Distributions.jl.
The reason for recommending Wren's built-in distributions is because they:

  - work with Wren's `@batch` macro in a meaningful way (i.e., Wren provides vectorised equivalents of them);

  - are compatible with Reactant (many Distributions.jl objects use scalar indexing and have type bounds on e.g. `<:Real`, which makes them incompatible); and

  - minimise code paths that can potentially cause errors with Enzyme.

!!! note "Role in ecosystem"

    Wren's own distributions are not intended to be a fully-fledged replacement for e.g. Distributions.jl.
    For example, a huge swathe of Distributions' API is simply unimplemented.

    In many ways, Wren's distributions are meant to be a proof-of-concept that such a system can be used for Wren's PPL aims.
    We would be very happy to retire the distributions defined in Wren if there was a centralised effort to develop GPU-friendly distributions that could be used across multiple PPLs, including Wren.

## Distributions.jl

You can also use all the distributions from [Distributions.jl](https://juliastats.org/Distributions.jl/stable/), as they already obey the interface above.

This is useful if you just want a familiar interface and don't need any of the features of Wren's built-in distributions.
Indeed, Wren promises that `@batch` will still work _correctly_ with these distributions; it's just that it will fall back to using `product_distribution`, which won't give you any speedups over the unbatched loop.

For a variance prior with shape 2 and scale 3, you can use
`Distributions.InverseGamma` directly:

```julia
using WrenCore
using Distributions: InverseGamma

@wren function variance_prior()
    variance ~ InverseGamma(2.0, 3.0)
end

parameters(variance_prior())
```

The sampled variance is positive. Wren's built-in `WrenDists.InverseGamma`
uses the same shape and scale convention.

## Custom distributions

Wren has a fully generic interface which allows you to create custom distributions so long as they provide:

  - `Base.rand(rng, dist)`
  - `DensityInterface.logdensityof(dist, sample)`

For more information and a worked example, please see the [custom distributions page](@ref custom-distributions).
