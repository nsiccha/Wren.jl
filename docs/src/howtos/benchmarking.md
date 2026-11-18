# [Benchmarking](@id benchmarking)

## `@bmodel`

The easiest way to benchmark models is to use the [`@bmodel`](@ref) macro:

```@example benchmarking
using Wren, Wren.Dists

@wren function f()
    x ~ Exponential()
    z ~ MvNormalScale(zeros(3), abs(x) * I)
end

@bmodel f()
```

When passed a single argument, `@bmodel` will benchmark:

  - `rand`: the time it takes to sample a set of parameters from the model
  - `logjoint`: the time it takes to compute the log joint probability of a set of parameters
  - `lp`: the time it takes to compute the log-joint probability, but with [a vectorised interface](@ref vectorisation) (this is the same as e.g. `LogDensityProblems.logdensity`, if you are familiar with that interface).

Note that all of these report the **fastest** time out of all the samples, since that varies the least between runs.

To access the data in this benchmark, you can directly access the fields of the [`BenchmarkResult`](@ref) object.
The relevant fields are named `{function_name}_bench`:

```@example benchmarking
br = @bmodel f()

br.lp_bench # This is a `Chairmarks.Sample`
```

```@example benchmarking
br.lp_bench.time, br.lp_bench.allocs
```

## AD

To additionally benchmark gradients, you can pass an `AbstractADType` as the second argument to `@bmodel`:

```@example benchmarking
using Enzyme

# `Wren.DEFAULT_AD` is the same as `AutoEnzyme(; mode=Reverse)`.
@bmodel f() DEFAULT_AD
```

In addition to the three columns above, this will also benchmark `lp_and_grad`, which is the time taken to compute the log-joint plus its gradient (i.e., `LogDensityProblems.logdensity_and_gradient`).

## Pitfalls

If you are trying to do comparative benchmarking i.e. comparing performance before and after, you may very well need to fix the number of BLAS threads to get consistent results.
You can do so either by calling `LinearAlgebra.BLAS.set_num_threads(1)`, or by setting the `OPENBLAS_NUM_THREADS` environment variable before starting Julia.

This recommendation in the context of Wren is mostly based on anecdotal evidence, but [BenchmarkTools.jl also recommends this](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/#Miscellaneous-tips-and-info): see also [this thread on Julia Discourse](https://discourse.julialang.org/t/regarding-the-multithreaded-performance-of-openblas/75450) for more info.

## More control

`@bmodel` is shorthand for the [`benchmark_model`](@ref) function, which also accepts keyword arguments to control how the model is vectorised (for example, which parameters to benchmark at).

The `latent_source` keyword chooses the benchmark point, and it is used exactly as given: a value supplied in unconstrained space, such as `TransformedValue(v, AsUnconstrainedVec())`, is benchmarked at exactly `v`.
The point is available afterwards as `result.vecmodel.sample`.
If you have already constructed a [`VecModel`](@ref), `benchmark_model(vecmodel)` benchmarks it at `vecmodel.sample`, using the AD backend that it was constructed with.

## Testing AD correctness

To check that an AD backend gives the right log-density and gradient, use [`test_ad`](@ref).
By default this compares against ForwardDiff, but you can also compare against another backend with [`WithBackend`](@ref), or against known values with [`WithExpectedResult`](@ref).
If the results don't match, an [`ADIncorrectException`](@ref) is thrown.

## Benchmarks over time

Wren has a benchmarking suite which runs several PosteriorDB models, plus their gradients under Enzyme and Mooncake.
The code for this is in `bench/suite`, and you can view the results at [https://juliabayes.org/Wren.jl/bench](https://juliabayes.org/Wren.jl/bench).

This suite is currently run manually by the maintainers in order to guarantee that the results are at least somewhat consistent from run to run.
(On GitHub Actions, the results are not consistent enough to be useful, since the underlying hardware is not guaranteed to be the same from run to run.)
