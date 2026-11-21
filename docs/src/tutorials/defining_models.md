# [Defining models](@id defining-models)

On this page, we show how to define a probabilistic model with basic Wren syntax.

This page assumes that you know what a probabilistic model and its various components (e.g. parameters, observations) are.
If you are not familiar with these concepts, you can read our [introduction to probabilistic programming](@ref beginner) first.

## Model bodies are functions

Wren allows you to define a probabilistic model as a function, but prefixed with the `@wren` macro:

```@example defining
using Wren, Wren.Dists

@wren function my_model(m)
    x ~ Normal(m)
    return nothing
end
```

The function body can be thought of as the probabilistic model itself: it provides a complete specification of how to perform forward simulation.
Here, our forward simulation says: draw a random quantity `x` from a normal distribution centred at `m`, and then return `nothing`.

The body of a Wren model is plain Julia code, with some extra constructs that allow you to define probabilistic aspects of the model (which we'll explain later on this page).
A corollary of this is that you can use almost any library inside a Wren model.

!!! note "Function arguments are not special"

    If you have used Turing.jl before, you may know that arguments have a special meaning there in that they are observations (unless passed as `missing`).

    This is not the case in Wren.
    In Wren, **arguments have no special meaning**: they are just quantities that can be used inside the model body.

## Instantiating a model

To actually run this model, you will first have to create a [`Model`](@ref) struct, which can be thought of as Wren's own representation of a probabilistic model.
To create one of these `Model`s, you need to call the function, passing in any required arguments.

Calling a normal Julia function will run its body and give you its return value (here `nothing`).
However, with the `@wren` macro, this is different: the function is rewritten such that calling it instead returns a `Model` object.

```@example defining
model = my_model(0.0)
```

As we can see, this `Model` is a wrapper around its own arguments (here `m = 0.0`), plus the function body.
(We'll come to `conditioned` later on this page.)
At this point in time, the model itself has not been run at all.
We will be able to do so by passing the newly created `Model` to all the APIs of Wren.

On this page, we'll demonstrate only one function for interacting with a `Model`, which is [`simulate`](@ref).
When given a `Model` object, this performs the forward simulation of the probabilistic model, returning a [`ModelResult`](@ref) struct that contains all the outputs we might be interested in.

```@example defining
simulate(model)
```

This includes the parameters, associated probabilities, as well as the original return value of the function body.

## Random quantities

The most important custom syntax that Wren provides is the tilde operator (`~`).
Much like the corresponding mathematical notation, `x ~ dist` means that `x` is a random quantity drawn from the distribution `dist`.

### Left-hand sides

The left-hand side of a tilde can be either

  - a top-level variable, e.g., `x ~ Normal()`,
  - a field, e.g., `m.x ~ Normal()`,
  - an array element, e.g., `a[1] ~ Normal()`,
  - or chains thereof, e.g., `m.a[1].x ~ Normal()`.

These are all things which can be assigned to in Julia (sometimes called 'lvalues').
Other expressions, including literals, are not (yet?) permitted.

Note that if you have a field or element on the left-hand side, the parent object must already exist.
That is, you cannot write `m[1] ~ Normal()` or `m.x ~ Normal()` if `m` is not already defined.

### Right-hand sides

Here, 'distributions' refer to any object that implements `Base.rand(rng, dist)` and `DensityInterface.logdensityof(dist, x)`; the object doesn't have to subtype any abstract type.
This includes those defined in Distributions.jl, as well as [Wren's own distributions](@ref distributions).
Apart from distributions, it is also possible to put a Wren `Model` itself on the right-hand side of a tilde.
In such a case, the `Model` is referred to as a *submodel*.
Please see the [submodels](@ref submodels) page for more information.

## Parameters vs observations

Random quantities can be either parameters or observations.
Parameters are the quantities that we want to infer, while observations are the quantities that we have observed in the real world.

**By default, *all* `x ~ dist` statements are treated as parameters.**

```@example defining
@wren function some_parameters(scalar, vector, nt)
    scalar ~ Normal()
    for i in eachindex(vector)
        vector[i] ~ Normal()
    end
    nt.a ~ Normal()
    return nothing
end

simulate(some_parameters(0.0, zeros(3), (; a=0.0)))
```

Notice here how all of `scalar`, `vector[i]`, and `nt.a` appear as 'parameters' in the `ModelResult`.
It doesn't matter that they were passed in as arguments.

There are two ways to turn a random quantity into an observation:

 1. Marking the tilde with `@observe`:

    ```@example defining
    @wren function observations(y)
        x ~ Normal()
        @observe y ~ Normal(x)
    end
    
    simulate(observations(1.0))
    ```

 2. Conditioning the model on a value.
    First, you need to instantiate the unconditioned model; then, you can condition it using the `|` operator, passing in a `NamedTuple` of observations.

    Conditioning on `y` actually does two things: firstly, it marks the corresponding random quantity as an observation, and secondly, it sets the value of that random quantity to the value you passed in.
    Notice that `@info y` works here and uses the value of `y` that the model was conditioned on.

    ```@example defining
    @wren function observations()
        @info y
        x ~ Normal()
        y ~ Normal(x)
    end
    
    conditioned_model = observations() | (; y=1.0)
    ```

In general, conditioning is more flexible as you can 'recover' the original model by calling `decondition`:

```@example defining
original_model = decondition(conditioned_model)
simulate(original_model)
```

However, if you don't need this functionality, `@observe` can be more explicit.
`@observe` can also be used if you want to calculate the log-likelihood of some quantity that is calculated inside the model body, rather than being passed in as an argument.

## Special nodes

The above should be enough to get you started with defining models in Wren.
However, there are also some special nodes that you can use in your model body.

### [`@iid` and `@batch`](@id batched)

In the common case where you have an array of parameters that share the same distribution, you can use these macros, which have the same semantics as a naive for-loop but can have optimised performance.

`@iid` is used for _independent and identically distributed_ random quantities, i.e., each element of the array is drawn independently from the same distribution.
You can either pass in an existing array, or if you want to create a new array, you can specify how many elements there are with the `size` keyword argument.

```@example defining
@wren function with_iid(existing_array, n)
    @iid new_array ~ Normal() size=n
    @iid existing_array ~ Normal()
end
```

`@batch` is intended to be applied to for-loops, and attempts to vectorise the loop body where safe.
The following has exactly the same semantics as the version without `@batch`, but may be faster (depending on various factors).

```@example defining
@wren function with_batch(x, y)
    @batch for i in eachindex(x)
        x[i] ~ Normal()
        @observe y[i] ~ Normal(x[i])
    end
end
```

It's worth pointing out here that `@batch` is not 'magical': it works only with a restricted set of syntax and is not guaranteed to be faster than a naive for-loop.
It is often worth [benchmarking your model](@ref benchmarking) if you want to optimise performance.

### [`@track`](@id track)

Inside a model body, you may have other quantities of interest that you want to extract, even if they are not parameters.
You can mark these quantities with the `@track` macro.

By default, `simulate` will return these quantities:

```@example defining
@wren function with_track()
    x ~ Normal()
    @track x_sq = x^2
    y ~ Normal(x)
end

model = with_track() | (; y=1.0)
simulate(model)
```

You can write any expression on the right-hand side of `@track sym = expr`, including `begin` or `let` blocks.
However, the name of a tracked quantity must be a single symbol, so names like `@track x[1]` or `@track x.a` are not allowed.

!!! note "Tracked quantities are not copied"

    After initialising a tracked quantity `@track my_array`, mutating `my_array` will change its values.
    To create a snapshot of `my_array` at the time it is initialised, you need to copy explicitly: `@track snapshot = deepcopy(my_array)`.

By default, all tracked quantities are calculated and returned during model evaluation or MCMC sampling.
You can, however, skip the statements marked with `@track`—for example because they're expensive and not needed at this time—by passing `track=NoQuantities()` to `simulate`, `prior` or `nuts`.
If you want to keep specific tracked quantities, use `SomeQuantities((:x,))` instead.

!!! note "Side effects may be skipped"

    Quantities marked with `@track` are not evaluated unless they are explicitly requested.
    This means that if you write, for example, `@track x = push!(my_vec, elem)` and `x` is not requested or otherwise used, the side effect of `push!` will not be executed (i.e. `my_vec` will not be mutated).

### [`@addloglikelihood`](@id addloglike)

This manually increments the log-likelihood of the model.
For example, the following is equivalent to `@observe y ~ Normal(x)` if `y` is 1.0 (albeit rather less clear).

```@example defining
@wren function with_addloglikelihood()
    x ~ Normal()
    @addloglikelihood logdensityof(Normal(x), 1.0)
end

simulate(with_addloglikelihood())
```

!!! "warning"

    `@addloglikelihood` should only be used as an escape hatch in cases where none of the above syntax works at all.
    If you have a reason for using this, please [do let us know](https://github.com/JuliaBayes/Wren.jl/issues) as it may indicate that some other feature is missing from Wren.

## Technical restrictions

Although Wren models can contain almost completely arbitrary Julia code, due to some technical limitations of [RuntimeGeneratedFunctions.jl](@ref compilation), certain constructs are not allowed in Wren models.

The biggest one of these is that Wren models cannot close over local variables, or in other words, your model cannot use local variables defined outside the model body.

For example:

```@repl defining
for d in [Normal()]
    # d is a local variable here.

    @wren function closes_over_local()
        # ... but our model uses it
        x ~ d
    end

    simulate(closes_over_local())
end
```

The easiest way around this is to pass the variable into the model as an argument:

```@example defining
res = Any[]
for d in [Normal()]

    @wren function passes_local_in(d)
        x ~ d # OK since d is an argument.
    end

    push!(res, simulate(passes_local_in(d)))
end
only(res)
```

Closing over global variables is fine (although in general we recommend avoiding global variables; it's cleaner to pass things you need into the model as arguments).

Another rather niche restriction is that Wren models cannot contain macros that create closures.
The most likely situation where you would run into this is if you are using `Reactant.@trace`; please see the [Working with Reactant](@ref reactant) page for more information.
