# [Bayesian workflow](@id workflow)

In the worked example below, we demonstrate how Wren and the JuliaBayes ecosystem enable the [Bayesian Workflow](https://avehtari.github.io/Bayesian-Workflow/).

We'll concoct an example with the [Palmer penguins dataset](https://github.com/devmotion/PalmerPenguins.jl).
This dataset includes measurements of three different Antarctic penguins species.

We first have to prepare the dataset.
We'll z-standardise the variables because it usually improves sampling efficiency and simplifies choosing priors.

```@example workflow
using Wren, PalmerPenguins, DataFrames, FlexiChains
using WrenDists: Normal, Exponential
using StatsBase: denserank, fit, ZScoreTransform, reconstruct

# Load data
penguins = DataFrame(PalmerPenguins.load())

# Drop missing values first (this ensures that standardisation doesn't trip over)
dropmissing!(penguins)

# Fit standardisation transforms so we can unstandardise predictions later
bill_zs = fit(ZScoreTransform, Float64.(penguins.bill_length_mm))
mass_zs = fit(ZScoreTransform, Float64.(penguins.body_mass_g))

# Tidy up the data
standardize(x) = (x .- mean(x)) ./ std(x)
transform!(penguins, names(penguins, Real) .=> standardize => identity)
transform!(penguins, :species => denserank => :species_idx)
nothing # hide
```

Let's say we're interested in learning if a penguin's bill length increases with body mass, while taking into account that bill lengths between species might differ, too.
The model below uses [`@iid`](@ref iid) for the per-species intercepts and a [`@batch`](@ref batched) loop for the likelihood:

```@example workflow
# Define a model for penguin bill length as a function of species and body mass
@wren function bill_model(bill_length, species, body_mass)
    n_species = maximum(species)
    @iid beta1 ~ Normal(0, 1) size=n_species
    beta2 ~ Normal(0, 1)
    sigma ~ Exponential(1)
    mu = beta1[species] .+ beta2 .* body_mass
    @batch for i in eachindex(bill_length)
        bill_length[i] ~ Normal(mu[i], sigma)
    end
end
```

## Prior predictions

We can now instantiate the Wren model and use it with `simulate(model)` to draw from the prior (see [Working with `Models`](@ref using-models)):

```@example workflow
model = bill_model(penguins.bill_length_mm, penguins.species_idx, penguins.body_mass_g)
simulate(model)
```

!!! note

    You may have noticed that we did not use `@observe` in our model definition.
    This is because `@observe param ~ dist` will always treat `param` as an observed variable, even if it was not conditioned on.
    The consequence would be that we could draw neither prior nor posterior predictions for it.

Alternatively, we can use [`prior`](@ref) to create a [`FlexiChain`](@extref FlexiChains.FlexiChain) that contains as many prior draws as we specify, and summarise it with [`summarystats`](@extref FlexiChains :jl:method:`StatsBase.summarystats-Tuple{FlexiChains.FlexiChain}`).

```@example workflow
chain_prior = prior(model, 1000, 4; progress=false)
summarystats(chain_prior[[@vn(beta1), @vn(beta2), @vn(sigma)]])
```

While we could technically continue working with the chain, the `simulate(model)` syntax lends itself nicely towards visualising individual samples from the prior distribution.

```@example workflow
using CairoMakie

colors = Makie.wong_colors()
fig = Figure()
ax = Axis(fig[1, 1], xlabel="body mass (g)", ylabel="bill length (mm)")

xs_z = -3:0.1:3

# Draw 20 prior predictions for the slope of body mass
for _ in 1:20
    vnt = parameters(simulate(model))
    ys_z = vnt[@vn(beta2)] * xs_z
    ys = reconstruct(bill_zs, ys_z)
    xs = reconstruct(mass_zs, xs_z)
    lines!(xs, ys, color=(colors[1], 0.6))
end

fig
```

You can read more about working with [`VarName`](@extref VarNames.VarName) and [`VarNamedTuple`](@extref VarNames.VarNamedTuple) in the [VarNames.jl docs](https://github.com/JuliaBayes/VarNames.jl).

We may also want to check if the prior distribution of `bill_length` looks reasonable, i.e., that it doesn't include impossible values such as negative `bill_length`.
FlexiChains provides helpful functions here – first, we reconstruct the draws on the millimeter scale with [`transform_values`](@extref FlexiChains.transform_values) and then visualise the prior predictive distribution with [`pushforward_hist`](@extref FlexiChains.Makie.pushforward_hist):

```@example workflow
using FlexiChains: transform_values
chain_prior =
    transform_values(chain_prior, @vn(bill_length) => (v -> reconstruct(bill_zs, v)))
observed = reconstruct(bill_zs, penguins.bill_length_mm)
FlexiChains.Makie.pushforward_hist(
    chain_prior,
    @vn(bill_length);
    observed=observed,
    axis=(; xlabel="bill_length_mm", limits=((0, 100), nothing)),
)
```

## Conditioning on data

Having had a look at the prior, we're now ready to condition the model on the observed data and sample from the posterior.
Wren's default autodiff backend is Enzyme (see [Working with Enzyme](@ref enzyme)), so we need to make sure to import that first.
Then we condition with `|` (see [`condition`](@ref)) and sample with [`nuts`](@ref):

```@example workflow
import Enzyme
model_cond = model | (bill_length=penguins.bill_length_mm,)
chain_post = nuts(model_cond, 1000, 4; progress=false)
summarystats(chain_post)
```

Normally we would like to check that chains mixed and sampling went well with more diagnostics, such as diagnostic plots `plot(chain_post)` or checking for divergent transitions `sum(chain_post[:is_divergent]) / prod(size(chain_post))`, but we'll leave it at mentioning these for this simple model.

## Posterior predictions

Towards the end of the Bayesian workflow, we typically want to check if the model approximates the data-generating process reasonably well with a posterior predictive check.

A posterior predictive check uses the fitted parameters and values of the covariates `species` and `body_mass` to predict new values of the outcome `bill_length`.
This can be done with either the same covariate values used to fit the model, or with a new set of covariate values.
A new set could either represent a hypothetical new group of penguins that we want to predict for, or help us visualise how `bill_length` changes with `body_mass`, resulting in a plot similar to the prior slope visualisation above.

We'll begin with the former and once again use [`simulate`](@ref)—but this time we'll simulate from the conditioned model instead of the prior!
`FlexiChains.pushforward_hist` then plots histograms for our posterior prediction and overlays the observed data, giving us an opportunity to spot if the two would clearly diverge:

```@example workflow
chain_pred = simulate(model, chain_post)
chain_pred =
    transform_values(chain_pred, @vn(bill_length) => (v -> reconstruct(bill_zs, v)))
FlexiChains.Makie.pushforward_hist(
    chain_pred,
    @vn(bill_length);
    observed=observed,
    axis=(; xlabel="bill length (mm)"),
)
```

The model seems to capture the distribution of the observed data well!

Now let's see what the model learned about how `body_mass` relates to `bill_length`, by drawing single samples from the chain and extracting them with [`parameters`](@extref VarNames.parameters).

```@example workflow
fig = Figure()
ax = Axis(fig[1, 1], xlabel="body mass (g)", ylabel="bill length (mm)")

xs_z = -3:0.1:3

# Draw 20 posterior samples of the slope of body mass
for _ in 1:20
    vnt = rand(chain_post; parameters_only=true)
    ys_z = vnt[@vn(beta2)] * xs_z
    ys = reconstruct(bill_zs, ys_z)
    xs = reconstruct(mass_zs, xs_z)
    lines!(xs, ys, color=(colors[1], 0.6))
end

fig
```

As we may have expected, more penguin means more bill!

## Tracked quantities

Quite regularly in applied statistical modelling, we're after a quantity that we can't express as a single parameter of our model.
Rather, it's often some downstream quantity that we compute by combining several parameters with (potentially new / hypothetical) observations.
You can tell Wren that you want to track these downstream quantities with [`@track`](@ref track).

Continuing with the penguin model from before, let's now assume that there is something special about the conditions on each island.
These conditions make _all_ the penguins on a given island more similar to each other; that could for example be caused by better food sources near some islands than others, or some islands being more prone to disruptive weather events than others.

We can express these correlated observations in Bayesian modelling with a hierarchical model.
Since this model component is quite common and thus reusable, it's a good candidate to be written as a [submodel](@ref submodels):

```@example workflow
@wren function hierarchical(size)
    mu ~ Normal()
    tau ~ Exponential(1)
    @iid z ~ Normal() size=size
    return mu .+ tau * z
end
```

!!! info "Intro to hierarchical models"

    In this hierarchical model, we define a global average bill length across all islands $\mu$ and the heterogeneity between islands $\tau$.
    To illustrate this, let's assume a scenario where $\tau$ is rather small; this would correspond to a world where conditions between islands aren't that different after all.
    It might be helpful to also consider the most extreme setting where $\tau = 0$: here, all islands would have identical conditions, and the hierarchical structure would disappear entirely.
    This scenario is called "complete pooling".
    On the contrary, if we assume $\tau$ to be really, really large, this would correspond to a scenario where islands are _so different_ that we might as well estimate each of their means individually.
    This is called "no pooling".

    In short, learning $\tau$ from the data ("partial pooling") infers _just how much_ information we should share between islands.
    This comes with many nice benefits like avoiding overfitting!

The tracked quantity we're after in this example is the proportion of Adelie penguins with a dangerously long bill (there's no domain knowledge in this, we just assume that bills longer than 42 mm are scary).

```@example workflow
import StatsBase
const SCARY_BILL_ZSCALE = only(StatsBase.transform(bill_zs, [42.0]))

@wren function bill_model_hierarchical(bill_length, island, body_mass)
    n_islands = maximum(island)

    beta1 = (isl ~ hierarchical(n_islands))
    beta2 ~ Normal(0, 1)
    sigma ~ Exponential(1)

    mu = beta1[island] .+ beta2 .* body_mass
    @batch for i in eachindex(bill_length)
        bill_length[i] ~ Normal(mu[i], sigma)
    end

    @track scary_bill = let
        pred_bill = rand.(Normal.(mu, sigma))
        mean(pred_bill .> SCARY_BILL_ZSCALE)
    end
end
```

You might rightfully wonder why we're only considering Adelie penguins this time.
The reason for this is that the Adelies are the only species occurring on all three islands; for both Gentoos and Chinstraps, we only have data from one island, respectively.
In short, we're simply not set up to estimate how island conditions influence the bill lengths of these two species!

Having recovered from that disappointment, let's now condition our expanded model on the observations again and sample.

```@example workflow
# Adelies are the only species found on all three islands
adelies = subset(penguins, :species => ByRow(==("Adelie")))
transform!(adelies, :island => denserank => :island_idx)

model =
    bill_model_hierarchical(
        adelies.bill_length_mm,
        adelies.island_idx,
        adelies.body_mass_g,
    ) | (bill_length=adelies.bill_length_mm,)
chain_tq = nuts(model, 1000, 4; progress=false)
summarystats(chain_tq)
```

So far so good, but the result for `scary_bill` in `chain_tq` only tells us what we'd expect for the _specific_ islands we collected data from.
Sometimes, however, that's not what we're after in a hierarchical model.

What if we want to quantify the effect in general, in the hypothetical "population" of islands that gave rise to our observed sample?
To transfer our inferences to this population, we need to sample from the inferred population of islands `Normal(isl.mu, isl.tau)`.

But how would we do this in our model?
After all, the number of islands is part of our model—our `z` parameter has as many entries as we have islands!

Before figuring out how to escape the structure of our initial model, we first define a new data set with 300 hypothetical new islands (the number 300 is arbitrary here, it should just be large enough to represent the hypothetical "population" of islands).

```@example workflow
n_new = 300
island_new = 1:n_new
# Resample the observed body mass
body_mass_g_new = rand(adelies.body_mass_g, n_new)
nothing # hide
```

Now you may wonder how we can tell our model to ignore the individual `isl.z` parameters in the model.
The key here is that `simulate` will use the parameters from the `chain` _if_ they are stored there, and otherwise fall back to sampling from the prior.

That's exactly what we want here!

By sampling `isl.z` from the prior, we get draws from `Normal(isl.mu, isl.tau)`.
So all we need to do is drop `isl.z` from the chain:

```@example workflow
chain_sub = chain_tq[filter(!=(@vn(isl.z)), FlexiChains.parameters(chain_tq))]
model_new = bill_model_hierarchical(zeros(n_new), island_new, body_mass_g_new)
chain_new = simulate(model_new, chain_sub)
mean(chain_new[@vn(scary_bill)]), std(chain_new[@vn(scary_bill)])
```

As you can see, our estimate for the hypothetical population is quite similar to that of the specific islands that we estimated our first model with.
The reason for this is simply because there isn't a lot of heterogeneity between these islands (see the estimate for `isl.tau` in `chain_tq`), and even if there was, our estimate would only be weakly informed by the data given we only recorded data for three islands.
