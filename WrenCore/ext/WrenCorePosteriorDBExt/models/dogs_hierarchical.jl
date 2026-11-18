# PosteriorDB: dogs-dogs_hierarchical
module DogsHierarchical

using PosteriorDB
using WrenCore: @wren, @observe, @batch
using WrenDists: WrenDists as WD
using ..Dogs: dog_history

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(pdb, "dogs-dogs_hierarchical"),
    ))
end

@wren function dogs_hierarchical(y, shocks, avoids)
    a ~ WD.Uniform(0.0, 1.0)
    b ~ WD.Uniform(0.0, 1.0)
    p = a .^ shocks .* b .^ avoids
    @batch for i in eachindex(y)
        @observe y[i] ~ WD.Bernoulli(p[i])
    end
    return nothing
end

const wren_model = let
    y = data["y"]
    all(==(1), view(y, :, 1)) || throw(ArgumentError("expected first trial successes"))
    response, shocks, avoids = dog_history(y)
    # First trials have p = 1 and y = 1, so each has exactly zero log density.
    # Omit them to avoid the 0 * log(0) term in batched Bernoulli evaluation.
    keep = findall(>(0), shocks .+ avoids)
    dogs_hierarchical(response[keep], shocks[keep], avoids[keep])
end

end
