# PosteriorDB: wells_data-wells_dist
module WellsDist

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "wells_data-wells_dist")))
end

@wren function wells_dist_model(switched, X)
    @iid beta ~ WD.Flat() size=size(X, 2)
    logit_p = X * beta
    @batch for i in eachindex(switched)
        @observe switched[i] ~ WD.BernoulliLogit(logit_p[i])
    end
    return nothing
end

@wren function wells_glm_model(switched, X)
    alpha ~ WD.Flat()
    @iid beta ~ WD.Flat() size=size(X, 2)
    logit_p = alpha .+ X * beta
    @batch for i in eachindex(switched)
        @observe switched[i] ~ WD.BernoulliLogit(logit_p[i])
    end
    return nothing
end

const wren_model = let
    dist = data["dist"]
    wells_dist_model(data["switched"], hcat(ones(length(dist)), dist))
end

end
