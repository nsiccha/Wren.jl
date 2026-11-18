# PosteriorDB: lsat_data-lsat_model
module LSAT

using PosteriorDB
using WrenCore: @wren, @observe, @batch, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "lsat_data-lsat_model")))
end

function expanded_responses(data)
    T, N = data["T"], data["N"]
    r = Matrix{Int}(undef, T, N)
    previous = 0
    for pattern in 1:data["R"]
        for student in (previous + 1):data["culm"][pattern]
            for question in 1:T
                r[question, student] = data["response"][pattern, question]
            end
        end
        previous = data["culm"][pattern]
    end
    previous == N || throw(ArgumentError("LSAT response counts do not sum to N"))
    return r
end

@wren function lsat(r, T, N)
    @iid alpha ~ WD.Normal(0.0, 100.0) size=T
    @iid theta ~ WD.Normal(0.0, 1.0) size=N
    # Stan's positive Normal factor differs by the constant log(2).
    beta ~ WD.HalfNormal(0.0, 100.0)
    logits = Vector{typeof(beta + theta[1] + alpha[1])}(undef, length(r))
    for n in eachindex(r)
        logits[n] = beta * theta[cld(n, T)] - alpha[mod1(n, T)]
    end
    @batch for n in eachindex(r)
        @observe r[n] ~ WD.BernoulliLogit(logits[n])
    end
    return nothing
end

const wren_model = lsat(expanded_responses(data), data["T"], data["N"])

end
