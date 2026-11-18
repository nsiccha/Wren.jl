# PosteriorDB: bones_data-bones_model
module Bones

using PosteriorDB
using LogExpFunctions: log1pexp
using WrenCore: @wren, @addloglikelihood, @iid
using WrenDists: WrenDists as WD

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "bones_data-bones_model")))
end

function bones_logp(grade, gamma, delta, ncat, theta)
    total = zero(theta[1])
    for j in axes(grade, 2), i in axes(grade, 1)
        category = grade[i, j]
        category == -1 && continue
        last = ncat[j]
        if category == 1
            eta = delta[j] * (theta[i] - gamma[j, 1])
            total -= log1pexp(eta)
        elseif category == last
            eta = delta[j] * (theta[i] - gamma[j, last - 1])
            total -= log1pexp(-eta)
        else
            previous = delta[j] * (theta[i] - gamma[j, category - 1])
            current = delta[j] * (theta[i] - gamma[j, category])
            # log(logistic(previous) - logistic(current)), without subtracting
            # two probabilities that may round to the same number.
            total += -log1pexp(-previous) - log1pexp(current) +
                log(-expm1(current - previous))
        end
    end
    return total
end

@wren function bones(grade, gamma, delta, ncat, nchild)
    @iid theta ~ WD.Normal(0.0, 36.0) size=nchild
    @addloglikelihood bones_logp(grade, gamma, delta, ncat, theta)
    return nothing
end

const wren_model = bones(data["grade"], data["gamma"], data["delta"],
                         data["ncat"], data["nChild"])

end
