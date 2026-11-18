# PosteriorDB: three_docs1200-ldaK2
module LDAThreeDocs

using PosteriorDB
using WrenCore: @wren, @addloglikelihood
using WrenDists: WrenDists as WD
using ..LogAddExp: logaddexp

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "three_docs1200-ldaK2",
    )))
end

function lda_logdensity(w, doc, theta, phi)
    total = zero(theta[1, 1] + phi[1, 1])
    for n in eachindex(w)
        total += logaddexp(log(theta[1, doc[n]]) + log(phi[w[n], 1]),
            log(theta[2, doc[n]]) + log(phi[w[n], 2]))
    end
    return total
end

@wren function lda_k2(w, doc, V, M)
    theta = zeros(2, M)
    for m in 1:M
        theta[:, m] ~ WD.Dirichlet(ones(2))
    end
    phi = zeros(V, 2)
    for k in 1:2
        phi[:, k] ~ WD.Dirichlet(ones(V))
    end
    @addloglikelihood lda_logdensity(w, doc, theta, phi)
    return nothing
end

const wren_model = let
    data["V"] == 3 && data["M"] == 3 && data["N"] == 1200 ||
        error("unexpected public three-document LDA fixture")
    lda_k2(data["w"], data["doc"], data["V"], data["M"])
end

end
