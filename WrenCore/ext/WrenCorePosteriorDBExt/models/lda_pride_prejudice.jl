# PosteriorDB: prideprejudice_{chapter,paragraph}-ldaK5
module LDAPridePrejudice

using PosteriorDB
using WrenCore: @wren, @addloglikelihood
using WrenDists: WrenDists as WD
using ..LogAddExp: logaddexp

function lda_logdensity(w, doc, theta, phi)
    total = zero(theta[1, 1] + phi[1][1])
    for n in eachindex(w)
        m, word = doc[n], w[n]
        word_logp = log(theta[1, m]) + log(phi[1][word])
        for k in 2:5
            word_logp = logaddexp(word_logp,
                log(theta[k, m]) + log(phi[k][word]))
        end
        total += word_logp
    end
    return total
end

@wren function lda_k5(w, doc, M, alpha, beta)
    theta = zeros(5, M)
    for m in 1:M
        theta[:, m] ~ WD.Dirichlet(alpha)
    end
    # Named sites allow callers to provide an explicit unconstrained start for
    # the 1495-word simplices; a sparse prior draw can fail Plaice's inverse
    # simplex transform after floating-point rounding.
    phi1 ~ WD.Dirichlet(beta)
    phi2 ~ WD.Dirichlet(beta)
    phi3 ~ WD.Dirichlet(beta)
    phi4 ~ WD.Dirichlet(beta)
    phi5 ~ WD.Dirichlet(beta)
    @addloglikelihood lda_logdensity(w, doc, theta,
        (phi1, phi2, phi3, phi4, phi5))
    return nothing
end

function load_case(name, M)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    data["V"] == 1495 && data["M"] == M && data["N"] == 32877 &&
        length(data["alpha"]) == 5 && length(data["beta"]) == 1495 ||
        error("unexpected public Pride and Prejudice LDA fixture: $name")
    return lda_k5(data["w"], data["doc"], M, data["alpha"], data["beta"])
end

const chapter = (wren_model=load_case("prideprejudice_chapter-ldaK5", 61),)
const paragraph = (wren_model=load_case("prideprejudice_paragraph-ldaK5", 2025),)

end
