# PosteriorDB: mnist_100-nn_rbm1bJ10
module MNIST100RBM

using PosteriorDB
using WrenCore: @wren, @addloglikelihood, @iid
using WrenDists: WrenDists as WD
using ..LogAddExp: logaddexp

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
        pdb, "mnist_100-nn_rbm1bJ10",
    )))
end

function mnist_logdensity(y, x, alpha, beta, alpha1, beta1)
    J = length(alpha1)
    K = length(beta1) + 1
    total = zero(alpha[1, 1] + beta[1, 1] + alpha1[1] + beta1[1])
    for n in eachindex(y)
        hidden = [tanh(alpha1[j] + sum(x[n, m] * alpha[m, j]
            for m in axes(x, 2))) for j in 1:J]
        logits = [beta1[k] + sum(hidden[j] * beta[j, k] for j in 1:J)
            for k in 1:(K - 1)]
        lognormalizer = one(total)
        for score in logits
            lognormalizer = logaddexp(lognormalizer, score)
        end
        chosen = y[n] == 1 ? one(total) : logits[y[n] - 1]
        total += chosen - lognormalizer
    end
    return total
end

@wren function rbm(y, x, N, M, K, J, s2_alpha, s2_beta)
    sigma2_alpha ~ WD.InverseGamma(0.25, 0.25 * s2_alpha)
    sigma2_beta ~ WD.InverseGamma(0.25, 0.25 * s2_beta)
    @iid alpha ~ WD.Normal(0.0, sqrt(sigma2_alpha)) size=(M, J)
    @iid beta ~ WD.Normal(0.0, sqrt(sigma2_beta)) size=(J, K - 1)
    @iid alpha1 ~ WD.Normal(0.0, 1.0) size=J
    @iid beta1 ~ WD.Normal(0.0, 1.0) size=(K - 1)
    @addloglikelihood mnist_logdensity(y, x, alpha, beta, alpha1, beta1)
    return nothing
end

const wren_model = let
    data["N"] == 100 && data["M"] == 784 && data["K"] == 10 ||
        error("unexpected public MNIST 100 fixture")
    M, J = data["M"], 10
    s2_alpha = (0.05 / M^2)^2
    s2_beta = (0.05 / J^2)^2
    rbm(data["y"], Float64.(data["x"]), data["N"], M, data["K"], J,
        s2_alpha, s2_beta)
end

end
