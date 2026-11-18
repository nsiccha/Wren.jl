module IRTCommon

using Statistics: mean, std
using WrenCore: logdensityof
using WrenDists: WrenDists as WD
using ..LogAddExp: logaddexp

# Centre every covariate after the intercept and divide it by twice its standard
# deviation. The Stan sources mean to divide a two-valued covariate by its range
# instead, but count the matching rows as
#     minmax_count = minmax_count + W[j, k] == min_w || W[j, k] == max_w;
# which Stan parses as `(minmax_count + W[j, k] == min_w) || (W[j, k] == max_w)`.
# That is 0 or 1, never `rows(W)`, so the range branch is not taken for more than
# one person and the PosteriorDB targets use `2 * sd` for every covariate.
function adjusted_covariates(W)
    J, K = size(W)
    W_adj = Matrix{Float64}(undef, J, K)
    W_adj[:, 1] .= W[:, 1]
    for k in 2:K
        column = view(W, :, k)
        W_adj[:, k] .= (column .- mean(column)) ./ (2 * std(column))
    end
    return W_adj
end

function beta_theta_logdensity(W_adj, beta_free, theta, lambda_adj)
    total = zero(beta_free[1] + theta[1] + lambda_adj[1])
    beta_last = -sum(beta_free)
    for b in beta_free
        total += logdensityof(WD.Normal(0.0, 3.0), b)
    end
    total += logdensityof(WD.Normal(0.0, 3.0), beta_last)
    for j in eachindex(theta)
        location = zero(total)
        for k in eachindex(lambda_adj)
            location += W_adj[j, k] * lambda_adj[k]
        end
        total += logdensityof(WD.Normal(location, 1.0), theta[j])
    end
    return total
end

function graded_logdensity(y, eta, thresholds, offset, first, m)
    score = zero(eta + offset + thresholds[first])
    selected = score
    lognormalizer = score
    for r in 1:m
        score += eta - offset - thresholds[first + r - 1]
        selected = y == r ? score : selected
        lognormalizer = logaddexp(lognormalizer, score)
    end
    return selected - lognormalizer
end

end
