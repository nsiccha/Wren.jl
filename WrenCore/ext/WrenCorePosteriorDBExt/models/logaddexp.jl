# Shared by the marginalised PosteriorDB models: HMM forward passes, mixtures, and the
# occupancy and capture-recapture likelihoods.
module LogAddExp

"""
    logaddexp(a, b)

`log(exp(a) + exp(b))`, with the derivative `(1/2, 1/2)` at a tie `a == b`.

`LogExpFunctions.logaddexp` replaces `a - b` by a constant zero whenever `a == b`, so
AD of its source (Enzyme, for example) returns `(1, 0)` at a tie. Symmetric points, such
as an all-zero unconstrained vector, are exact ties in these models. Here the difference
is replaced only where it is `NaN`, at `a == b == ±Inf`, as `LogExpFunctions.logsumexp`
does.
"""
@inline function logaddexp(a, b)
    x, y = promote(a, b)
    hi, lo = x > y ? (x, y) : (y, x)
    d = lo - hi
    return hi + log1p(exp(ifelse(isnan(d) & (hi == lo), zero(d), d)))
end

end
