# 0.1.1

`nuts` records the gradient cost of each chain. Every iteration has two new extras: `n_steps`, the leapfrog steps (and so gradient evaluations) of that transition, and `n_gradient_evaluations`, the chain's running total of gradient evaluations including the initial point, the initial step-size search and warm-up. Sampling itself is unchanged.

# 0.1.0

Initial release.
