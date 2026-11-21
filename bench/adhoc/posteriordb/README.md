# posteriordb

```
cd bench/adhoc
julia --project=. posteriordb/run.jl
```

This folder runs a series of 10 PosteriorDB benchmarks, comparing Wren and Turing.

The Wren models come from WrenCore's PosteriorDB extension (`posteriordb_model`), the
same as in the main benchmark suite. The Turing models in `models/` are written to match
them. Both Wren and Turing are timed with the fastest sample, and BLAS is limited to one
thread.

## The models

| Posterior                              | dim | Family                               |
|:-------------------------------------- | ---:|:------------------------------------ |
| `Rate_1_data-Rate_1_model`             | 1   | Binomial proportion                  |
| `GLM_Binomial_data-GLM_Binomial_model` | 3   | Binomial regression                  |
| `gp_pois_regr-gp_regr`                 | 3   | Gaussian process regression          |
| `GLM_Poisson_Data-GLM_Poisson_model`   | 4   | Poisson regression                   |
| `kidiq-kidscore_interaction`           | 5   | Linear regression with interaction   |
| `sblri-blr`                            | 6   | Bayesian linear regression           |
| `arK-arK`                              | 7   | Autoregressive series                |
| `eight_schools-eight_schools_centered` | 10  | Centred hierarchical normal          |
| `GLMM_Poisson_data-GLMM_Poisson_model` | 45  | Poisson mixed model                  |
| `Mh_data-Mh_model`                     | 388 | Capture-recapture with heterogeneity |
