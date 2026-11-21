# Wren benchmarks

## Benchmark suite

`suite/` benchmarks Wren on a set of PosteriorDB models.

By default, this is intended to be used for 'production' benchmarking. In other
words, the benchmarks used here are saved in the repo's history (specifically
the `benchmark-results` branch) so that we can keep a record of performance
over time.

To run the benchmarks locally (e.g. to try out local changes), run:

```sh
julia --project=bench bench/suite/run.jl --no-save
```

### Instructions for production benchmarking

Generally, we'll want to run prod benchmarks on:

  - Each commit to main.
  - Each release (i.e. tag) of WrenCore.

By default, runs are labelled with just the short SHA. That's fine for main
commits. However, for version releases we'll want to label the run with the
version number, so that we can easily see the performance of each release. To
do so, check out the release tag and run:

```sh
julia --project=bench -e 'import Pkg; Pkg.update()' # Don't forget this!
julia --project=bench bench/suite/run.jl --release
```

`--release` labels the run with the release's tag, `WrenCore-vX.Y.Z` (using the
version in `WrenCore/Project.toml`), and checks that the current commit has
that tag.

These will be grouped specially at the top of the dropdown.

This is kind of manual, but it's good enough, really.

To remove runs from the `benchmark-results` branch (e.g. if one was noisy), run:

```sh
julia --project=bench bench/suite/remove.jl <file or sha> [<file or sha> ...]
```

A sha removes every run of that commit. It lists the matching runs and asks
before removing them. Removed runs are still in the branch's history.

Production benchmarks aren't currently run on CI because timings are too
inconsistent. (If one day we have dedicated machines, we could do that
instead.) Instead, the production benchmarks are currently manually run on
Penny's laptop.

The suite can also be run on CI by hand, from the 'Benchmarks' workflow in the
Actions tab. The results table is shown in the job summary and the JSON file
is uploaded as an artifact; tick 'save' to also save it to the
`benchmark-results` branch.

### The website

`web/` is a small website that shows one run as a table, or compares two runs
(pick a 'tip' and a 'base'; cells are coloured by how much faster or slower tip
is, and hovering shows the base time).

It's hosted at https://juliabayes.org/Wren.jl/bench/. The 'Benchmark website'
workflow (`.github/workflows/bench-site.yml`) builds it and publishes it to the
`bench/` folder of the `gh-pages` branch. It runs:

  - whenever the `benchmark-results` branch is pushed to (that branch has a
    small workflow of its own, `.github/workflows/trigger-bench-site.yml`, which
    starts the one on `main`);
  - whenever `bench/web` changes on `main`;
  - by hand, from the Actions tab.

To view it locally:

```sh
cd bench/web
pnpm install
scripts/fetch-results.sh   # copy runs from the benchmark-results branch
pnpm dev
```

You can also load JSON files from disk with the 'Load file…' buttons.

## Ad hoc benchmarks

`adhoc/` has other one-off benchmarks (e.g. Wren vs Turing), with their own environment.
See `adhoc/README.md`.
