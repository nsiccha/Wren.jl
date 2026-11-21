# WrenDists

WrenDists provides a small collection of Reactant- and vectorisation-friendly probability distributions.

The distributions defined in WrenDists are intended to be used only with Wren, and thus have a very limited API, essentially consisting of:

  - `Base.rand([rng, ]dist)`
  - `DensityInterface.logdensityof(dist, x)`

See https://juliabayes.org/Wren.jl/dev/tutorials/distributions/#wrendists for more information.
