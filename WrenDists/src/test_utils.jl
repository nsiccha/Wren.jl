# TODO(penelopeysm): use DIT.Scenario for this when the following is fixed:
# https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/1069
struct ADTestCase{F,X,C,S}
    f::F
    x::X
    contexts::C
    skipped_adtypes::S
end

# Overloaded in extension
function test_dist end
test_dist_reactant(d, points) = nothing
test_halft_quantile_reactant(d, us) = nothing

# Test utilities for Plaice.test_all
Plaice.is_continuous(::WrenDists.UnivariateDist) = true
Plaice.is_continuous(::WrenDists.DiscreteUnivariateDist) = false
Plaice.is_continuous(::WrenDists.MultivariateDist) = true
Plaice.is_continuous(::WrenDists.DiscreteMultivariateDist) = false
Plaice.test_name(d::WrenDists.WrenDist) = repr(d)
Plaice.rand_safe_ad(d::WrenDists.WrenDist) = rand(d)
Plaice.can_test_in_support(::WrenDists.UnivariateDist, x) = true
function Plaice.test_in_support(::WrenDists.UnconstrainedUnivariateDist, x)
    @test isfinite(x)
end
function Plaice.test_in_support(::WrenDists.PositiveUnivariateDist, x)
    @test x > 0
end
function Plaice.test_in_support(::WrenDists.Beta, x)
    @test 0 <= x <= 1
end
function Plaice.test_in_support(d::WrenDists.Uniform, x)
    @test d.a <= x <= d.b
end
function Plaice.test_in_support(d::WrenDists.TruncatedFlat, x)
    @test d.lower < x < d.upper
end

# Dirichlet needs special logjac test helpers because vec_length != unconstrained_vec_length
Plaice.to_vec_for_logjac_test(::WrenDists.Dirichlet) = x -> x[1:(end-1)]
Plaice.from_vec_for_logjac_test(::WrenDists.Dirichlet) = y -> vcat(y, 1 - sum(y))

# Independent / IID
Plaice.is_continuous(d::WrenDists.IID) = Plaice.is_continuous(WrenDists._dist(d))
Plaice.is_continuous(d::WrenDists.Independent) = all(Plaice.is_continuous, d.components)
Plaice.test_name(d::WrenDists.Independent) = repr(d)
Plaice.rand_safe_ad(d::WrenDists.Independent) = rand(d)
Plaice.can_test_in_support(::WrenDists.IIDUnivariate, x) = true
Plaice.can_test_in_support(::WrenDists.IndependentArray, x) = true
function Plaice.test_in_support(d::WrenDists.IIDUnivariate, x)
    for xi in x
        Plaice.test_in_support(WrenDists._dist(d), xi)
    end
end
function Plaice.test_in_support(d::WrenDists.IndependentArray, x)
    for (di, xi) in zip(d.components, x)
        Plaice.test_in_support(di, xi)
    end
end
