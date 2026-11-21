module WrenCoreOptimizationExt

using WrenCore
import DifferentiationInterface as DI
import Optimization
using VarNames: densify!!

function WrenCore.optimize(
    model::Model,
    solver;
    adtype::DI.AbstractADType=DEFAULT_AD,
    latent_source=FromPrior(),
    lp_function=logjoint,
    reduce_copies=true,
    kwargs...,
)
    density = VecModel(model, adtype; latent_source, lp_function, reduce_copies)
    fg = function (gradient, x, p)
        gradient === nothing && return -lp(density, x)
        value, _ = lp_and_grad!(gradient, density, x)
        gradient .*= -1
        return -value
    end
    objective = Optimization.OptimizationFunction(
        (x, p) -> -lp(density, x);
        grad=(g, x, p) -> (fg(g, x, p); nothing),
        fg,
    )
    problem = Optimization.OptimizationProblem(objective, copy(density.sample))
    solution = Optimization.solve(problem, solver; kwargs...)
    params = densify!!(parameters(density, solution.u))
    return (; params, solution)
end

end
