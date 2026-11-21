module WrenCoreEnzymeExt

using WrenCore
using WrenCore: _docs_link, Accessors
using DifferentiationInterface: DifferentiationInterface as DI
using Enzyme: Enzyme, Reverse, Forward, set_runtime_activity

@inline Enzyme.EnzymeRules.inactive(::typeof(WrenCore._get_range_and_transform), args...) =
    nothing
@inline Enzyme.EnzymeRules.inactive(::typeof(WrenCore.target_transform), args...) = nothing

"""
    _handle_enzyme(vm::VecModel{M,<:AutoEnzyme})

Handle Enzyme-specific quirks so that end users don't have to deal with it.

Specifically, this:

- Runs the gradient once to force compilation of the rules, so that subsequent
  calls are fast. Note that `DI.prepare_gradient` does not actually force rule
  compilation.

- If Enzyme requires runtime activity, and `WrenCore.AUTO_RUNTIME_ACTIVITY[]` is true, then it
  will enable runtime activity (with a warning).
"""
function WrenCore._handle_enzyme(vm::WrenCore.VecModel{M,<:DI.AutoEnzyme}) where {M<:Model}
    err = try
        lp_and_grad(vm, vm.sample)
        # Successful
        return vm
    catch e
        e
    end
    if err isa Enzyme.Compiler.EnzymeRuntimeActivityError &&
       WrenCore.AUTO_RUNTIME_ACTIVITY[] !== false
        if WrenCore.AUTO_RUNTIME_ACTIVITY[] === nothing
            @info "enabling Enzyme runtime activity for model $(vm.model.name)"
        end
        return WrenCore.VecModel(
            vm.model,
            enable_runtime_activity(vm.adtype),
            vm.lp_function,
            vm.parameter_space,
            vm.vecinfo,
            vm.sample,
        )
    end
    # Throw outside the `catch` block so that the error isn't shown with Enzyme's
    # internal stacktrace, or printed a second time as "caused by".
    # In debug mode, throw the original error so that it's shown in full.
    if err isa Enzyme.Compiler.EnzymeError && !WrenCore.WREN_COMPILATION_DEBUG[]
        throw(WrenEnzymeError(err))
    else
        throw(err)
    end
end

function enable_runtime_activity(::DI.AutoEnzyme{Nothing,FA}) where {FA}
    mode = set_runtime_activity(Reverse)  # promote to reverse mode if unknown.
    return DI.AutoEnzyme{typeof(mode),FA}(mode)
end
function enable_runtime_activity(adt::DI.AutoEnzyme{Mode,FA}) where {Mode,FA}
    mode = set_runtime_activity(adt.mode)
    return DI.AutoEnzyme{typeof(mode),FA}(mode)
end

"""
    WrenEnzymeError(err)

Wraps an error thrown by Enzyme when differentiating a model. It is shown without a
stacktrace, since the stacktrace only points into Wren internals; the useful information
(including where in the model the error happened) is in Enzyme's own error message. The
original error is stored in the `err` field.

If `WrenCore.WREN_COMPILATION_DEBUG[]` is `true`, Enzyme errors are not wrapped, and are
shown in full.
"""
struct WrenEnzymeError{E<:Exception} <: Exception
    err::E
end
Base.showerror(io::IO, e::WrenEnzymeError) = _show_enzyme_error(io, e.err)

const _ISSUES_LINK = "https://github.com/JuliaBayes/Wren.jl/issues"

function _print_heading(io::IO, heading::String)
    println(io)
    printstyled(io, "━━━ ", heading, " ", "━"^(76 - length(heading)), "\n"; bold=true)
    return nothing
end

# Improved Enzyme error messages.
function _show_enzyme_error(io::IO, ::Enzyme.Compiler.IllegalTypeAnalysisException)
    printstyled(
        io,
        "IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.\n";
        bold=true,
    )

    _print_heading(io, "Why?")
    print(
        io,
        """
        This error usually means that some value has a `Union` type (for example,
        `Union{Nothing,Float64}`, or equivalently, that some part of your model is
        type-unstable.
        """,
    )

    _print_heading(io, "What can you do?")
    print(
        io,
        """
        1. Ideally, remove the type instability. This will also make your model faster.

        2. Try setting

               Enzyme.API.strictAliasing!(false)

        3. If you are stuck, please open an issue at

                $(_ISSUES_LINK)

           with the definition of your model to help reproduce the error.

        4. You can obtain a full stack trace by setting

               WrenCore.WREN_COMPILATION_DEBUG[] = true

           Note however that this stack trace is often very long and quite unhelpful
           (indeed, that is why we hide it by default).
        """,
    )
    return nothing
end

function _show_enzyme_error(io::IO, ::Enzyme.Compiler.EnzymeRuntimeActivityError)
    printstyled(
        io,
        "EnzymeRuntimeActivityError: potential need for runtime activity detected\n";
        bold=true,
    )

    _print_heading(io, "Why?")
    print(
        io,
        """
        Enzyme requires runtime activity in order to correctly differentiate your
        model. See $(WrenCore._docs_link("explanations/enzyme/#Runtime-activity"))
        for more information.
        """,
    )

    _print_heading(io, "What can you do?")
    print(
        io,
        """
        1. Allow Wren to automatically enable runtime activity for you by setting

               WrenCore.AUTO_RUNTIME_ACTIVITY[] = true

           Although enabling runtime activity does make Enzyme slower, it will often still
           be faster than other AD backends.

        2. Use an explicit adtype that has runtime activity enabled, such as
            
               using Enzyme, ADTypes
               AutoEnzyme(; mode=set_runtime_activity(Reverse))

        3. Rewrite the model to not need runtime activity. Note that this is not always
           possible; see $(WrenCore._docs_link("explanations/enzyme/#Runtime-activity"))
           for more information on this. Generally, it would be better to just enable
           runtime activity instead of rewriting the model into a form that is harder to
           work with.

        4. If you don't think your model should require runtime activity, please open
           an issue at

                $(_ISSUES_LINK)

           with the definition of your model to help reproduce the error.
        """,
    )
    return nothing
end

# Fallback method for other Enzyme erorrs that we don't yet have a custom message for.
function _show_enzyme_error(io::IO, err::Enzyme.Compiler.EnzymeError)
    # Enzyme's "Failure within method" section shows the method that failed, but for Wren
    # models this is always the model's RuntimeGeneratedFunction, whose very long type
    # signature tells the user nothing. Setting `mi` to `nothing` makes Enzyme skip it.
    if hasfield(typeof(err), :mi)
        err = try
            Accessors.@set err.mi = nothing
        catch
            # some error types don't allow `mi` to be `nothing`
            err
        end
    end
    showerror(io, err)
    return nothing
end
Base.showerror(io::IO, e::WrenEnzymeError, ::Any; backtrace=true) = showerror(io, e)

end # module
