module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
using Distributions: Bernoulli, Beta, Binomial, Cauchy, Dirichlet, Exponential, InverseGamma, LogNormal, Logistic, Normal, TDist, logpdf
using Enzyme
import ForwardDiff
using PosteriorDB: PosteriorDB
using StableRNGs
using Statistics: mean, std
using Test
using WrenCore
using WrenDists: WrenDists

@testset "posteriordb.jl" begin
    function pride_lda_source(M)
        # A Dirichlet(0.1) prior draw may round a simplex prefix above one in
        # Plaice's inverse transform (Wren.jl #202). Start in vector space.
        word_init = TransformedValue(zeros(1494), AsUnconstrainedVec())
        params = VarNamedTuple(phi1=word_init, phi2=word_init, phi3=word_init,
            phi4=word_init, phi5=word_init)
        topic_init = TransformedValue(zeros(4), AsUnconstrainedVec())
        topic_template = zeros(5, M)
        for m in 1:M
            params = WrenCore.VarNames.templated_setindex!!(params, topic_init,
                @vn(theta[:, m]), topic_template)
        end
        return FromValues(params)
    end

    # Helpers shared by the density and gradient testsets below. Each oracle is
    # written from the Stan program, independently of the Wren model.
    logistic(x) = inv(1 + exp(-x))
    logit_jacobian(x) = -log1p(exp(-x)) - log1p(exp(x))
    logaddexp(a, b) = a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))
    bernoulli_logit(y, eta) = y * eta - log1p(exp(eta))
    posteriordb_data(name) = PosteriorDB.load(PosteriorDB.dataset(
        PosteriorDB.posterior(PosteriorDB.database(), name)))
    enzyme_model(name; kwargs...) =
        VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse); kwargs...)

    # The density and the full native Enzyme gradient against an oracle.
    function check(name, dimension, oracle, points; atol=1e-5)
        vm = enzyme_model(name)
        @test vm.dimension == dimension
        for point in points
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, oracle(point); atol)
            @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol)
        end
    end

    # For models too large for a full ForwardDiff gradient: the density, and the
    # native Enzyme gradient in the `selected` coordinates.
    function check_selected(vm, oracle, point, selected;
            atol=1e-5, rtol=0.0, gradient_rtol=rtol)
        @test vm.dimension == length(point)
        value, gradient = lp_and_grad(vm, point)
        @test isapprox(value, oracle(point); atol, rtol)
        function selected_oracle(z)
            full = Vector{eltype(z)}(point)
            full[selected] .= z
            return oracle(full)
        end
        @test isapprox(gradient[selected],
            ForwardDiff.gradient(selected_oracle, point[selected]);
            atol, rtol=gradient_rtol)
    end


    @testset "$name" for name in available_posteriordb_models()
        model = posteriordb_model(name)
        # LSAT has 1000 person effects; ForwardDiff's full Jacobian is too large
        # for the generic smoke test. Its targeted checks use native Enzyme below.
        adtype = name in ("lsat_data-lsat_model", "election88-election88_full",
            "sat-hier_2pl", "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
            "science_irt-grsm_latent_reg_irt",
            "timssAusTwn_irt-gpcm_latent_reg_irt",
            "three_men1-ldaK2", "three_men2-ldaK2", "three_men3-ldaK2",
            "prideprejudice_chapter-ldaK5", "prideprejudice_paragraph-ldaK5",
            "traffic_accident_nyc-bym2_offset_only",
            "mnist_100-nn_rbm1bJ10") ?
            AutoEnzyme(; mode=Reverse) :
            DI.AutoForwardDiff()
        pride_m = name == "prideprejudice_chapter-ldaK5" ? 61 :
            name == "prideprejudice_paragraph-ldaK5" ? 2025 : nothing
        vm = pride_m === nothing ? VecModel(model, adtype) :
            VecModel(model, adtype; latent_source=pride_lda_source(pride_m),
                reduce_copies=false)
        # dogs_log uses flat coordinates with bounded likelihood factors; a
        # prior draw need not lie in their finite-density support.
        if name == "dogs-dogs_log"
            v = [-0.2, 0.3]
            @test isfinite(lp(vm, v))
        elseif pride_m !== nothing
            v = 0.01 .* sin.(1:vm.dimension)
        else
            @test isfinite(logjoint(model, parameters(model, StableRNG(468))))
            v = vector_parameters(vm, StableRNG(468))
        end
        @test length(v) == vm.dimension
        logp, grad = lp_and_grad(vm, v)
        @test isfinite(logp)
        @test length(grad) == length(v)
        @test all(isfinite, grad)
    end

    @test_throws ArgumentError posteriordb_model("not-a-posterior")

    # A Gaussian regression `y ~ Normal(X * beta, sigma)` with flat priors: besides
    # the likelihood, the density has only the Jacobian of `sigma = exp(v[end])`.
    function check_flat_gaussian_regression(name, X, y, points; atol)
        dimension = size(X, 2) + 1
        vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
        @test vm.dimension == dimension
        function reference(v)
            mu = X * view(v, 1:(dimension - 1))
            sigma = exp(v[end])
            return v[end] + sum(logpdf(Normal(mu[i], sigma), y[i])
                for i in eachindex(y))
        end
        for point in points
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol)
        end
    end

    @testset "Kilpisjarvi Gaussian regression density and native Enzyme gradient" begin
        name = "kilpisjarvi_mod-kilpisjarvi"
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), name,
        )))
        vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
        @test vm.dimension == 3
        function reference(v)
            alpha, beta, log_sigma = v
            sigma = exp(log_sigma)
            return logpdf(Normal(data["pmualpha"], data["psalpha"]), alpha) +
                logpdf(Normal(data["pmubeta"], data["psbeta"]), beta) +
                log_sigma + sum(logpdf(Normal(alpha + beta * data["x"][i], sigma),
                    data["y"][i]) for i in eachindex(data["y"]))
        end
        for point in ([0.0, 0.0, log(1.0)], [1.5, -0.3, log(2.0)])
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol=1e-5)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-5)
        end
    end

    @testset "Rats growth density and native Enzyme gradient" begin
        name = "rats_data-rats_model"
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), name,
        )))
        N = data["N"]
        vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
        @test vm.dimension == 5 + 2N
        function reference(v)
            mu_alpha, mu_beta = v[1], v[2]
            log_sigma_y, log_sigma_alpha, log_sigma_beta = view(v, 3:5)
            sigma_y = exp(log_sigma_y)
            sigma_alpha = exp(log_sigma_alpha)
            sigma_beta = exp(log_sigma_beta)
            alpha = view(v, 6:(5 + N))
            beta = view(v, (6 + N):(5 + 2N))
            return logpdf(Normal(0, 100), mu_alpha) +
                logpdf(Normal(0, 100), mu_beta) +
                log_sigma_y + log_sigma_alpha + log_sigma_beta +
                sum(logpdf(Normal(mu_alpha, sigma_alpha), a) for a in alpha) +
                sum(logpdf(Normal(mu_beta, sigma_beta), b) for b in beta) +
                sum(logpdf(Normal(alpha[data["rat"][i]] +
                    beta[data["rat"][i]] * (data["x"][i] - data["xbar"]),
                    sigma_y), data["y"][i]) for i in eachindex(data["y"]))
        end
        for offset in (0.0, 0.2)
            point = vcat(250.0 + offset, 6.0 - offset, log(25.0), log(35.0),
                log(2.0), 250.0 .+ sin.(1:N), 6.0 .+ cos.(1:N))
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol=1e-5)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-5)
        end
    end

    @testset "Pilots bounded-scale density and native Enzyme gradient" begin
        name = "pilots-pilots"
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), name,
        )))
        G, S = data["n_groups"], data["n_scenarios"]
        vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
        @test vm.dimension == 5 + G + S
        function reference(v)
            mu_a, mu_b = v[1], v[2]
            raw = view(v, 3:5)
            sigma = 100 ./ (1 .+ exp.(-raw))
            a = view(v, 6:(5 + G))
            b = view(v, (6 + G):(5 + G + S))
            jacobian = sum(-log1p(exp(-u)) - log1p(exp(u))
                for u in raw)
            return logpdf(Normal(), mu_a) + logpdf(Normal(), mu_b) +
                jacobian +
                sum(logpdf(Normal(10mu_a, sigma[1]), value) for value in a) +
                sum(logpdf(Normal(10mu_b, sigma[2]), value) for value in b) +
                sum(logpdf(Normal(a[data["group_id"][i]] +
                    b[data["scenario_id"][i]], sigma[3]), data["y"][i])
                    for i in eachindex(data["y"]))
        end
        for offset in (0.0, 0.1)
            point = vcat(0.04 + offset, 0.02 - offset,
                log(0.5 / 99.5), log(0.8 / 99.2), log(0.4 / 99.6),
                0.4 .+ 0.1 .* sin.(1:G), 0.2 .+ 0.1 .* cos.(1:S))
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol=1e-5)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-5)
        end
    end

    @testset "flat-prior Kidscore densities and native Enzyme gradients" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "kidiq_with_mom_work-kidscore_interaction_c",
        )))
        hs = data["mom_hs"]
        iq = data["mom_iq"]
        work = data["mom_work"]
        centered_hs = hs .- mean(hs)
        centered_iq = iq .- mean(iq)
        scaled_hs = centered_hs ./ (2 * std(hs))
        scaled_iq = centered_iq ./ (2 * std(iq))
        cases = (
            ("kidiq_with_mom_work-kidscore_interaction_c",
                hcat(ones(length(hs)), centered_hs, centered_iq,
                    centered_hs .* centered_iq)),
            ("kidiq_with_mom_work-kidscore_interaction_c2",
                hcat(ones(length(hs)), hs .- 0.5, iq .- 100.0,
                    (hs .- 0.5) .* (iq .- 100.0))),
            ("kidiq_with_mom_work-kidscore_interaction_z",
                hcat(ones(length(hs)), scaled_hs, scaled_iq,
                    scaled_hs .* scaled_iq)),
            ("kidiq_with_mom_work-kidscore_mom_work",
                hcat(ones(length(work)), work .== 2, work .== 3, work .== 4)),
        )
        for (name, X) in cases
            check_flat_gaussian_regression(name, X, data["kid_score"],
                ([100.0, 2.0, 0.2, 0.1, log(15.0)],
                 [95.0, -1.0, 0.3, -0.1, log(20.0)]); atol=1e-5)
        end
    end

    @testset "eight NES regression densities and native Enzyme gradients" begin
        for year in 1972:4:2000
            name = "nes$(year)-nes"
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            age = data["age_discrete"]
            X = hcat(ones(data["N"]), data["real_ideo"], data["race_adj"],
                age .== 2, age .== 3, age .== 4, data["educ1"], data["gender"],
                data["income"])
            check_flat_gaussian_regression(name, X, data["partyid7"],
                ([3.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.2, 0.1, log(1.5)],
                 [4.0, -0.1, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.2, log(2.0)]);
                atol=1e-5)
        end
    end

    @testset "earnings regression densities and native Enzyme gradients" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "earnings-earn_height",
        )))
        earn, height, male = data["earn"], data["height"], data["male"]
        N = data["N"]
        zheight = (height .- mean(height)) ./ std(height)
        cases = (
            ("earnings-earn_height", earn, hcat(ones(N), height)),
            ("earnings-log10earn_height", log10.(earn), hcat(ones(N), height)),
            ("earnings-logearn_height", log.(earn), hcat(ones(N), height)),
            ("earnings-logearn_height_male", log.(earn), hcat(ones(N), height, male)),
            ("earnings-logearn_interaction", log.(earn),
                hcat(ones(N), height, male, height .* male)),
            ("earnings-logearn_interaction_z", log.(earn),
                hcat(ones(N), zheight, male, zheight .* male)),
            ("earnings-logearn_logheight_male", log.(earn),
                hcat(ones(N), log.(height), male)),
        )
        for (name, y, X) in cases
            points = [vcat(scale .* sin.(1:size(X, 2)), log(20.0))
                for scale in (0.1, -0.2)]
            check_flat_gaussian_regression(name, X, y, points; atol=1e-4)
        end
    end

    @testset "Mesquite regression densities and native Enzyme gradients" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "mesquite-mesquite",
        )))
        d1, d2 = data["diam1"], data["diam2"]
        canopy, height = data["canopy_height"], data["total_height"]
        density, group = data["density"], data["group"]
        weight, N = data["weight"], data["N"]
        logvolume = log.(d1 .* d2 .* canopy)
        logarea = log.(d1 .* d2)
        logshape = log.(d1 ./ d2)
        cases = (
            ("mesquite-mesquite", weight,
                hcat(ones(N), d1, d2, canopy, height, density, group)),
            ("mesquite-logmesquite", log.(weight),
                hcat(ones(N), log.(d1), log.(d2), log.(canopy),
                    log.(height), log.(density), group)),
            ("mesquite-logmesquite_logva", log.(weight),
                hcat(ones(N), logvolume, logarea, group)),
            ("mesquite-logmesquite_logvas", log.(weight),
                hcat(ones(N), logvolume, logarea, logshape,
                    log.(height), log.(density), group)),
            ("mesquite-logmesquite_logvash", log.(weight),
                hcat(ones(N), logvolume, logarea, logshape, log.(height), group)),
            ("mesquite-logmesquite_logvolume", log.(weight),
                hcat(ones(N), logvolume)),
        )
        for (name, y, X) in cases
            points = [vcat(scale .* sin.(1:size(X, 2)), log(20.0))
                for scale in (0.1, -0.2)]
            check_flat_gaussian_regression(name, X, y, points; atol=1e-6)
        end
    end

    @testset "Rate posteriors: Stan density and unconstrained gradient" begin
        pdb = PosteriorDB.database()
        for (number, point) in (
            (2, [0.3, -0.4]), (2, [-1.2, 1.3]), (2, [2.0, -2.0]),
            (3, [0.3]), (3, [-1.2]), (3, [2.0]),
            (4, [0.3, -0.4]), (4, [-1.2, 1.3]), (4, [2.0, -2.0]),
            (5, [-0.4]), (5, [1.3]), (5, [-2.0]),
        )
            name = "Rate_$(number)_data-Rate_$(number)_model"
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == length(point)

            p = inv(1 + exp(-point[1]))
            jac = log(p) + log1p(-p)
            if number == 4
                q = inv(1 + exp(-point[2]))
                expected = logpdf(Binomial(data["n"], p), data["k"]) + jac +
                    log(q) + log1p(-q)
                expected_grad = [data["k"] + 1 - (data["n"] + 2) * p, 1 - 2q]
            elseif number == 2
                q = inv(1 + exp(-point[2]))
                expected = logpdf(Binomial(data["n1"], p), data["k1"]) + jac +
                    logpdf(Binomial(data["n2"], q), data["k2"]) + log(q) + log1p(-q)
                expected_grad = [
                    data["k1"] + 1 - (data["n1"] + 2) * p,
                    data["k2"] + 1 - (data["n2"] + 2) * q,
                ]
            else
                expected = logpdf(Binomial(data["n1"], p), data["k1"]) +
                    logpdf(Binomial(data["n2"], p), data["k2"]) + jac
                expected_grad = [
                    data["k1"] + data["k2"] + 1 -
                    (data["n1"] + data["n2"] + 2) * p,
                ]
            end
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, expected; atol=1e-9)
            @test isapprox(gradient, expected_grad; atol=1e-9)
        end
    end

    @testset "noncentered schools: Stan density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "eight_schools-eight_schools_noncentered",
        )))
        J, y, sigma = data["J"], data["y"], data["sigma"]
        vm = VecModel(posteriordb_model("eight_schools-eight_schools_noncentered"),
            AutoEnzyme(; mode=Reverse))
        @test vm.dimension == J + 2

        function reference(v)
            theta_trans, mu, logtau = view(v, 1:J), v[J+1], v[J+2]
            tau = exp(logtau)
            theta = theta_trans .* tau .+ mu
            return sum(logpdf(Normal(), x) for x in theta_trans) +
                logpdf(Normal(0, 5), mu) + logpdf(Cauchy(0, 5), tau) + log(2) +
                logtau + sum(logpdf(Normal(theta[i], sigma[i]), y[i]) for i in eachindex(y))
        end
        for point in (
            vcat(collect(range(-0.8, 0.8; length=J)), 0.4, -0.3),
            vcat(sin.(1:J), -1.1, 0.5),
        )
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol=1e-9)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-8)
        end
    end

    @testset "pooled radon: Stan density and gradient" begin
        pdb = PosteriorDB.database()
        for name in ("radon_all-radon_pooled", "radon_mn-radon_pooled")
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
            y, floor_measure = data["log_radon"], data["floor_measure"]
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == 3

            function reference(v)
                alpha, beta, logsigma = v
                sigma_y = exp(logsigma)
                return logpdf(Normal(0, 10), alpha) + logpdf(Normal(0, 10), beta) +
                    logpdf(Normal(), sigma_y) + log(2) + logsigma +
                    sum(logpdf(Normal(alpha + beta * floor_measure[i], sigma_y), y[i])
                        for i in eachindex(y))
            end
            for point in ([0.2, -0.5, log(0.8)], [1.1, 0.4, log(1.4)])
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-6)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-6)
            end
        end
    end

    @testset "wells logistic posteriors: Stan density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "wells_data-wells_dist",
        )))
        y, dist, arsenic = data["switched"], data["dist"], data["arsenic"]
        educ, assoc = data["educ"], data["assoc"]
        dist100 = dist ./ 100.0
        c_dist100 = (dist .- sum(dist) / length(dist)) ./ 100.0
        c_arsenic = arsenic .- sum(arsenic) / length(arsenic)
        educ4 = educ ./ 4.0
        c_educ4 = (educ .- sum(educ) / length(educ)) ./ 4.0
        cases = (
            ("wells_data-wells_dist", hcat(ones(length(dist)), dist), false),
            ("wells_data-wells_dist100_model", hcat(dist100), true),
            ("wells_data-wells_dist100ars_model", hcat(dist100, arsenic), true),
            ("wells_data-wells_dae_model", hcat(dist100, arsenic, educ4), true),
            ("wells_data-wells_dae_c_model",
                hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic, educ4), true),
            ("wells_data-wells_dae_inter_model",
                hcat(c_dist100, c_arsenic, c_educ4, c_dist100 .* c_arsenic,
                    c_dist100 .* c_educ4, c_arsenic .* c_educ4), true),
            ("wells_data-wells_daae_c_model",
                hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic, assoc, educ4), true),
            ("wells_data-wells_interaction_model",
                hcat(dist100, arsenic, dist100 .* arsenic), true),
            ("wells_data-wells_interaction_c_model",
                hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic), true),
        )
        for (name, X, has_alpha) in cases
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            dimension = size(X, 2) + has_alpha
            @test vm.dimension == dimension
            for scale in (0.2, -0.3)
                point = scale .* sin.(1:dimension)
                logit_p = has_alpha ? point[1] .+ X * view(point, 2:dimension) : X * point
                p = @. inv(1 + exp(-logit_p))
                expected = sum(y[i] * logit_p[i] - log1p(exp(logit_p[i])) for i in eachindex(y))
                residual = y .- p
                expected_grad = has_alpha ? vcat(sum(residual), X' * residual) : X' * residual
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, expected; atol=1e-6)
                @test isapprox(gradient, expected_grad; atol=1e-6)
            end
        end
    end

    @testset "kidscore linear posteriors: density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "kidiq-kidscore_momhs",
        )))
        y, mom_hs, mom_iq = data["kid_score"], data["mom_hs"], data["mom_iq"]
        intercept = ones(length(y))
        cases = (
            ("kidiq-kidscore_momhs", hcat(intercept, mom_hs)),
            ("kidiq-kidscore_momiq", hcat(intercept, mom_iq)),
            ("kidiq-kidscore_momhsiq", hcat(intercept, mom_hs, mom_iq)),
        )
        for (name, X) in cases
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            k = size(X, 2)
            @test vm.dimension == k + 1
            for log_sigma in (log(50.0), log(85.0))
                beta = 0.2 .* cos.(1:k)
                point = vcat(beta, log_sigma)
                sigma = exp(log_sigma)
                residual = y - X * beta
                expected = -length(y) * log(sigma * sqrt(2π)) -
                    sum(abs2, residual) / (2sigma^2) +
                    log(2 / (π * 2.5)) - log1p((sigma / 2.5)^2) + log_sigma
                expected_grad = vcat(
                    X' * residual / sigma^2,
                    -length(y) + sum(abs2, residual) / sigma^2 -
                        2sigma^2 / (2.5^2 + sigma^2) + 1,
                )
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, expected; atol=1e-6)
                @test isapprox(gradient, expected_grad; atol=1e-6)
            end
        end
    end

    @testset "radon densities and native Enzyme gradients" begin
        # Sums of Normal log densities, and a half-normal(0, 1) scale on its log
        # coordinate with the Jacobian. The oracles below are built from these.
        normal_sum(x, mu, sigma) =
            -length(x) * log(sigma * sqrt(2π)) - sum(abs2, x .- mu) / (2sigma^2)
        half_normal_scale(log_sigma) =
            log(2 / π) / 2 - exp(2log_sigma) / 2 + log_sigma

        function hierarchical_intercept(data, v)
            J = data["J"]
            alpha_raw, beta, mu_alpha = view(v, 1:J), view(v, (J + 1):(J + 2)), v[J + 3]
            sigma_alpha, sigma_y = exp(v[J + 4]), exp(v[J + 5])
            alpha = mu_alpha .+ sigma_alpha .* alpha_raw
            mu = alpha[data["county_idx"]] .+ beta[1] .* data["log_uppm"] .+
                beta[2] .* data["floor_measure"]
            return normal_sum(alpha_raw, 0, 1) + normal_sum(beta, 0, 10) +
                normal_sum(mu_alpha, 0, 10) + half_normal_scale(v[J + 4]) +
                half_normal_scale(v[J + 5]) + normal_sum(data["log_radon"], mu, sigma_y)
        end
        function partially_pooled(data, v)
            J = data["J"]
            alpha_raw, mu_alpha = view(v, 1:J), v[J + 1]
            sigma_alpha, sigma_y = exp(v[J + 2]), exp(v[J + 3])
            alpha = mu_alpha .+ sigma_alpha .* alpha_raw
            return normal_sum(alpha_raw, 0, 1) + normal_sum(mu_alpha, 0, 10) +
                half_normal_scale(v[J + 2]) + half_normal_scale(v[J + 3]) +
                normal_sum(data["log_radon"], alpha[data["county_idx"]], sigma_y)
        end
        function variable_intercept(data, v)
            J = data["J"]
            alpha_raw, beta, mu_alpha = view(v, 1:J), v[J + 1], v[J + 2]
            sigma_alpha, sigma_y = exp(v[J + 3]), exp(v[J + 4])
            alpha = mu_alpha .+ sigma_alpha .* alpha_raw
            mu = alpha[data["county_idx"]] .+ data["floor_measure"] .* beta
            return normal_sum(alpha_raw, 0, 1) + normal_sum(beta, 0, 10) +
                normal_sum(mu_alpha, 0, 10) + half_normal_scale(v[J + 3]) +
                half_normal_scale(v[J + 4]) + normal_sum(data["log_radon"], mu, sigma_y)
        end
        function variable_slope(data, v)
            J = data["J"]
            alpha, beta_raw, mu_beta = v[1], view(v, 2:(J + 1)), v[J + 2]
            sigma_beta, sigma_y = exp(v[J + 3]), exp(v[J + 4])
            beta = mu_beta .+ sigma_beta .* beta_raw
            mu = alpha .+ data["floor_measure"] .* beta[data["county_idx"]]
            return normal_sum(beta_raw, 0, 1) + normal_sum(alpha, 0, 10) +
                normal_sum(mu_beta, 0, 10) + half_normal_scale(v[J + 3]) +
                half_normal_scale(v[J + 4]) + normal_sum(data["log_radon"], mu, sigma_y)
        end
        function variable_intercept_slope(data, v)
            J = data["J"]
            sigma_y, sigma_alpha, sigma_beta = exp(v[1]), exp(v[2]), exp(v[3])
            alpha_raw, beta_raw = view(v, 4:(J + 3)), view(v, (J + 4):(2J + 3))
            mu_alpha, mu_beta = v[2J + 4], v[2J + 5]
            alpha = mu_alpha .+ sigma_alpha .* alpha_raw
            beta = mu_beta .+ sigma_beta .* beta_raw
            mu = alpha[data["county_idx"]] .+
                data["floor_measure"] .* beta[data["county_idx"]]
            return normal_sum(alpha_raw, 0, 1) + normal_sum(beta_raw, 0, 1) +
                normal_sum(mu_alpha, 0, 10) + normal_sum(mu_beta, 0, 10) +
                sum(half_normal_scale, view(v, 1:3)) +
                normal_sum(data["log_radon"], mu, sigma_y)
        end
        function county_intercept(data, v)
            J = data["J"]
            alpha, beta, sigma_y = view(v, 1:J), v[J + 1], exp(v[J + 2])
            mu = alpha[data["county_idx"]] .+ beta .* data["floor_measure"]
            return normal_sum(alpha, 0, 10) + normal_sum(beta, 0, 10) +
                half_normal_scale(v[J + 2]) + normal_sum(data["log_radon"], mu, sigma_y)
        end

        # One oracle and one point per model family.
        families = (
            ("radon_hierarchical_intercept_noncentered", hierarchical_intercept,
                J -> vcat(0.05 .* sin.(1:J), [0.08, -0.11], 0.2, log(0.7), log(0.9))),
            ("radon_partially_pooled_noncentered", partially_pooled,
                J -> vcat(0.05 .* sin.(1:J), 0.2, log(0.7), log(0.9))),
            ("radon_variable_intercept_noncentered", variable_intercept,
                J -> vcat(0.05 .* sin.(1:J), 0.08, 0.2, log(0.7), log(0.9))),
            ("radon_variable_slope_noncentered", variable_slope,
                J -> vcat(0.08, 0.05 .* sin.(1:J), 0.2, log(0.7), log(0.9))),
            ("radon_variable_intercept_slope_noncentered", variable_intercept_slope,
                J -> vcat(log(0.9), log(0.7), log(0.8), 0.04 .* sin.(1:J),
                    0.04 .* cos.(1:J), 0.2, -0.1)),
            ("radon_county_intercept", county_intercept,
                J -> vcat(0.04 .* sin.(1:J), 0.2, log(0.9))),
        )
        for (variant, family, point_for) in families,
                dataset in ("radon_mn", "radon_all")

            name = "$dataset-$variant"
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            oracle(v) = family(data, v)
            point = point_for(data["J"])
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == length(point)
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, oracle(point); atol=1e-5)
            # Both datasets bind the same model definition, so the full gradient is
            # checked on the smaller Minnesota dataset only.
            if dataset == "radon_mn"
                @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol=1e-5)
            end
        end
    end

    @testset "radon county bounded-scale density and native Enzyme gradient" begin
        name = "radon_mod-radon_county"
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), name,
        )))
        J = data["J"]
        vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
        @test vm.dimension == J + 3
        normal_logp(x, mu, sigma) =
            -log(sigma * sqrt(2π)) - (x - mu)^2 / (2 * sigma^2)
        function reference(v)
            mu_a = v[1]
            ratio_a = inv(1 + exp(-v[2]))
            ratio_y = inv(1 + exp(-v[3]))
            sigma_a, sigma_y = 100 * ratio_a, 100 * ratio_y
            a = view(v, 4:length(v))
            jacobian = log(ratio_a) + log1p(-ratio_a) +
                log(ratio_y) + log1p(-ratio_y)
            return normal_logp(mu_a, 0.0, 1.0) + jacobian +
                sum(normal_logp(value, mu_a, sigma_a) for value in a) +
                sum(normal_logp(data["y"][i], a[data["county"][i]], sigma_y)
                    for i in eachindex(data["y"]))
        end
        for offset in (0.0, 0.1)
            point = vcat(0.2 + offset, log(1.0 / 99.0), log(0.8 / 99.2),
                0.1 .* sin.(1:J))
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol=1e-5)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-5)
        end
    end

    @testset "centered radon: density and native Enzyme gradients" begin
        normal_sum(x, mu, sigma) =
            -length(x) * log(sigma * sqrt(2π)) - sum(abs2, x .- mu) / (2sigma^2)
        normal_one(x, mu, sigma) =
            -log(sigma * sqrt(2π)) - (x - mu)^2 / (2sigma^2)
        half_one(sigma) = 0.5log(2 / π) - sigma^2 / 2
        function check(name, dimension, oracle, points)
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == dimension
            for point in points
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, oracle(point); atol=1e-5)
                @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol=1e-5)
            end
        end

        for name in (
            "radon_all-radon_hierarchical_intercept_centered",
            "radon_mn-radon_hierarchical_intercept_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            log_uppm = data["log_uppm"]
            floor_measure = data["floor_measure"]
            y = data["log_radon"]
            function oracle(v)
                alpha = view(v, 1:J)
                beta = view(v, J + 1:J + 2)
                mu_alpha = v[J + 3]
                sigma_alpha = exp(v[J + 4])
                sigma_y = exp(v[J + 5])
                mu = alpha[county_idx] .+ beta[1] .* log_uppm .+
                    beta[2] .* floor_measure
                return normal_sum(alpha, mu_alpha, sigma_alpha) +
                    normal_sum(beta, 0.0, 10.0) + normal_one(mu_alpha, 0.0, 10.0) +
                    half_one(sigma_alpha) + half_one(sigma_y) +
                    normal_sum(y, mu, sigma_y) + v[J + 4] + v[J + 5]
            end
            points = [
                vcat(0.04 .* sin.(1:J), [0.08, -0.11], 0.2, log(0.7), log(0.9)),
                vcat(0.03 .* cos.(1:J), [-0.05, 0.13], -0.1, log(1.1), log(1.2)),
            ]
            check(name, J + 5, oracle, points)
        end

        for name in (
            "radon_all-radon_partially_pooled_centered",
            "radon_mn-radon_partially_pooled_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            y = data["log_radon"]
            function oracle(v)
                alpha = view(v, 1:J)
                mu_alpha = v[J + 1]
                sigma_alpha = exp(v[J + 2])
                sigma_y = exp(v[J + 3])
                return normal_sum(alpha, mu_alpha, sigma_alpha) +
                    normal_one(mu_alpha, 0.0, 10.0) +
                    half_one(sigma_alpha) + half_one(sigma_y) +
                    normal_sum(y, alpha[county_idx], sigma_y) + v[J + 2] + v[J + 3]
            end
            points = [
                vcat(0.04 .* sin.(1:J), 0.2, log(0.7), log(0.9)),
                vcat(0.03 .* cos.(1:J), -0.1, log(1.1), log(1.2)),
            ]
            check(name, J + 3, oracle, points)
        end

        for name in (
            "radon_all-radon_variable_intercept_centered",
            "radon_mn-radon_variable_intercept_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            floor_measure = data["floor_measure"]
            y = data["log_radon"]
            function oracle(v)
                alpha = view(v, 1:J)
                beta = v[J + 1]
                mu_alpha = v[J + 2]
                sigma_alpha = exp(v[J + 3])
                sigma_y = exp(v[J + 4])
                mu = alpha[county_idx] .+ beta .* floor_measure
                return normal_sum(alpha, mu_alpha, sigma_alpha) +
                    normal_one(beta, 0.0, 10.0) + normal_one(mu_alpha, 0.0, 10.0) +
                    half_one(sigma_alpha) + half_one(sigma_y) +
                    normal_sum(y, mu, sigma_y) + v[J + 3] + v[J + 4]
            end
            points = [
                vcat(0.04 .* sin.(1:J), 0.1, 0.2, log(0.7), log(0.9)),
                vcat(0.03 .* cos.(1:J), -0.1, -0.2, log(1.1), log(1.2)),
            ]
            check(name, J + 4, oracle, points)
        end

        for name in (
            "radon_all-radon_variable_slope_centered",
            "radon_mn-radon_variable_slope_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            floor_measure = data["floor_measure"]
            y = data["log_radon"]
            function oracle(v)
                alpha = v[1]
                beta = view(v, 2:J + 1)
                mu_beta = v[J + 2]
                sigma_beta = exp(v[J + 3])
                sigma_y = exp(v[J + 4])
                mu = alpha .+ beta[county_idx] .* floor_measure
                return normal_one(alpha, 0.0, 10.0) +
                    normal_sum(beta, mu_beta, sigma_beta) +
                    normal_one(mu_beta, 0.0, 10.0) +
                    half_one(sigma_beta) + half_one(sigma_y) +
                    normal_sum(y, mu, sigma_y) + v[J + 3] + v[J + 4]
            end
            points = [
                vcat(0.2, 0.04 .* sin.(1:J), 0.1, log(0.7), log(0.9)),
                vcat(-0.1, 0.03 .* cos.(1:J), -0.2, log(1.1), log(1.2)),
            ]
            check(name, J + 4, oracle, points)
        end

        for name in (
            "radon_all-radon_variable_intercept_slope_centered",
            "radon_mn-radon_variable_intercept_slope_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            floor_measure = data["floor_measure"]
            y = data["log_radon"]
            function oracle(v)
                sigma_y = exp(v[1])
                sigma_alpha = exp(v[2])
                sigma_beta = exp(v[3])
                alpha = view(v, 4:J + 3)
                beta = view(v, J + 4:2J + 3)
                mu_alpha = v[2J + 4]
                mu_beta = v[2J + 5]
                mu = alpha[county_idx] .+ beta[county_idx] .* floor_measure
                return normal_sum(alpha, mu_alpha, sigma_alpha) +
                    normal_sum(beta, mu_beta, sigma_beta) +
                    normal_one(mu_alpha, 0.0, 10.0) +
                    normal_one(mu_beta, 0.0, 10.0) +
                    half_one(sigma_y) + half_one(sigma_alpha) + half_one(sigma_beta) +
                    normal_sum(y, mu, sigma_y) + v[1] + v[2] + v[3]
            end
            points = [
                vcat(log(0.9), log(0.7), log(0.8), 0.04 .* sin.(1:J),
                    0.04 .* cos.(1:J), 0.2, -0.1),
                vcat(log(1.2), log(1.1), log(0.9), 0.03 .* cos.(1:J),
                    0.03 .* sin.(1:J), -0.2, 0.1),
            ]
            check(name, 2J + 5, oracle, points)
        end
    end

    @testset "logaddexp shared by the marginalised models" begin
        logaddexp_model =
            Base.get_extension(WrenCore, :WrenCorePosteriorDBExt).LogAddExp.logaddexp
        @test logaddexp_model(0.3, -1.2) ≈ log(exp(0.3) + exp(-1.2))
        @test logaddexp_model(-Inf, 0.5) == 0.5
        @test logaddexp_model(-Inf, -Inf) == -Inf
        @test logaddexp_model(Inf, Inf) == Inf
        @test isnan(logaddexp_model(NaN, 1.0))
        @test isnan(logaddexp_model(1.0, NaN))
        # Both arguments get half of the derivative at a tie.
        pair(v) = logaddexp_model(v[1], v[2])
        @test only(Enzyme.gradient(Enzyme.Reverse, pair, [0.7, 0.7])) ≈ [0.5, 0.5]
        @test ForwardDiff.gradient(pair, [0.7, 0.7]) ≈ [0.5, 0.5]
    end

    @testset "discrete PosteriorDB densities and native Enzyme gradients" begin
        election_data = posteriordb_data("election88-election88_full")
        na, nb, nc, nd, ne = (election_data[k] for k in
            ("n_age", "n_edu", "n_age_edu", "n_state", "n_region_full"))
        function election_oracle(v)
            beta = view(v, 1:5)
            raw = view(v, 6:10)
            scales = 100 ./ (1 .+ exp.(-raw))
            a = view(v, 11:(10 + na))
            b = view(v, (11 + na):(10 + na + nb))
            c = view(v, (11 + na + nb):(10 + na + nb + nc))
            d = view(v, (11 + na + nb + nc):(10 + na + nb + nc + nd))
            e = view(v, (11 + na + nb + nc + nd):length(v))
            eta = beta[1] .+ beta[2] .* election_data["black"] .+
                beta[3] .* election_data["female"] .+
                beta[5] .* (election_data["female"] .* election_data["black"]) .+
                beta[4] .* election_data["v_prev_full"] .+
                a[election_data["age"]] .+ b[election_data["edu"]] .+
                c[election_data["age_edu"]] .+ d[election_data["state"]] .+
                e[election_data["region_full"]]
            prior = sum(logpdf(Normal(0, 100), bi) for bi in beta) +
                sum(logit_jacobian(ui) for ui in raw) +
                sum(logpdf(Normal(0, scales[1]), ai) for ai in a) +
                sum(logpdf(Normal(0, scales[2]), bi) for bi in b) +
                sum(logpdf(Normal(0, scales[3]), ci) for ci in c) +
                sum(logpdf(Normal(0, scales[4]), di) for di in d) +
                sum(logpdf(Normal(0, scales[5]), ei) for ei in e)
            return prior + sum(bernoulli_logit.(election_data["y"], eta))
        end
        election_point = vcat([0.2, 0.1, -0.1, 0.2, 0.1],
            fill(log(1.0 / 99.0), 5), 0.05 .* sin.(1:(na + nb + nc + nd + ne)))
        check("election88-election88_full", 10 + na + nb + nc + nd + ne,
            election_oracle, [election_point])

        hmm_data = posteriordb_data("hmm_example-hmm_example")
        function hmm_oracle(v)
            p11, p21 = logistic(v[1]), logistic(v[2])
            mu1 = exp(v[3])
            mu2 = mu1 + exp(v[4])
            y = hmm_data["y"]
            g1 = logpdf(Normal(mu1, 1), y[1])
            g2 = logpdf(Normal(mu2, 1), y[1])
            for t in 2:length(y)
                next1 = logaddexp(g1 + log(p11), g2 + log(p21)) +
                    logpdf(Normal(mu1, 1), y[t])
                next2 = logaddexp(g1 + log1p(-p11), g2 + log1p(-p21)) +
                    logpdf(Normal(mu2, 1), y[t])
                g1, g2 = next1, next2
            end
            return logit_jacobian(v[1]) + logit_jacobian(v[2]) +
                v[3] + v[4] + logpdf(Normal(3, 1), mu1) +
                logpdf(Normal(10, 1), mu2) + logaddexp(g1, g2)
        end
        check("hmm_example-hmm_example", 4, hmm_oracle,
            [[0.8, -1.2, log(3.0), log(7.0)],
             [-0.3, 0.4, log(2.5), log(6.5)]])

        hmm_gaussian_data = posteriordb_data("hmm_gaussian_simulated-hmm_gaussian")
        function simplex3_oracle(a, b)
            z1 = logistic(a - log(2))
            z2 = logistic(b)
            probabilities = [z1, (1 - z1) * z2, (1 - z1) * (1 - z2)]
            jacobian = logit_jacobian(a - log(2)) + log1p(-z1) +
                logit_jacobian(b)
            return probabilities, jacobian
        end
        function hmm_gaussian_oracle(v)
            pi, jacobian = simplex3_oracle(v[1], v[2])
            rows = [simplex3_oracle(v[2i + 1], v[2i + 2]) for i in 1:3]
            A = [rows[i][1][j] for i in 1:3, j in 1:3]
            jacobian += sum(row[2] for row in rows) + 4 * log(2.0)
            mu = [v[9], v[9] + exp(v[10]), v[9] + exp(v[10]) + exp(v[11])]
            sigma = exp.(v[12:14])
            jacobian += v[10] + v[11] + sum(v[12:14])
            y = hmm_gaussian_data["y"]
            # Stan's `log(pi1) + normal_lpdf(y[1] | mu, sigma)` adds the summed
            # emission density of all three states to every state.
            first_emission = sum(logpdf(Normal(mu[j], sigma[j]), y[1]) for j in 1:3)
            alpha = [log(pi[j]) + first_emission for j in 1:3]
            for t in 2:length(y)
                alpha = [logpdf(Normal(mu[j], sigma[j]), y[t]) +
                    logaddexp(logaddexp(alpha[1] + log(A[1, j]),
                        alpha[2] + log(A[2, j])), alpha[3] + log(A[3, j]))
                    for j in 1:3]
            end
            return jacobian + logaddexp(logaddexp(alpha[1], alpha[2]), alpha[3])
        end
        check("hmm_gaussian_simulated-hmm_gaussian", 14, hmm_gaussian_oracle,
            [[0.2, -0.4, 0.5, -0.3, -0.2, 0.4, 0.1, -0.1,
              10.0, log(10.0), log(10.0), log(5.0), log(5.0), log(5.0)],
             [-0.2, 0.3, -0.4, 0.2, 0.3, -0.5, -0.1, 0.4,
              11.0, log(10.0), log(10.0), log(6.0), log(5.0), log(4.0)]])

        bball_data = [posteriordb_data("bball_drive_event_$(i)-hmm_drive_$(i)")
            for i in 0:1]
        function bball_oracle(v, fixture, gaussian)
            p11, p21 = logistic(v[1]), logistic(v[2])
            phi1 = gaussian ? v[3] : exp(v[3])
            phi2 = phi1 + exp(v[4])
            lambda1 = gaussian ? v[5] : exp(v[5])
            lambda2 = lambda1 + exp(v[6])
            prior = logpdf(Beta(4, 2), p11) + logpdf(Beta(2, 4), p21) +
                logit_jacobian(v[1]) + logit_jacobian(v[2]) +
                v[4] + v[6] + (gaussian ? zero(v[3]) : v[3] + v[5]) +
                logpdf(Normal(0, 1), phi1) + logpdf(Normal(3, 1), phi2) +
                logpdf(Normal(0, 1), lambda1) + logpdf(Normal(3, 1), lambda2)
            function emission(t, k)
                phi, lambda = k == 1 ? (phi1, lambda1) : (phi2, lambda2)
                if gaussian
                    return logpdf(Normal(phi, fixture["tau"]), fixture["u"][t]) +
                        logpdf(Normal(lambda, fixture["rho"]), fixture["v"][t])
                end
                return logpdf(Exponential(inv(phi)), fixture["u"][t]) +
                    logpdf(Exponential(inv(lambda)), fixture["v"][t])
            end
            g1, g2 = emission(1, 1), emission(1, 2)
            for t in 2:fixture["N"]
                next1 = emission(t, 1) + logaddexp(g1 + log(p11), g2 + log(p21))
                next2 = emission(t, 2) +
                    logaddexp(g1 + log1p(-p11), g2 + log1p(-p21))
                g1, g2 = next1, next2
            end
            return prior + logaddexp(g1, g2)
        end
        check("bball_drive_event_0-hmm_drive_0", 6,
            v -> bball_oracle(v, bball_data[1], false),
            [[0.7, -0.5, log(6.0), log(2.0), log(0.02), log(0.03)],
             [-0.1, 0.3, log(5.0), log(2.5), log(0.025), log(0.02)]])
        check("bball_drive_event_1-hmm_drive_1", 6,
            v -> bball_oracle(v, bball_data[2], true),
            [[0.7, -0.5, -1.8, log(1.5), 3.9, log(1.0)],
             [-0.1, 0.3, -1.7, log(1.4), 3.8, log(1.2)]])

        iohmm_data = posteriordb_data("iohmm_reg_simulated-iohmm_reg")
        function iohmm_oracle(v)
            pi, jacobian = simplex3_oracle(v[1], v[2])
            w = reshape(v[3:14], 4, 3)
            b = reshape(v[15:26], 4, 3)
            sigma = exp.(v[27:29])
            jacobian += sum(v[27:29])
            prior = log(2.0) +
                sum(logpdf(Normal(0, 5), vi) for vi in v[3:26]) +
                sum(logpdf(Normal(0, 3), si) + log(2.0) for si in sigma)
            u, y = iohmm_data["u"], iohmm_data["y"]
            regress(t, coefficients, j) =
                sum(u[t, m] * coefficients[m, j] for m in 1:4)
            g = [log(pi[j]) + logpdf(Normal(regress(1, b, j), sigma[j]), y[1])
                for j in 1:3]
            for t in 2:length(y)
                logits = [regress(t, w, j) for j in 1:3]
                norm = logaddexp(logaddexp(logits[1], logits[2]), logits[3])
                # Match the Stan source's destination-independent logA[t][i].
                common = logaddexp(logaddexp(g[1] + logits[1] - norm,
                    g[2] + logits[2] - norm), g[3] + logits[3] - norm)
                g = [common + logpdf(Normal(regress(t, b, j), sigma[j]), y[t])
                    for j in 1:3]
            end
            return prior + jacobian + logaddexp(logaddexp(g[1], g[2]), g[3])
        end
        # At zero all three states tie in every forward step. A `logaddexp` that
        # drops the derivative at a tie gives 0.889 instead of 8189.55 for the
        # second residual scale there.
        check("iohmm_reg_simulated-iohmm_reg", 29, iohmm_oracle,
            [zeros(29),
             vcat([-0.3, 0.5], 0.15 .* cos.(1:24),
                 [log(2.5), log(3.5), log(4.0)])])

        butterfly_data = posteriordb_data("butterfly-multi_occupancy")
        function butterfly_oracle(v)
            alpha, beta = v[1], v[2]
            omega = logistic(v[3])
            rho_unit = logistic(v[4])
            rho = 2 * rho_unit - 1
            sigma1, sigma2 = exp(v[5]), exp(v[6])
            S, n, J, K = (butterfly_data[key] for key in ("S", "n", "J", "K"))
            uv1 = view(v, 7:(6 + S))
            uv2 = view(v, (7 + S):length(v))
            prior = logpdf(Cauchy(0, 2.5), alpha) +
                logpdf(Cauchy(0, 2.5), beta) +
                logpdf(Beta(2, 2), omega) + logit_jacobian(v[3]) +
                logpdf(Beta(2, 2), rho_unit) +
                logit_jacobian(v[4]) +
                logpdf(Cauchy(0, 2.5), sigma1) + log(2.0) + v[5] +
                logpdf(Cauchy(0, 2.5), sigma2) + log(2.0) + v[6]
            likelihood = zero(prior)
            for i in 1:S
                z1 = uv1[i] / sigma1
                z2 = (uv2[i] / sigma2 - rho * z1) / sqrt(1 - rho^2)
                prior += logpdf(Normal(), z1) + logpdf(Normal(), z2) -
                    log(sigma1) - log(sigma2) - 0.5 * log1p(-rho^2)
                psi = logistic(alpha + uv1[i])
                theta = logistic(beta + uv2[i])
                never_at_site = logaddexp(log(psi) +
                    logpdf(Binomial(K, theta), 0), log1p(-psi))
                if i <= n
                    likelihood += log(omega)
                    for j in 1:J
                        count = butterfly_data["X"][i, j]
                        likelihood += count > 0 ?
                            log(psi) + logpdf(Binomial(K, theta), count) :
                            never_at_site
                    end
                else
                    likelihood += logaddexp(log1p(-omega),
                        log(omega) + J * never_at_site)
                end
            end
            return prior + likelihood
        end
        butterfly_S = butterfly_data["S"]
        butterfly_point = vcat([0.1, -0.2, 0.2, 0.1, log(0.8), log(1.2)],
            0.03 .* sin.(1:butterfly_S), 0.03 .* cos.(1:butterfly_S))
        check("butterfly-multi_occupancy", 6 + 2butterfly_S,
            butterfly_oracle, [butterfly_point])

        dog_nonhier_data = posteriordb_data("dogs-dogs_nonhierarchical")
        function dog_nonhier_oracle(v)
            mu1, mu2 = v[1], v[2]
            sigma1, sigma2 = exp(v[3]), exp(v[4])
            rho = 2 * logistic(v[5]) - 1
            J = dog_nonhier_data["n_dogs"]
            z1 = view(v, 6:(5 + J))
            z2 = view(v, (6 + J):length(v))
            prior = logpdf(Logistic(), mu1) + logpdf(Logistic(), mu2) +
                logpdf(Normal(), sigma1) + log(2.0) + v[3] +
                logpdf(Normal(), sigma2) + log(2.0) + v[4] +
                log(3 / 8) + 2 * log1p(-rho^2) +
                sum(logpdf(Normal(), zi) for zi in z1) +
                sum(logpdf(Normal(), zi) for zi in z2)
            likelihood = zero(prior)
            for j in 1:J
                logit_a = mu1 + sigma1 * z1[j] + sigma2 * rho * z2[j]
                logit_b = mu2 + sigma2 * sqrt(1 - rho^2) * z2[j]
                a, b = logistic(logit_a), logistic(logit_b)
                shocks = 0
                avoids = 0
                for t in 1:dog_nonhier_data["n_trials"]
                    likelihood += logpdf(Bernoulli(a^shocks * b^avoids),
                        dog_nonhier_data["y"][j, t])
                    shocks += dog_nonhier_data["y"][j, t]
                    avoids += 1 - dog_nonhier_data["y"][j, t]
                end
            end
            return prior + likelihood
        end
        dog_nonhier_J = dog_nonhier_data["n_dogs"]
        dog_nonhier_point = vcat([-0.3, 0.2, log(0.5), log(0.6), 0.1],
            0.05 .* sin.(1:dog_nonhier_J), 0.05 .* cos.(1:dog_nonhier_J))
        check("dogs-dogs_nonhierarchical", 5 + 2dog_nonhier_J,
            dog_nonhier_oracle, [dog_nonhier_point])

        sat_data = posteriordb_data("sat-hier_2pl")
        function sat_oracle(v)
            I, J = sat_data["I"], sat_data["J"]
            theta = view(v, 1:J)
            xi1 = view(v, (J + 1):(J + I))
            xi2 = view(v, (J + I + 1):(J + 2I))
            mu1, mu2 = v[J + 2I + 1], v[J + 2I + 2]
            tau1, tau2 = exp(v[J + 2I + 3]), exp(v[J + 2I + 4])
            rho = 2 * logistic(v[J + 2I + 5]) - 1
            prior = sum(logpdf(Normal(), tj) for tj in theta) +
                logpdf(Normal(), mu1) + logpdf(Normal(0, 5), mu2) +
                logpdf(Exponential(10), tau1) + v[J + 2I + 3] +
                logpdf(Exponential(10), tau2) + v[J + 2I + 4] +
                log(35 / 64) + 4 * log1p(-rho^2)
            for i in 1:I
                z1 = (xi1[i] - mu1) / tau1
                z2 = ((xi2[i] - mu2) / tau2 - rho * z1) / sqrt(1 - rho^2)
                prior += logpdf(Normal(), z1) + logpdf(Normal(), z2) -
                    log(tau1) - log(tau2) - 0.5 * log1p(-rho^2)
            end
            likelihood = sum(logpdf(Bernoulli(logistic(
                exp(xi1[sat_data["ii"][n]]) *
                (theta[sat_data["jj"][n]] - xi2[sat_data["ii"][n]]))),
                sat_data["y"][n]) for n in 1:sat_data["N"])
            return prior + likelihood
        end
        sat_I, sat_J = sat_data["I"], sat_data["J"]
        sat_point = vcat(0.2 .* sin.(1:sat_J),
            0.03 .* sin.(1:sat_I), 0.03 .* cos.(1:sat_I),
            [0.0, 0.0, log(0.5), log(1.0), 0.1])
        check_selected(enzyme_model("sat-hier_2pl"), sat_oracle, sat_point,
            [1, sat_J, sat_J + 1, sat_J + sat_I, sat_J + sat_I + 1,
             sat_J + 2sat_I, sat_J + 2sat_I + 1, sat_J + 2sat_I + 2,
             sat_J + 2sat_I + 3, sat_J + 2sat_I + 4, sat_J + 2sat_I + 5])

        fims_data = posteriordb_data("fims_Aus_Jpn_irt-2pl_latent_reg_irt")
        fims_I, fims_J, fims_K = (fims_data[k] for k in ("I", "J", "K"))
        fims_W = fims_data["W"]
        @test all(fims_W[:, 1] .== 1)
        # The Stan source scales every covariate by twice its standard deviation;
        # its range branch for two-valued covariates is never taken.
        fims_W_adj = Float64.(fims_W)
        for k in 2:fims_K
            fims_W_adj[:, k] = (fims_W[:, k] .- mean(fims_W[:, k])) ./
                (2 * std(fims_W[:, k]))
        end
        function fims_oracle(v)
            alpha = exp.(view(v, 1:fims_I))
            beta_free = view(v, (fims_I + 1):(2fims_I - 1))
            beta = vcat(beta_free, -sum(beta_free))
            theta = view(v, (2fims_I):(2fims_I + fims_J - 1))
            lambda = view(v, (2fims_I + fims_J):length(v))
            prior = sum(logpdf(LogNormal(1, 1), alpha[i]) + v[i]
                for i in eachindex(alpha)) +
                sum(logpdf(Normal(0, 3), b) for b in beta) +
                sum(logpdf(TDist(3), l) for l in lambda) +
                sum(logpdf(Normal(sum(fims_W_adj[j, k] * lambda[k]
                    for k in 1:fims_K), 1), theta[j]) for j in 1:fims_J)
            likelihood = sum(logpdf(Bernoulli(logistic(
                alpha[fims_data["ii"][n]] * theta[fims_data["jj"][n]] -
                beta[fims_data["ii"][n]])), fims_data["y"][n])
                for n in 1:fims_data["N"])
            return prior + likelihood
        end
        fims_point = vcat(0.03 .* sin.(1:fims_I),
            0.1 .* cos.(1:(fims_I - 1)),
            0.1 .* sin.(1:fims_J), [0.1, -0.2, 0.3, -0.1])
        check_selected(enzyme_model("fims_Aus_Jpn_irt-2pl_latent_reg_irt"),
            fims_oracle, fims_point,
            [1, fims_I, fims_I + 1, 2fims_I - 1, 2fims_I, 2fims_I + fims_J - 1,
             2fims_I + fims_J, 2fims_I + fims_J + 1, 2fims_I + fims_J + 2,
             2fims_I + fims_J + 3])

        science_data = posteriordb_data("science_irt-grsm_latent_reg_irt")
        science_I, science_J = science_data["I"], science_data["J"]
        science_m = maximum(science_data["y"])
        @test science_data["K"] == 1
        @test all(science_data["W"] .== 1)
        function science_oracle(v)
            alpha = exp.(view(v, 1:science_I))
            beta_free = view(v, (science_I + 1):(2science_I - 1))
            beta = vcat(beta_free, -sum(beta_free))
            kappa_free = view(v, (2science_I):(2science_I + science_m - 2))
            kappa = vcat(kappa_free, -sum(kappa_free))
            theta = view(v, (2science_I + science_m - 1):
                (2science_I + science_m + science_J - 2))
            lambda = v[end]
            prior = sum(logpdf(LogNormal(1, 1), alpha[i]) + v[i]
                for i in eachindex(alpha)) +
                sum(logpdf(Normal(0, 3), b) for b in beta) +
                sum(logpdf(Normal(0, 3), k) for k in kappa) +
                logpdf(TDist(3), lambda) +
                sum(logpdf(Normal(lambda, 1), t) for t in theta)
            likelihood = zero(prior)
            for n in 1:science_data["N"]
                i, j = science_data["ii"][n], science_data["jj"][n]
                eta = theta[j] * alpha[i] - beta[i]
                scores = cumsum(vcat(0.0, eta .- kappa))
                likelihood += scores[science_data["y"][n] + 1] -
                    log(sum(exp, scores))
            end
            return prior + likelihood
        end
        science_point = vcat(1 .+ 0.03 .* sin.(1:science_I),
            0.05 .* cos.(1:(science_I - 1)),
            0.05 .* sin.(1:(science_m - 1)),
            0.1 .* sin.(1:science_J), [0.1])
        check_selected(enzyme_model("science_irt-grsm_latent_reg_irt"),
            science_oracle, science_point,
            [1, science_I, science_I + 1, 2science_I - 1, 2science_I,
             2science_I + science_m - 2, 2science_I + science_m - 1,
             2science_I + science_m + science_J - 2, length(science_point)])

        timss_data = posteriordb_data("timssAusTwn_irt-gpcm_latent_reg_irt")
        timss_I, timss_J, timss_K = (timss_data[k] for k in ("I", "J", "K"))
        timss_m = [maximum(timss_data["y"][n]
            for n in 1:timss_data["N"] if timss_data["ii"][n] == i)
            for i in 1:timss_I]
        timss_pos = cumsum([1; timss_m[1:(end - 1)]])
        timss_steps = sum(timss_m)
        timss_W = timss_data["W"]
        @test all(timss_W[:, 1] .== 1)
        timss_W_adj = Float64.(timss_W)
        for k in 2:timss_K
            timss_W_adj[:, k] = (timss_W[:, k] .- mean(timss_W[:, k])) ./
                (2 * std(timss_W[:, k]))
        end
        function timss_oracle(v)
            alpha = exp.(view(v, 1:timss_I))
            beta_free = view(v, (timss_I + 1):(timss_I + timss_steps - 1))
            beta = vcat(beta_free, -sum(beta_free))
            theta = view(v, (timss_I + timss_steps):
                (timss_I + timss_steps + timss_J - 1))
            lambda = view(v, (timss_I + timss_steps + timss_J):length(v))
            prior = sum(logpdf(LogNormal(1, 1), alpha[i]) + v[i]
                for i in eachindex(alpha)) +
                sum(logpdf(Normal(0, 3), b) for b in beta) +
                sum(logpdf(TDist(3), l) for l in lambda) +
                sum(logpdf(Normal(sum(timss_W_adj[j, k] * lambda[k]
                    for k in 1:timss_K), 1), theta[j]) for j in 1:timss_J)
            likelihood = zero(prior)
            for n in 1:timss_data["N"]
                i, j = timss_data["ii"][n], timss_data["jj"][n]
                steps = view(beta, timss_pos[i]:(timss_pos[i] + timss_m[i] - 1))
                scores = cumsum(vcat(0.0, theta[j] * alpha[i] .- steps))
                likelihood += scores[timss_data["y"][n] + 1] -
                    log(sum(exp, scores))
            end
            return prior + likelihood
        end
        timss_point = vcat(1 .+ 0.03 .* sin.(1:timss_I),
            0.05 .* cos.(1:(timss_steps - 1)),
            0.1 .* sin.(1:timss_J), 0.1 .* cos.(1:timss_K))
        check_selected(enzyme_model("timssAusTwn_irt-gpcm_latent_reg_irt"),
            timss_oracle, timss_point,
            [1, timss_I, timss_I + 1, timss_I + timss_steps - 1,
             timss_I + timss_steps, timss_I + timss_steps + timss_J - 1,
             timss_I + timss_steps + timss_J, length(timss_point)])

        lda_docs_data = posteriordb_data("three_docs1200-ldaK2")
        function lda_docs_oracle(v)
            theta1 = logistic.(view(v, 1:3))
            phi1, jac1 = simplex3_oracle(v[4], v[5])
            phi2, jac2 = simplex3_oracle(v[6], v[7])
            prior = sum(logit_jacobian(v[m]) for m in 1:3) +
                jac1 + jac2 + 2log(2)
            likelihood = sum(log(theta1[lda_docs_data["doc"][n]] *
                phi1[lda_docs_data["w"][n]] +
                (1 - theta1[lda_docs_data["doc"][n]]) *
                phi2[lda_docs_data["w"][n]]) for n in 1:lda_docs_data["N"])
            return prior + likelihood
        end
        check("three_docs1200-ldaK2", 7, lda_docs_oracle,
            [[0.2, -0.3, 0.4, -0.2, 0.1, 0.3, -0.4]])

        function lda_simplex_oracle(raw, V)
            probabilities = Vector{eltype(raw)}(undef, V)
            remaining = one(eltype(raw))
            jacobian = zero(eltype(raw))
            for r in 1:(V - 1)
                z = logistic(raw[r] - log(V - r))
                probabilities[r] = remaining * z
                jacobian += log(z) + log1p(-z) + log(remaining)
                remaining *= 1 - z
            end
            probabilities[V] = remaining
            return probabilities, jacobian
        end
        for (name, M) in (("three_men1-ldaK2", 6),
                ("three_men2-ldaK2", 14), ("three_men3-ldaK2", 9))
            data = posteriordb_data(name)
            V = data["V"]
            dimension = M + 2(V - 1)
            function men_lda_oracle(v)
                theta1 = logistic.(view(v, 1:M))
                phi1, jac1 = lda_simplex_oracle(view(v, (M + 1):(M + V - 1)), V)
                phi2, jac2 = lda_simplex_oracle(view(v,
                    (M + V):(M + 2(V - 1))), V)
                prior = sum(logit_jacobian(v[m]) for m in 1:M) +
                    jac1 + jac2 + 2sum(log, 1:(V - 1))
                likelihood = sum(log(theta1[data["doc"][n]] * phi1[data["w"][n]] +
                    (1 - theta1[data["doc"][n]]) * phi2[data["w"][n]])
                    for n in 1:data["N"])
                return prior + likelihood
            end
            check_selected(enzyme_model(name), men_lda_oracle,
                0.01 .* sin.(1:dimension),
                [1, M, M + 1, M + V - 1, M + V, dimension]; rtol=1e-8)
        end

        for (name, M) in (("prideprejudice_chapter-ldaK5", 61),
                ("prideprejudice_paragraph-ldaK5", 2025))
            data = posteriordb_data(name)
            V = data["V"]
            dimension = 4M + 5(V - 1)
            function pride_lda_oracle(v)
                theta = Matrix{eltype(v)}(undef, 5, M)
                phi = Vector{Vector{eltype(v)}}(undef, 5)
                prior = zero(eltype(v))
                alpha_dist, beta_dist = Dirichlet(data["alpha"]),
                    Dirichlet(data["beta"])
                for m in 1:M
                    offset = 4(m - 1)
                    topic, jac = lda_simplex_oracle(view(v,
                        (offset + 1):(offset + 4)), 5)
                    theta[:, m] = topic
                    prior += logpdf(alpha_dist, topic) + jac
                end
                for k in 1:5
                    offset = 4M + (k - 1) * (V - 1)
                    phi[k], jac = lda_simplex_oracle(view(v,
                        (offset + 1):(offset + V - 1)), V)
                    prior += logpdf(beta_dist, phi[k]) + jac
                end
                likelihood = sum(log(sum(theta[k, data["doc"][n]] *
                    phi[k][data["w"][n]] for k in 1:5)) for n in 1:data["N"])
                return prior + likelihood
            end
            point = 0.01 .* sin.(1:dimension)
            vm = enzyme_model(name; latent_source=pride_lda_source(M),
                reduce_copies=false)
            if M == 61
                check_selected(vm, pride_lda_oracle, point,
                    [1, 4M, 4M + 1, 4M + V - 1, 4M + V, dimension]; rtol=1e-8)
            else
                # 2025 documents: the density only.
                @test vm.dimension == dimension
                @test isapprox(lp(vm, point), pride_lda_oracle(point);
                    atol=1e-5, rtol=1e-8)
            end
        end

        traffic_data = posteriordb_data("traffic_accident_nyc-bym2_offset_only")
        traffic_N = traffic_data["N"]
        traffic_log_E = log.(traffic_data["E"])
        traffic_log_factorial = [sum(log, 1:y; init=0.0)
            for y in traffic_data["y"]]
        function traffic_oracle(v)
            beta0, sigma, rho = v[1], exp(v[2]), logistic(v[3])
            theta = view(v, 4:(3 + traffic_N))
            phi = view(v, (4 + traffic_N):(3 + 2traffic_N))
            prior = logpdf(Normal(), beta0) +
                logpdf(Normal(), sigma) + log(2.0) + v[2] +
                logpdf(Beta(0.5, 0.5), rho) + logit_jacobian(v[3]) +
                sum(logpdf(Normal(), t) for t in theta) +
                logpdf(Normal(0, 0.001 * traffic_N), sum(phi)) -
                0.5 * sum((phi[traffic_data["node1"][e]] -
                    phi[traffic_data["node2"][e]])^2
                    for e in 1:traffic_data["N_edges"])
            likelihood = sum(begin
                eta = traffic_log_E[n] + beta0 + sigma *
                    (sqrt(1 - rho) * theta[n] +
                    sqrt(rho / traffic_data["scaling_factor"]) * phi[n])
                traffic_data["y"][n] * eta - exp(eta) -
                    traffic_log_factorial[n]
            end for n in 1:traffic_N)
            return prior + likelihood
        end
        traffic_point = vcat([-4.0, log(0.2), 0.3],
            0.01 .* sin.(1:traffic_N), 0.01 .* cos.(1:traffic_N))
        check_selected(enzyme_model("traffic_accident_nyc-bym2_offset_only"),
            traffic_oracle, traffic_point,
            [1, 2, 3, 4, 3 + traffic_N, 4 + traffic_N, 3 + 2traffic_N]; rtol=1e-8)

        mnist_data = posteriordb_data("mnist_100-nn_rbm1bJ10")
        mnist_M, mnist_J, mnist_K = mnist_data["M"], 10, mnist_data["K"]
        mnist_x = Float64.(mnist_data["x"])
        mnist_s2_alpha = (0.05 / mnist_M^2)^2
        mnist_s2_beta = (0.05 / mnist_J^2)^2
        mnist_alpha_end = 2 + mnist_M * mnist_J
        mnist_beta_end = mnist_alpha_end + mnist_J * (mnist_K - 1)
        mnist_alpha1_end = mnist_beta_end + mnist_J
        mnist_dimension = mnist_alpha1_end + mnist_K - 1
        function mnist_oracle(v)
            sigma2_alpha, sigma2_beta = exp(v[1]), exp(v[2])
            alpha = reshape(view(v, 3:mnist_alpha_end), mnist_M, mnist_J)
            beta = reshape(view(v, (mnist_alpha_end + 1):mnist_beta_end),
                mnist_J, mnist_K - 1)
            alpha1 = view(v, (mnist_beta_end + 1):mnist_alpha1_end)
            beta1 = view(v, (mnist_alpha1_end + 1):mnist_dimension)
            prior = logpdf(InverseGamma(0.25, 0.25 * mnist_s2_alpha),
                sigma2_alpha) + v[1] +
                logpdf(InverseGamma(0.25, 0.25 * mnist_s2_beta),
                    sigma2_beta) + v[2] +
                sum(logpdf(Normal(0, sqrt(sigma2_alpha)), a) for a in alpha) +
                sum(logpdf(Normal(0, sqrt(sigma2_beta)), b) for b in beta) +
                sum(logpdf(Normal(), a) for a in alpha1) +
                sum(logpdf(Normal(), b) for b in beta1)
            likelihood = zero(prior)
            for n in 1:mnist_data["N"]
                hidden = [tanh(alpha1[j] + sum(mnist_x[n, m] * alpha[m, j]
                    for m in 1:mnist_M)) for j in 1:mnist_J]
                logits = vcat(1.0, [beta1[k] + sum(hidden[j] * beta[j, k]
                    for j in 1:mnist_J) for k in 1:(mnist_K - 1)])
                likelihood += logits[mnist_data["y"][n]] - log(sum(exp, logits))
            end
            return prior + likelihood
        end
        mnist_point = vcat([log(mnist_s2_alpha), log(mnist_s2_beta)],
            1e-8 .* sin.(1:(mnist_M * mnist_J)),
            1e-4 .* cos.(1:(mnist_J * (mnist_K - 1))),
            0.02 .* sin.(1:mnist_J),
            0.02 .* cos.(1:(mnist_K - 1)))
        check_selected(enzyme_model("mnist_100-nn_rbm1bJ10"), mnist_oracle,
            mnist_point,
            [1, 2, 3, mnist_alpha_end, mnist_alpha_end + 1, mnist_beta_end,
             mnist_beta_end + 1, mnist_dimension];
            rtol=1e-8, gradient_rtol=1e-6)

        dog_data = posteriordb_data("dogs-dogs")
        dog_y = dog_data["y"]
        dog_count, trials = size(dog_y)
        dog_shocks = [sum((dog_y[j, s] for s in 1:(t - 1)); init=0)
            for t in 1:trials for j in 1:dog_count]
        dog_avoids = [t - 1 - sum((dog_y[j, s] for s in 1:(t - 1)); init=0)
            for t in 1:trials for j in 1:dog_count]
        dog_response = vec(dog_y)

        function dogs_oracle(v)
            eta = v[1] .+ v[2] .* dog_avoids .+ v[3] .* dog_shocks
            return -3log(100sqrt(2π)) - sum(abs2, v) / 20000 +
                sum(bernoulli_logit.(dog_response, eta))
        end
        check("dogs-dogs", 3, dogs_oracle, [[0.2, -0.1, 0.1], [-0.1, 0.1, -0.1]])

        function dogs_hierarchical_oracle(v)
            a = logistic(v[1])
            b = logistic(v[2])
            p = a .^ dog_shocks .* b .^ dog_avoids
            keep = findall(>(0), dog_shocks .+ dog_avoids)
            return logit_jacobian(v[1]) + logit_jacobian(v[2]) +
                sum(dog_response[i] == 1 ? log(p[i]) : log1p(-p[i]) for i in keep)
        end
        check("dogs-dogs_hierarchical", 2, dogs_hierarchical_oracle,
            [[0.2, -0.3], [-0.2, 0.1]])

        function dogs_log_oracle(v)
            eta = v[1] .* dog_avoids .+ v[2] .* dog_shocks
            return -2log(100) + sum(bernoulli_logit.(dog_response, eta))
        end
        check("dogs-dogs_log", 2, dogs_log_oracle,
            [[-0.2, 0.3], [-0.1, 0.1]])

        glmm_data = posteriordb_data("GLMM_data-GLMM1_model")
        nsite = glmm_data["nsite"]
        obs = glmm_data["obs"]
        obssite = glmm_data["obssite"]
        log_factorial = [sum((log(i) for i in 1:k); init=0.0) for k in obs]
        function glmm_oracle(v)
            alpha = view(v, 1:nsite)
            mu_alpha = v[nsite + 1]
            sd_alpha = 5logistic(v[nsite + 2])
            eta = alpha[obssite]
            return -nsite * log(sd_alpha * sqrt(2π)) -
                sum(abs2, alpha .- mu_alpha) / (2sd_alpha^2) -
                log(10sqrt(2π)) - mu_alpha^2 / 200 +
                logit_jacobian(v[nsite + 2]) +
                sum(obs .* eta .- exp.(eta) .- log_factorial)
        end
        check("GLMM_data-GLMM1_model", nsite + 2, glmm_oracle,
            [vcat(0.04 .* sin.(1:nsite), 0.2, -0.1),
             vcat(0.03 .* cos.(1:nsite), -0.1, 0.2)])

        nes_data = posteriordb_data("nes_logit_data-nes_logit_model")
        function nes_oracle(v)
            eta = v[1] .+ v[2] .* nes_data["income"]
            return sum(bernoulli_logit.(nes_data["vote"], eta))
        end
        check("nes_logit_data-nes_logit_model", 2, nes_oracle,
            [[0.2, -0.1], [-0.1, 0.1]])

        survey_data = posteriordb_data("Survey_data-Survey_model")
        k = survey_data["k"]
        nmax = survey_data["nmax"]
        nmin = maximum(k)
        function survey_oracle(v)
            theta = logistic(v[1])
            parts = [sum(logpdf(Binomial(n, theta), ki) for ki in k) - log(nmax)
                for n in nmin:nmax]
            return reduce(logaddexp, parts) + logit_jacobian(v[1])
        end
        check("Survey_data-Survey_model", 1, survey_oracle, [[0.2], [-0.3]])
    end

    @testset "capture-recapture densities and native Enzyme gradients" begin
        function capture_data(name)
            data = posteriordb_data(name)
            y = data["y"]
            return y, vec(sum(y; dims=2))
        end

        function marginalized(omega, rowlogp, counts)
            present, absent = log(omega), log1p(-omega)
            return sum(counts[i] > 0 ? present + rowlogp[i] :
                logaddexp(present + rowlogp[i], absent) for i in eachindex(counts))
        end

        y0, counts0 = capture_data("M0_data-M0_model")
        trials0 = size(y0, 2)
        function m0_reference(v)
            omega, p = logistic.(v)
            rowlogp = [logpdf(Binomial(trials0, p), count) for count in counts0]
            return sum(logit_jacobian, v) + marginalized(omega, rowlogp, counts0)
        end
        check("M0_data-M0_model", 2, m0_reference, [[0.1, -0.2], [-0.3, 0.2]];
            atol=1e-6)

        yb, countsb = capture_data("Mb_data-Mb_model")
        function mb_reference(v)
            omega, p, c = logistic.(v)
            rowlogp = [sum(begin
                prob = j == 1 || yb[i, j - 1] == 0 ? p : c
                yb[i, j] == 1 ? log(prob) : log1p(-prob)
            end for j in axes(yb, 2)) for i in axes(yb, 1)]
            return sum(logit_jacobian, v) + marginalized(omega, rowlogp, countsb)
        end
        check("Mb_data-Mb_model", 3, mb_reference,
            [[0.1, -0.2, 0.3], [-0.3, 0.2, -0.1]]; atol=1e-6)

        yt, countst = capture_data("Mt_data-Mt_model")
        trials_t = size(yt, 2)
        function mt_reference(v)
            omega = logistic(v[1])
            p = logistic.(view(v, 2:length(v)))
            rowlogp = [sum(yt[i, j] == 1 ? log(p[j]) : log1p(-p[j])
                for j in axes(yt, 2)) for i in axes(yt, 1)]
            return sum(logit_jacobian, v) + marginalized(omega, rowlogp, countst)
        end
        check("Mt_data-Mt_model", trials_t + 1, mt_reference,
            [0.2 .* sin.(1:(trials_t + 1)), -0.2 .* cos.(1:(trials_t + 1))];
            atol=1e-6)

        yh, countsh = capture_data("Mth_data-Mth_model")
        m_h, t_h = size(yh)
        function mth_reference(v)
            omega = logistic(v[1])
            mean_lp = view(v, 2:(t_h + 1))
            sigma = 5logistic(v[t_h + 2])
            eps_raw = view(v, (t_h + 3):length(v))
            rowlogp = [sum(bernoulli_logit(yh[i, j], mean_lp[j] + sigma * eps_raw[i])
                for j in axes(yh, 2)) for i in axes(yh, 1)]
            return sum(logit_jacobian, view(v, 1:(t_h + 2))) +
                sum(logpdf(Normal(), e) for e in eps_raw) +
                marginalized(omega, rowlogp, countsh)
        end
        check("Mth_data-Mth_model", m_h + t_h + 2, mth_reference,
            [vcat(0.2 .* sin.(1:(t_h + 2)), 0.03 .* cos.(1:m_h))]; atol=1e-6)

        ybh, countsbh = capture_data("Mtbh_data-Mtbh_model")
        m_bh, t_bh = size(ybh)
        function mtbh_reference(v)
            omega = logistic(v[1])
            mean_lp = view(v, 2:(t_bh + 1))
            gamma = v[t_bh + 2]
            sigma = 3logistic(v[t_bh + 3])
            eps_raw = view(v, (t_bh + 4):length(v))
            rowlogp = [sum(bernoulli_logit(ybh[i, j],
                mean_lp[j] + sigma * eps_raw[i] +
                (j == 1 ? 0.0 : gamma * ybh[i, j - 1]))
                for j in axes(ybh, 2)) for i in axes(ybh, 1)]
            return sum(logit_jacobian, view(v, 1:(t_bh + 1))) +
                logpdf(Normal(0, 10), gamma) + logit_jacobian(v[t_bh + 3]) +
                sum(logpdf(Normal(), e) for e in eps_raw) +
                marginalized(omega, rowlogp, countsbh)
        end
        check("Mtbh_data-Mtbh_model", m_bh + t_bh + 3, mtbh_reference,
            [vcat(0.2 .* cos.(1:(t_bh + 3)), 0.03 .* sin.(1:m_bh))]; atol=1e-6)
    end

    @testset "item-response densities and native Enzyme gradients" begin
        irt_data = posteriordb_data("irt_2pl-irt_2pl")
        I, J = irt_data["I"], irt_data["J"]
        function irt_reference(v)
            sigma_theta = exp(v[1])
            theta = view(v, 2:(J + 1))
            sigma_a = exp(v[J + 2])
            a = exp.(view(v, (J + 3):(J + I + 2)))
            mu_b = v[J + I + 3]
            sigma_b = exp(v[J + I + 4])
            b = view(v, (J + I + 5):length(v))
            total = logpdf(Cauchy(0, 2), sigma_theta) + log(2) + v[1] +
                sum(logpdf(Normal(0, sigma_theta), x) for x in theta) +
                logpdf(Cauchy(0, 2), sigma_a) + log(2) + v[J + 2] +
                sum(logpdf(LogNormal(0, sigma_a), a[i]) + v[J + i + 2]
                    for i in eachindex(a)) +
                logpdf(Normal(0, 5), mu_b) +
                logpdf(Cauchy(0, 2), sigma_b) + log(2) + v[J + I + 4] +
                sum(logpdf(Normal(mu_b, sigma_b), x) for x in b)
            for j in 1:J, i in 1:I
                total += bernoulli_logit(irt_data["y"][i, j],
                    a[i] * (theta[j] - b[i]))
            end
            return total
        end
        irt_dimension = J + 2I + 4
        check("irt_2pl-irt_2pl", irt_dimension, irt_reference,
            [vcat([0.1], 0.1 .* sin.(1:J), [-0.2], 0.1 .* cos.(1:I),
                  [0.2, -0.1], 0.1 .* sin.(1:I))]; atol=1e-6)

        bones_data = posteriordb_data("bones_data-bones_model")
        function bones_reference(v)
            total = sum(logpdf(Normal(0, 36), x) for x in v)
            grade, gamma, delta, ncat = (bones_data[k] for k in
                ("grade", "gamma", "delta", "ncat"))
            for j in axes(grade, 2), i in axes(grade, 1)
                category = grade[i, j]
                category == -1 && continue
                q = [logistic(delta[j] * (v[i] - gamma[j, k]))
                    for k in 1:(ncat[j] - 1)]
                probabilities = [1 - q[1]; [q[k - 1] - q[k]
                    for k in 2:length(q)]; q[end]]
                total += log(probabilities[category])
            end
            return total
        end
        check("bones_data-bones_model", bones_data["nChild"], bones_reference,
            [0.1 .* sin.(1:bones_data["nChild"]),
             0.2 .* cos.(1:bones_data["nChild"])]; atol=1e-6)

        lsat_data = posteriordb_data("lsat_data-lsat_model")
        T, N = lsat_data["T"], lsat_data["N"]
        function lsat_reference(v)
            alpha = view(v, 1:T)
            theta = view(v, (T + 1):(T + N))
            beta = exp(v[end])
            total = sum(logpdf(Normal(0, 100), x) for x in alpha) +
                sum(logpdf(Normal(), x) for x in theta) +
                logpdf(Normal(0, 100), beta) + log(2) + v[end]
            previous = 0
            for pattern in 1:lsat_data["R"]
                for student in (previous + 1):lsat_data["culm"][pattern]
                    for question in 1:T
                        eta = beta * theta[student] - alpha[question]
                        total += bernoulli_logit(
                            lsat_data["response"][pattern, question], eta)
                    end
                end
                previous = lsat_data["culm"][pattern]
            end
            return total
        end
        lsat_vm = enzyme_model("lsat_data-lsat_model")
        @test lsat_vm.dimension == T + N + 1
        point = vcat(0.1 .* sin.(1:T), 0.05 .* cos.(1:N), [0.2])
        value, gradient = lp_and_grad(lsat_vm, point)
        @test isapprox(value, lsat_reference(point); atol=1e-6)
        for index in (1, T, T + 1, T + 500, T + N, T + N + 1)
            plus, minus = copy(point), copy(point)
            plus[index] += 1e-5
            minus[index] -= 1e-5
            finite_difference = (lsat_reference(plus) - lsat_reference(minus)) / 2e-5
            @test isapprox(gradient[index], finite_difference; atol=1e-4, rtol=1e-4)
        end
    end
end

end # module WrenPosteriorDBTests
