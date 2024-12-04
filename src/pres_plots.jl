using GLMakie, GaussianProcesses, Optim, CairoMakie, Colors

# Put this in the repl to activate some sane defaults
# set_theme!(fontsize = 11, fonts = (; regular = "new computer modern"))

axis_kwargs = (xminortickalign=1.0, yminortickalign=1.0, xgridvisible=false,
                 ygridvisible=false, xminorticks=IntervalsBetween(2),
                 yminorticks=IntervalsBetween(2),
                 xminorticksvisible=true, yminorticksvisible=true, 
                 xtickalign=0.5, ytickalign=0.5, 
                 xticksize=10.0, yticksize=10.0, yminorticksize=5.0,
                 xminorticksize=5.0)

function hprof_plot(t, X)
    f = Figure()
    N = size(X, 2)
    ax = Axis(f[1,1]; xlabel="Time (s)", ylabel="Temperature (°C)", axis_kwargs...)
    colours = [:orange, :red, :green, :blue, :purple, :black, :pink]
    #styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :solid, :dot]
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    for i in 1:N
        lines!(ax, t, X[:,i]; linestyle=:solid, color=colours[i], label=labels[i])
    end
    axislegend(ax, position=:rt, framevisible=false)
    return f
end
export hprof_plot

function ΔL_plot(X, Y)
    f = Figure()
    N = size(Y, 2)
    ax = Axis(f[1,1]; xlabel="Temperature (°C)", ylabel="ΔL (μm)", axis_kwargs..., xreversed = true)
    colours = [:orange, :red, :green, :blue, :purple, :black, :pink]
    #styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :solid, :dot]
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    for i in 1:N
        lines!(ax, X[:,i], Y[:,i]; linestyle=:solid, color=colours[i], label=labels[i])
    end
    axislegend(ax, position=:rt, framevisible=false)
    return f
end
export ΔL_plot

# Find the average cooling rate for the given temperature range
function get_average_CR(X, dTdt; Trange=(700,600))
    ncols = size(X,2)
    nrows = size(X,1)
    cr_averages = Vector{Float64}(undef, ncols)
    for i in 1:ncols
        in_range = []
        for j in 1:nrows
            if Trange[2] < X[j, i] < Trange[1]
                push!(in_range, j)
            end
        end
        cr_averages[i] = round(abs(mean(dTdt[in_range, i])), digits=1)
    end
    return cr_averages
end
export get_average_CR

function ΔL_plot(X, Y, dTdt; obfuscate = false)
    CairoMakie.activate!()
    inch = 96; pt = 4/3;
    f = Figure(size = (7.5inch, 4.5inch), fontsize=12pt)
    N = size(Y, 2)
    ax = Axis(f[1,1]; xlabel="Temperature (°C)",
              ylabel="ΔL (μm)", axis_kwargs..., xreversed = true)
    CRvec = get_average_CR(X, dTdt)
    Tmax = maximum(X)

    if obfuscate
        ax.xtickformat = x -> string.(round.(x./Tmax, sigdigits=2))
        ax.xlabel = "Temperature (scaled)"
    end

    #colors = range(colorant"red", stop=colorant"blue", length=size(X,2))
    #styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :solid, :dot]
    #labels = ["A", "B", "C", "D", "E", "F", "G"]
    for i in sortperm(CRvec)
        color = weighted_color_mean(log(CRvec[i] + 1)/log(maximum(CRvec) + 1),
                                    colorant"red", colorant"blue")
        lines!(ax, X[:,i], Y[:,i]; linestyle=:solid, color=color,
               label="CR = $(CRvec[i]) °C/s")
    end
    axislegend(ax, position=:rt, framevisible=false)
    return f
end
export ΔL_plot

function hprof_gen(rates, t, X)
    n, N = size(X)
    gens = [build_x_star(900.0, 30.0, i, 0.07, n)[1] for i in rates]

    f = Figure()
    ax = Axis(f[1,1]; xlabel="Time (s)", ylabel="Temperature (°C)", axis_kwargs...)
    colours = [:orange, :red, :green, :blue, :purple, :black, :pink]
    #styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :solid, :dot]
    labels = ["H", "I", "J", "K", "L", "M", "N"]
    for i in 1:length(rates)
        lines!(ax, t, gens[i]; linestyle=:solid, color=colours[i], label=labels[i])
    end
    axislegend(ax, position=:rt, framevisible=false)
    return f
end
export hprof_gen


function GP_plot(x)
    y = sin.(x * 1.5) .* exp.(-x) .+ 0.005.*randn(length(x))

    mZero = MeanZero()
    kern = SE(0.0, 0.0) 
    logObsNoise = log(sqrt(0.005))
    gp = GaussianProcesses.GP(x, y, mZero, kern, logObsNoise)
    optimize!(gp; method=ConjugateGradient())
    μ, σ2 = GaussianProcesses.predict_y(gp, range(0, 2π, length =100));


    f = Figure(size=(800,480))
    ax = Axis(f[1,1]; xlabel="x", ylabel="y", axis_kwargs..., aspect = 2.0,
              limits = (-0.05, 6.5, -0.25, 0.5))

    x_infer = range(0, 2π, length =100)
    y_infer = sin.(x_infer * 1.5) .* exp.(-x_infer)
    lines!(ax, x_infer, y_infer, color=:gray5, linestyle=:dot)
    band!(ax, x_infer, μ - sqrt.(σ2), μ + sqrt.(σ2), alpha=0.5)
    lines!(ax, x_infer, μ)
    scatter!(ax, x, y, color=:black)
    rowsize!(f.layout, 1, Auto(1))
    return f
end
export GP_plot
