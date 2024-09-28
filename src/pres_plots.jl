using GLMakie, GaussianProcesses, Optim

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
