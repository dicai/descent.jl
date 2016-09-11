
include("../optimizers/linesearch.jl")
include("../utils/functions.jl")
include("../utils/plot.jl")

using Gadfly

# general function to plot the output of the algorithms
function plot_values(fx1, fx2, fx3, fx4, xlab, ylab, title)
    nsamps = length(fx1)
    nsamps2 = length(fx2)
    nsamps3 = length(fx3)
    nsamps4 = length(fx4)
    
    Gadfly.plot(layer(x=1:nsamps, y=fx1, Geom.line, 
            Theme(default_color=color("blue"))),
        layer(x=1:nsamps2, y=fx2, Geom.line, 
            Theme(default_color=color("red"))),
        layer(x=1:nsamps3, y=fx3, Geom.line, 
            Theme(default_color=color("orange"))),
        layer(x=1:nsamps4, y=fx4, Geom.line, 
            Theme(default_color=color("purple"))),
        Guide.xlabel(xlab), Guide.ylabel(ylab), 
        Guide.title(title),
        Guide.manual_color_key("Legend", 
            ["Newton", "steepest", "Newton-CG", "BFGS"], 
            ["blue", "red", "orange", "purple"]),
        Scale.x_log10, Scale.y_log10)
end

@time xvals = line_search(fenton, [3.;4.], 
    fenton_g, fenton_h, "newton", 1000);
@time svals = line_search(fenton, [3.;4.], 
    fenton_g, fenton_h, "steepest", 1000);
@time cvals = line_search(fenton, [3.;4.], 
    fenton_g, fenton_h, "newton_CG", 1000);
@time qvals = line_search(fenton, [3.;4.], 
    fenton_g, fenton_h, "BFGS", 1000);

nsamps = length(xvals)
nsamps2 = length(svals)
nsamps3 = length(cvals)
nsamps4 = length(qvals)


fx = [fenton(xvals[i]) for i in 1:nsamps]
fx2 = [fenton(svals[i]) for i in 1:nsamps2]
fx3 = [fenton(cvals[i]) for i in 1:nsamps3]
fx4 = [fenton(qvals[i]) for i in 1:nsamps4]

plot_values(fx, fx2, fx3, fx4, "iteration", "f(x)", "Value of function")

grads = [norm(fenton_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(fenton_g(svals[i]), 2) for i in 1:nsamps2]
grads3 = [norm(fenton_g(cvals[i]), 2) for i in 1:nsamps3]
grads4 = [norm(fenton_g(qvals[i]), 2) for i in 1:nsamps4]

r1 = compute_grad_ratio(grads)
r2 = compute_grad_ratio(grads2)
r3 = compute_grad_ratio(grads3)
r4 = compute_grad_ratio(grads4)

plot_values(r1, r2, r3, r4, "iteration", 
    "gradient norm ratios", "gradient norm ratios")

plot_values(grads, grads2, grads3, grads4, "iteration", 
    "gradient norms", "gradient norms")

@time xvals = line_search(rosenbrock, ones(100)*3, rosenbrock_g, rosenbrock_h, 
    "newton", 2000, 1e-8);
@time svals = line_search(rosenbrock, ones(100)*3, rosenbrock_g, rosenbrock_h, 
    "steepest", 5000, 1e-8);
@time cvals = line_search(rosenbrock, ones(100)*3, rosenbrock_g, rosenbrock_h, 
    "newton_CG", 2000, 1e-8);
@time qvals = line_search(rosenbrock, ones(100)*3, rosenbrock_g, rosenbrock_h, 
    "BFGS", 2000, 1e-8);

func = rosenbrock
func_g = rosenbrock_g

nsamps = length(xvals)
nsamps2 = length(svals)
nsamps3 = length(cvals)
nsamps4 = length(qvals)


fx = [func(xvals[i]) for i in 1:nsamps]
fx2 = [func(svals[i]) for i in 1:nsamps2]
fx3 = [func(cvals[i]) for i in 1:nsamps3]
fx4 = [func(qvals[i]) for i in 1:nsamps4]

plot_values(fx, fx2, fx3, fx4, "iteration", "f(x)", "Value of function")

nsamps = length(xvals)

grads = [norm(func_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(func_g(svals[i]), 2) for i in 1:nsamps2]
grads3 = [norm(func_g(cvals[i]), 2) for i in 1:nsamps3]
grads4 = [norm(func_g(qvals[i]), 2) for i in 1:nsamps4]

r1 = compute_grad_ratio(grads)
r2 = compute_grad_ratio(grads2)
r3 = compute_grad_ratio(grads3)
r4 = compute_grad_ratio(grads4)

plot_values(r1, r2, r3, r4, "iteration", "gradient norm ratios", 
    "gradient norm ratios")

plot_values(grads, grads2, grads3, grads4, "iteration", "gradient norms", 
    "gradient norms")

@time xvals = line_search(cute, ones(100)*10, cute_g, cute_h, "newton", 
    2000, 1e-8);
@time svals = line_search(cute, ones(100)*10, cute_g, cute_h, "steepest", 
    2000, 1e-8);
@time cvals = line_search(cute, ones(100)*10, cute_g, cute_h, "newton_CG", 
    2000, 1e-8);
@time qvals = line_search(cute, ones(100)*10, cute_g, cute_h, "BFGS", 
    2000, 1e-8);

func = cute
func_g = cute_g

nsamps = length(xvals)
nsamps2 = length(svals)
nsamps3 = length(cvals)
nsamps4 = length(qvals)

fx = [func(xvals[i]) for i in 1:nsamps]
fx2 = [func(svals[i]) for i in 1:nsamps2]
fx3 = [func(cvals[i]) for i in 1:nsamps3]
fx4 = [func(qvals[i]) for i in 1:nsamps4]


plot_values(fx, fx2, fx3, fx4, "iteration", "f(x)", "Value of function")

grads = [norm(func_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(func_g(svals[i]), 2) for i in 1:nsamps2]
grads3 = [norm(func(cvals[i]), 2) for i in 1:nsamps3]
grads4 = [norm(func_g(qvals[i]), 2) for i in 1:nsamps4]

r1 = compute_grad_ratio(grads)
r2 = compute_grad_ratio(grads2)
r3 = compute_grad_ratio(grads3)
r4 = compute_grad_ratio(grads4)

plot_values(r1, r2, r3, r4, "iteration", "gradient norm ratio", 
    "gradient norm ratios")

plot_values(grads, grads2, grads3, grads4, "iteration", "gradient norm", 
    "gradient norms")




