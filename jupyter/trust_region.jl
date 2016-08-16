
include("../optimizers/trustregion.jl")
include("../utils/functions.jl")
using Gadfly

xvals = trust_region([3.;4.], 6, 1, 0.1, fenton, fenton_g, fenton_h, 2000, "dogleg");
cvals = trust_region([3.;4.], 6, 1, 0.1, fenton, fenton_g, fenton_h, 2000, "cg_steihaug");

println(xvals[end])

nsamps = length(xvals)
nsamps2 = length(cvals)

fx = [fenton(xvals[i]) for i in 1:nsamps]
fx2 = [fenton(cvals[i]) for i in 1:nsamps2]


Gadfly.plot(layer(x=1:nsamps, y=fx, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=fx2, Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3, y=fx3, Geom.line, Theme(default_color=color("orange"))),
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
Scale.x_log10, Scale.y_log10)

nsamps = length(xvals)

grads = [norm(fenton_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(fenton_g(cvals[i]), 2) for i in 1:nsamps2]
#grads3 = [norm(fenton_g(cvals[i]), 2) for i in 1:nsamps3]


Gadfly.plot(
layer(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2-1, y=grads2[2:nsamps2,:]./grads2[1:nsamps2-1,:], Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3-1, y=grads3[2:nsamps3,:]./grads3[1:nsamps3-1,:], Geom.line, Theme(default_color=color("orange"))),
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
Scale.x_log10, Scale.y_log10)

Gadfly.plot(layer(x=1:nsamps, y=grads, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=grads2, Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3, y=grads3, Geom.line, Theme(default_color=color("orange"))),
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
    Scale.x_log10, Scale.y_log10)

xvals = trust_region(randn(100), 6, 3, 0.1, rosenbrock, rosenbrock_g, rosenbrock_h, 1000, "dogleg");
cvals = trust_region(randn(100), 6, 3, 0.1, rosenbrock, rosenbrock_g, rosenbrock_h, 1000, "cg_steihaug");

nsamps = length(xvals)
nsamps2 = length(cvals)

func = rosenbrock
func_g = rosenbrock_g

fx = [func(xvals[i]) for i in 1:nsamps]
fx2 = [func(cvals[i]) for i in 1:nsamps2]


Gadfly.plot(layer(x=1:nsamps, y=fx, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=fx2, Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3, y=fx3, Geom.line, Theme(default_color=color("orange"))),
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
Scale.x_log10, Scale.y_log10)

nsamps = length(xvals)
nsamps2 = length(cvals)

grads = [norm(func_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(func_g(cvals[i]), 2) for i in 1:nsamps2]
#grads3 = [norm(func_g(cvals[i]), 2) for i in 1:nsamps3]


Gadfly.plot(
layer(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2-1, y=grads2[2:nsamps2,:]./grads2[1:nsamps2-1,:], Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3-1, y=grads3[2:nsamps3,:]./grads3[1:nsamps3-1,:], Geom.line, Theme(default_color=color("orange"))),
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
Scale.x_log10, Scale.y_log10)

Gadfly.plot(layer(x=1:nsamps, y=grads, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=grads2, Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3, y=grads3, Geom.line, Theme(default_color=color("orange"))),
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
    Scale.x_log10, Scale.y_log10)

xvals = trust_region(ones(50)*10, 6, 3, 0.1, cute, cute_g, cute_h, 2000, "dogleg");
cvals = trust_region(ones(50)*10, 6, 3, 0.1, cute, cute_g, cute_h, 2000, "cg_steihaug");

nsamps = length(xvals)
nsamps2 = length(cvals)

func = cute
func_g = cute_g

fx = [func(xvals[i]) for i in 1:nsamps]
fx2 = [func(cvals[i]) for i in 1:nsamps2]


Gadfly.plot(layer(x=1:nsamps, y=fx, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=fx2, Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3, y=fx3, Geom.line, Theme(default_color=color("orange"))),
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
Scale.x_log10, Scale.y_log10)

nsamps = length(xvals)

grads = [norm(func_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(func_g(cvals[i]), 2) for i in 1:nsamps2]
#grads3 = [norm(func_g(cvals[i]), 2) for i in 1:nsamps3]


Gadfly.plot(
layer(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2-1, y=grads2[2:nsamps2,:]./grads2[1:nsamps2-1,:], Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3-1, y=grads3[2:nsamps3,:]./grads3[1:nsamps3-1,:], Geom.line, Theme(default_color=color("orange"))),
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
Scale.x_log10, Scale.y_log10)

Gadfly.plot(layer(x=1:nsamps, y=grads, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=grads2, Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3, y=grads3, Geom.line, Theme(default_color=color("orange"))),
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
    Scale.x_log10, Scale.y_log10)

xvals = trust_region(ones(50)*10, 6, 3, 0.1, cute2, cute2_g, cute2_h, 2000, "dogleg");
cvals = trust_region(ones(50)*10, 6, 3, 0.1, cute2, cute2_g, cute2_h, 2000, "cg_steihaug");

nsamps = length(xvals)
nsamps2 = length(cvals)

func = cute2
func_g = cute2_g

fx = [func(xvals[i]) for i in 1:nsamps]
fx2 = [func(cvals[i]) for i in 1:nsamps2]


Gadfly.plot(layer(x=1:nsamps, y=fx, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=fx2, Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3, y=fx3, Geom.line, Theme(default_color=color("orange"))),
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
Scale.x_log10, Scale.y_log10)

nsamps = length(xvals)

grads = [norm(func_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(func_g(cvals[i]), 2) for i in 1:nsamps2]
#grads3 = [norm(func_g(cvals[i]), 2) for i in 1:nsamps3]


Gadfly.plot(
layer(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2-1, y=grads2[2:nsamps2,:]./grads2[1:nsamps2-1,:], Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3-1, y=grads3[2:nsamps3,:]./grads3[1:nsamps3-1,:], Geom.line, Theme(default_color=color("orange"))),
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
Scale.x_log10, Scale.y_log10)

Gadfly.plot(layer(x=1:nsamps, y=grads, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=grads2, Geom.line, Theme(default_color=color("red"))),
#layer(x=1:nsamps3, y=grads3, Geom.line, Theme(default_color=color("orange"))),
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"),
Guide.manual_color_key("Legend", ["Newton dogleg", "Steihaug-CG", "quasi-Newton SR1"], ["blue", "red", "orange"]),
    Scale.x_log10, Scale.y_log10)


