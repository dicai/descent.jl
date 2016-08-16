
include("../optimizers/linesearch.jl")
include("../utils/functions.jl")
using Gadfly

xvals = line_search(fenton, [3.;4.], fenton_g, fenton_h, "newton", 1000);
svals = line_search(fenton, [3.;4.], fenton_g, fenton_h, "steepest", 1000);

nsamps = length(xvals)
nsamps2 = length(svals)

fx = [fenton(xvals[i]) for i in 1:nsamps]
fx2 = [fenton(svals[i]) for i in 1:nsamps2]


Gadfly.plot(layer(x=1:nsamps, y=fx, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=fx2, Geom.line, Theme(default_color=color("red"))),
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
Scale.x_log10, Scale.y_log10)

Gadfly.plot(layer(x=1:nsamps, y=xvals, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=svals, Geom.line, Theme(default_color=color("red"))),
Guide.xlabel("iteration"), Guide.ylabel("x"), Guide.title("Value of function"),
Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
Scale.x_log10, Scale.y_log10)

nsamps = length(xvals)

grads = [norm(fenton_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(fenton_g(svals[i]), 2) for i in 1:nsamps2]

Gadfly.plot(
layer(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2-1, y=grads2[2:nsamps2,:]./grads2[1:nsamps2-1,:], Geom.line, Theme(default_color=color("red"))),
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"),
Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
Scale.x_log10, Scale.y_log10)

grads2[end-10:end]

[log(x) for x in grads2[end-10:end]]

[log(abs(log(x))) for x in grads[1:end-1]]

[log(x) for x in grads[1:end-1]]

Gadfly.plot(layer(x=1:nsamps, y=grads, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=grads2, Geom.line, Theme(default_color=color("red"))),
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"),
    Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
    Scale.x_log10, Scale.y_log10)

xvals = line_search(rosenbrock, ones(100)*2, rosenbrock_g, rosenbrock_h, "newton", 2000);
svals = line_search(rosenbrock, ones(100)*2, rosenbrock_g, rosenbrock_h, "steepest", 2000);

func = rosenbrock
func_g = rosenbrock_g

nsamps = length(xvals)
nsamps2 = length(svals)

fx = [func(xvals[i]) for i in 1:nsamps]
fx2 = [func(svals[i]) for i in 1:nsamps2]

Gadfly.plot(layer(x=1:nsamps, y=fx, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=fx2, Geom.line, Theme(default_color=color("red"))),
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
Scale.x_log10, Scale.y_log10)

grads = [norm(func_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(func_g(svals[i]), 2) for i in 1:nsamps2]

Gadfly.plot(
layer(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2-1, y=grads2[2:nsamps2,:]./grads2[1:nsamps2-1,:], Geom.line, Theme(default_color=color("red"))),
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"),
Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
Scale.x_log10, Scale.y_log10)

Gadfly.plot(layer(x=1:nsamps, y=grads, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=grads2, Geom.line, Theme(default_color=color("red"))),
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"),
    Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
    Scale.x_log10, Scale.y_log10)

xvals = line_search(cute, ones(100)*10, cute_g, cute_h, "newton", 1000);
svals = line_search(cute, ones(100)*10, cute_g, cute_h, "steepest", 2000);

func = cute
func_g = cute_g

nsamps = length(xvals)
nsamps2 = length(svals)

fx = [func(xvals[i]) for i in 1:nsamps]
fx2 = [func(svals[i]) for i in 1:nsamps2]

Gadfly.plot(layer(x=1:nsamps, y=fx, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=fx2, Geom.line, Theme(default_color=color("red"))),
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
Scale.x_log10, Scale.y_log10)

grads = [norm(func_g(xvals[i]), 2) for i in 1:nsamps]
grads2 = [norm(func_g(svals[i]), 2) for i in 1:nsamps2]

Gadfly.plot(
layer(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2-1, y=grads2[2:nsamps2,:]./grads2[1:nsamps2-1,:], Geom.line, Theme(default_color=color("red"))),
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"),
Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
Scale.x_log10, Scale.y_log10)

Gadfly.plot(layer(x=1:nsamps, y=grads, Geom.line, Theme(default_color=color("blue"))),
layer(x=1:nsamps2, y=grads2, Geom.line, Theme(default_color=color("red"))),
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"),
    Guide.manual_color_key("Legend", ["Newton", "steepest"], ["blue", "red"]),
    Scale.x_log10, Scale.y_log10)


