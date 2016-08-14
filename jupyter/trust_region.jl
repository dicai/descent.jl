
include("../optimizers/trustregion.jl")
include("../utils/functions.jl")
using Gadfly

xvals = trust_region([3.;4.], 6, 1, 0.1, fenton, fenton_g, fenton_h, 2000, "dogleg");
println(xvals[end])

nsamps = length(xvals)
fx = [fenton(xvals[i]) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps, y=fx, Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"))

nsamps = length(xvals)
grads = [norm(fenton_g(xvals[i]), 2) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"))

Gadfly.plot(x=1:nsamps, y=grads, Geom.point, Geom.line, 
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"))

xvals = trust_region(randn(100), 6, 3, 0.1, rosenbrock, rosenbrock_g, rosenbrock_h, 1000, "dogleg");

nsamps = length(xvals)
fx = [rosenbrock(xvals[i]) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps, y=fx, Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"))

nsamps = length(xvals)
grads = [norm(rosenbrock_g(xvals[i]), 2) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratios"), Guide.title("gradient norm ratios"))

Gadfly.plot(x=1:nsamps, y=grads, Geom.point, Geom.line, 
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"))

xvals = trust_region(ones(50)*10, 6, 3, 0.1, cute, cute_g, cute_h, 2000, "dogleg");

nsamps = length(xvals)
fx = [cute(xvals[i]) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps, y=fx, Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"))

nsamps = length(xvals)
grads = [norm(cute_g(xvals[i]), 2) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("gradient norm ratio"), Guide.title("gradient norm ratios"))

Gadfly.plot(x=1:nsamps, y=grads, Geom.point, Geom.line, 
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"))


