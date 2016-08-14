
include("../optimizers/linesearch.jl")
include("../utils/functions.jl")
using Gadfly

xvals = line_search(fenton, [3.;4.], fenton_g, fenton_h, "newton", 1000);
svals = line_search(fenton, [3.;4.], fenton_g, fenton_h, "steepest", 1000);

nsamps = length(xvals)
fx = [fenton(xvals[i]) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps, y=fx, Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"))

nsamps = length(xvals)
grads = [norm(fenton_g(xvals[i]), 2) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.point, Geom.line, 
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"))

xvals = line_search(rosenbrock, ones(100)*2, rosenbrock_g, rosenbrock_h, "newton", 100);

nsamps = length(xvals)
fx = [rosenbrock(xvals[i]) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps, y=fx, Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"))

nsamps = length(xvals)
grads = [norm(rosenbrock_g(xvals[i]), 2) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.point, Geom.line, 
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"))

xvals = line_search(cute, ones(100)*10, cute_g, cute_h, "newton", 100);

nsamps = length(xvals)
fx = [rosenbrock(xvals[i]) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps, y=fx, Geom.point, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"))

nsamps = length(xvals)
grads = [norm(rosenbrock_g(xvals[i]), 2) for i in 1:nsamps]
Gadfly.plot(x=1:nsamps-1, y=grads[2:nsamps,:]./grads[1:nsamps-1,:], Geom.point, Geom.line, 
    Guide.xlabel("iteration"), Guide.ylabel("gradient norm"), Guide.title("gradient norms"))


