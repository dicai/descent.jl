
include("../optimizers/linesearch.jl")
include("../utils/functions.jl")
using Gadfly

function optimize_linear_constraints(f, A, f_g, f_h, yinit=0)
    """
    Convert constrained optimization problem min f(x) s.t. Ax = b 
        to unconstrained problem min g(y) = f(B*y + x0), where x0 is a solution to Ax = b
        and B is a basis for the kernel of A.
    
    Defaults to using Newton's line search algorithm.
    
    f: constrained objective
    A: constraint matrix
    f_g: gradient of f
    f_h: hessian of f
    yinit: init y with a vector of all yinit values
    """
    
    m, n = size(A)
    b = zeros(m)
    x0 = A \ b
    B = nullspace(A)
    
    y0 = ones(size(B)[2]) .* yinit
    
    g(y) = f(B*y + x0)
    g_g(y) = B' * f_g(B*y + x0)
    g_h(y) = B' * f_h(B*y + x0) * B
    
    yvals = line_search(g, y0, g_g, g_h, "newton", 5000)
    xvals = [B*y + x0 for y in yvals]
    lambdas = [A' \ f_g(x) for x in xvals]
    
    return yvals, xvals, lambdas
end

function generate_constraint_matrix(n)
    A = ones(2,n)
    for i in 1:n
        if i % 2 == 1
            A[2,i] = -1
        end
    end
    return A
end

A = generate_constraint_matrix(30)
b = zeros(2)
x0 = A \ b; B = nullspace(A)
f = cute
f_g = cute_g
f_h = cute_h
g(y) = f(B*y + x0)
g_g(y) = B' * f_g(B*y + x0)
g_h(y) = B' * f_h(B*y + x0) * B

yvals, xvals, lambdas = optimize_linear_constraints(f, A, f_g, f_h, 0);

niters = length(xvals)
fx = [f(x) for x in xvals]
Gadfly.plot(x=1:niters, y=fx, Geom.line, 
    Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
    Scale.x_log10, Scale.y_log10)

grads = [norm(g_g(y),2) for y in yvals]
ratios = grads[2:niters,:]./grads[1:niters-1,:]
Gadfly.plot(x=1:niters-1, y=ratios, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f_g(x)"), Guide.title("Gradient norm ratios"),
    Scale.x_log10, Scale.y_log10)

Gadfly.plot(x=1:niters, y=grads, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f_g(x)"), Guide.title("Gradient norm"),
    Scale.x_log10, Scale.y_log10)

# Lagrange multipliers
lambdas[end]

A = generate_constraint_matrix(100)
x0 = A \ b; B = nullspace(A)
f = cute
f_g = cute_g
f_h = cute_h
g(y) = f(B*y + x0)
g_g(y) = B' * f_g(B*y + x0)
g_h(y) = B' * f_h(B*y + x0) * B

yvals, xvals, lambdas = optimize_linear_constraints(f, A, f_g, f_h, 0);
# Lagrange multipliers
println(lambdas[end])
# Final value
println(f(xvals[end]))

niters = length(xvals)
fx = [f(x) for x in xvals]
#gy = [g(y) for y in yvals]

Gadfly.plot(x=1:niters, y=fx, Geom.line, 
    Guide.xlabel("iteration"), Guide.ylabel("f(x)"), Guide.title("Value of function"),
    Scale.x_log10, Scale.y_log10)

grads = [norm(g_g(y),2) for y in yvals]
ratios = grads[2:niters,:]./grads[1:niters-1,:]
Gadfly.plot(x=1:niters-1, y=ratios, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f_g(x)"), Guide.title("Gradient norm ratios"),
    Scale.x_log10, Scale.y_log10)

Gadfly.plot(x=1:niters, y=grads, Geom.line, 
Guide.xlabel("iteration"), Guide.ylabel("f_g(x)"), Guide.title("Gradient norm"),
    Scale.x_log10, Scale.y_log10)


