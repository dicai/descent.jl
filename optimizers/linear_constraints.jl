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
