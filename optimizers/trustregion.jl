function trust_region(x0, deltahat, delta0, eta, f, g, h, max_iter=100, tol=10e-7)

    """
    x0: vector of the initial starting point of the algorithm
    deltahat: bound on the step length
    delta0: starting trust region radius
    eta: acceptance threshold
    f: objective function
    grad: gradient function
    hess: Hessian function
    max_iter: maximum number of iterations
    tol: when to break out
    """

    @assert deltahat > 0
    @assert eta >= 0 && eta < 1/4
    @assert delta0 > 0 && delta0 < deltahat

    delta = delta0
    n = length(x0)
    xvals = reshape(x0, (1,n))


    for i in 1:max_iter
        println(i)

        xcurr = xvals[i,:]'[:,1]

        fk = f(xcurr)
        gk = g(xcurr)
        Bk = h(xcurr)

        p = solve_subproblem(delta, gk, Bk, "cauchy")
        rho = compute_rho(xcurr, f, gk, Bk, p)

        # decrease trust region radius
        if rho < 1/4
            delta *= 1/4

        else
            # increase trust region radius
            if rho > 3/4 && norm(p,2) == delta
                delta = min(2*delta, deltahat)
            end
        end

        # accept the step
        if rho > eta
            xnew = xcurr + p
        # reject the step
        else
            xnew = xcurr
        end

        xvals = vcat(xvals, xnew')

        if abs(mean(xnew - xcurr)) <= tol
            return xvals
        end

    end
    println("Finished algorithm without converging.")
    return xvals

end

function model(p, xcurr, fk, gk, Bk)
    """
    p: vector in which to step in
    xcurr: vector of the current iterate
    fk: scalar, f(xk)
    gk: vector, gradient(xk)
    Bk: symmetric matrix, e.g., Hessian(xk)
    """
    return fk + gk' * p + 1/2 * p' * Bk * p
end

function compute_rho(xcurr, f, gk, Bk, p)
    """
    xcurr: vector of the current iterate
    f: the objective function
    gk: gradient(xk)
    Bk: symmetric matrix
    p: step
    """
    fk = f(xcurr)
    n = length(p)
    return ((fk - f(xcurr + p)) ./ (model(zeros(n), xcurr, fk, gk, Bk) - model(p, xcurr, fk, gk, Bk)))[1]
end



function solve_subproblem(delta, gk, Bk, method="cauchy")
    if method == "cauchy"
        return cauchy_point(delta, gk, Bk)
    end
end


function cauchy_point(delta, gk, Bk)
    """
    delta: positive scalar, trust-region radius at current iteration
    gk: gradient at current iterate
    Bk: symmetric matrix, e.g., Hessian

    Returns the cauchy point, a vector.
    """
    tau = 1
    if (gk' * Bk * gk)[1] > 0
        tau = min(norm(gk,2)^3 ./ (delta * gk'*Bk*gk), 1)[1]
    end

    # Cauchy point equation
    return -tau * delta / norm(gk, 2) * gk
end


function dogleg(delta, gk, Bk, p_U)
    p_B = -Bk \ gk

    # compute tau
    diff = p_B - p_U
    a = norm(diff, 2)^2
    b = (2 * p_U' * diff)[1]
    c = norm(p_U, 2)^2 - delta^2
    tau = (1 + (-b + sqrt(b^2 - 4 * a *c)) / (2*a))

    try
        @assert tau >=0 && tau <= 2
    catch
        println(tau)
    end

    if tau <= 1
        return tau * p_U
    else
        return p_U + (tau-1) * diff
    end

end
