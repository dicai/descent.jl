function trust_region(x0, deltahat, delta0, eta, f, g, h, max_iter=100, method="cauchy", tol=10e-7)

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
    xvals = Vector{Vector{Float64}}(); push!(xvals, x0)
    ncalls = 0

    for i in 1:max_iter
        i % 100 == 0 && println("Iteration: ", i)

        xcurr = xvals[end]

        fk = f(xcurr)
        gk = g(xcurr)
        Bk = h(xcurr)

        if !check_posdef(Bk)
            Bk = cholesky_mod(1e-3, Bk)
            ncalls += 1
        end

        p = solve_subproblem(delta, gk, Bk, method)
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

        push!(xvals, xnew)

        if abs(mean(xnew - xcurr)) <= tol
            println("Number of indefinite fixes ", ncalls)
            println("Number of iterations: ", i)
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

function cholesky_mod(beta, H, max_iter=1000)
    @assert beta > 0
    if minimum(diag(H)) > 0
        tau = 0
    else
        tau = -minimum(diag(H)) + beta
    end
    #while true
    for i in 1:max_iter
        candidate = H + tau * eye(H)
        check_posdef(candidate) && return candidate
        tau = max(2*tau, beta)
    end

    error("Infinite loop encountered")
end

function check_posdef(A)
    try
        L = chol(A)
        return true
    catch
        return false
    end
end



function solve_subproblem(delta, gk, Bk, method="cauchy")

    if method == "cauchy"
        return cauchy_point(delta, gk, Bk)
    elseif method == "dogleg"
        p_U = -gk' * gk ./ ((gk' * Bk * gk)' .* gk)
        p_B = -Bk \ gk
        pnorm = norm(p_B, 2)
        if pnorm > delta
            return cauchy_point(delta, gk, Bk)
            #return p_B
        elseif pnorm < delta
            return p_B
            #-Bk \ gk
        else
            return dogleg(delta, gk, Bk, p_U)
        end
    else
        println("defaulting to Cauchy point")
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
    #p_U = -gk' * gk ./ ((gk' * Bk * gk)' .* gk)

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
