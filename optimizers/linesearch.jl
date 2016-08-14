function line_search(f, x0, g, h, opt_type="steepest", max_iter=50, tol=1e-6)
    """
    f: objective function
    x0: starting value of iterate
    grad_f: gradient of the objective
    hess_f: Hessian of the objective, not required for all methods
    opt_type: type of step direction
    """
    n = length(x0)
    xvals = Vector{Vector{Float64}}(); push!(xvals, x0)
    ncalls = 0

    for i in 1:max_iter
        i % 100 == 0 && println("Iteration: ", i)
        xcurr = xvals[end]

        p, alpha, nc = compute_steps(xcurr, f, g, h, opt_type)
        xnew = xcurr + alpha * p
        push!(xvals, xnew)
        ncalls += nc

        if abs(mean(xnew - xcurr)) <= tol
            println("Number of indefinite fixes ", ncalls)
            println("Number of iterations ", i)
            return(xvals)
        end
    end

    println("Finished algorithm without converging.")
    return(xvals)
end

function compute_steps(xcurr, f, g, h, opt_type)
    """
    Computes the step size and step direction depending on the type of method

    xcurr: current iterate
    f: objective function
    grad_f: gradient of the objective
    hess_f: Hessian of the objective, not required for all methods
    opt_type: type of step direction
    """

    fk = f(xcurr)
    gk = g(xcurr)
    ncalls = 0

    if opt_type == "steepest"
        p = steepest_descent(gk)
        alpha = get_step_size(3, 0.9, xcurr, f, g, p)
    elseif opt_type == "newton"
        Bk = h(xcurr)
        if !check_posdef(Bk)
            ncalls += 1
            Bk = cholesky_mod(1e-3, Bk)
        end
        p = newton(gk, Bk)
        alpha = get_step_size(1, 0.9, xcurr, f, g, p)
    end
    return p, alpha, ncalls
end

function get_step_size(alpha_init, rho, xcurr, f, grad_f, p_k, c=1e-4)

    alpha = alpha_init
    fk = f(xcurr)
    gradfk = grad_f(xcurr)

    while f(xcurr + alpha*p_k) > fk + c * alpha * sum(gradfk'*p_k)
        alpha = rho * alpha
    end
    return(alpha)
end

function steepest_descent(gk)
    return(-gk)
end

function newton(gk, Bk)
    return(-Bk \ gk)
end

function cholesky_mod(beta, H)
    @assert beta > 0
    if minimum(diag(H)) > 0
        tau = 0
    else
        tau = -minimum(diag(H)) + beta
    end
    while true
        candidate = H + tau * eye(H)
        check_posdef(candidate) && return candidate
        tau = max(2*tau, beta)
    end
end

function check_posdef(A)
    try
        L = chol(A)
        return true
    catch
        return false
    end
end
