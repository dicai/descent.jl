function line_search(f, x0, grad_f, hess_f, opt_type="steepest", max_iter=50, tol=10e-7)
    """
    f: objective function
    x0: starting value of iterate
    grad_f: gradient of the objective
    hess_f: Hessian of the objective, not required for all methods
    opt_type: type of step direction
    """
    n = length(x0)
    xvals = reshape(x0, (1,n))
    alpha = 1

    for i in 1:max_iter
        println(i)
        xcurr = xvals[i,:]'[:,1]

        p, alpha = compute_steps(xcurr, f, grad_f, hess_f, opt_type)
        xnew = xcurr + alpha * p
        xvals = vcat(xvals, xnew')

        if abs(mean(xnew - xcurr)) <= tol
            return(xvals)
        end
    end
    println("Finished algorithm without converging.")
    return(xvals)
end

function compute_steps(xcurr, f, grad_f, hess_f, opt_type)
    """
    Computes the step size and step direction depending on the type of method

    xcurr: current iterate
    f: objective function
    grad_f: gradient of the objective
    hess_f: Hessian of the objective, not required for all methods
    opt_type: type of step direction
    """
    if opt_type == "steepest"
            p = steepest_descent(xcurr, grad_f)
            alpha = get_step_size(1, 0.8, xcurr, f, grad_f, p)
    elseif opt_type == "newton"
            p = newton(xcurr, grad_f, hess_f)
            alpha = 1
    end
    return p, alpha
end

function get_step_size(alpha_init, rho, xcurr, f, grad_f, p_k, c=10e-4)

    alpha = alpha_init
    fk = f(xcurr)
    gradfk = grad_f(xcurr)

    while (f(xcurr + alpha*p_k) > fk + sum(c * alpha * gradfk'*p_k))
        alpha = rho * alpha
    end

    return(alpha)
end

################################################################################
# step directions
################################################################################

function steepest_descent(xcurr, grad_f)
    return(-grad_f(xcurr))
end

function newton(xcurr, grad_f, hess_f)
    return(-hess_f(xcurr) \ grad_f(xcurr))
end
