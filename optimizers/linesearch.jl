function line_search(f, x0, g, h, opt_type="steepest", max_iter=50, tol=1e-6)
    """
    f: objective function
    x0: starting value of iterate
    grad_f: gradient of the objective
    hess_f: Hessian of the objective, not required for all methods
    opt_type: type of step direction
        "steepest"
        "newton"
        "newton_CG"
    """
    println("Using method ", opt_type)
    n = length(x0)
    xvals = Vector{Vector{Float64}}(); push!(xvals, x0)
    ncalls = 0

    for i in 1:max_iter
        i % 100 == 0 && println(i)
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

    elseif opt_type == "newton_CG"
        Bk = h(xcurr)
        if !check_posdef(Bk)
            ncalls += 1
            Bk = cholesky_mod(1e-3, Bk)
        end
        p = newton_CG(gk, Bk)
        alpha = get_step_size(1, 0.9, xcurr, f, g, p)
    elseif opt_type == "quasi_Newton_BFGS"
        println(1)
    elseif opt_type == "quasi_Newton_SR1"
        println(1)
    else
        println(1)
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


function newton_CG(gk, Bk)
    gradnorm = norm(gk, 2)[1]
    tol = min(0.5, sqrt(gradnorm)) * gradnorm

    z = zeros(gk); r = gk; d = -gk; j = 1

    while true
        if (d' * Bk * d)[1] <= 0
            if j == 1
                return -fk
            else
                return z
            end
        end
        a = (r'*r ./ (d' * Bk * d))[1]
        z += a * d
        rold = r
        r += a * Bk * d
        if norm(r, 2) <= tol
            return z
        end
        beta = ((r' * r) ./ (rold' * rold))[1]
        d = -r + beta * d
        j += 1
    end
end

