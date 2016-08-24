# define exceptions for stopping conditions in algorithm
type OptimalPointFound <: Exception end
type UnboundedFunctionException <: Exception end

function get_min(x, d, assts)
    n = length(assts)
    e = zeros(n)
    e[assts] = d
    ratios = []
    for i in 1:n
        if (assts[i]) & (e[i] > 0)
            push!(ratios, (x[i]/e[i], i))
        end
    end

    try
        return minimum(ratios)
    catch
        println(ratios)
    end
end

function get_q(s_N, assts)
    n = length(assts)
    e = zeros(n)
    e[!assts] = s_N

    try
        return indmax(e.<0)
    catch
            print e
    end
end

# a single iteration of the simplex algorithm
function simplex_iteration(assts, x, A, b, c)
    """
    We have a linear program of the form: min c^T x s.t. Ax = b

    arguments:
        assts: true indicates the element belongs to the basic set and false to the nonbasic set
        x: the vector x as given above in the LP
        A: constraint matrix, of the form Ax = b
        b: solutions of constraints
        c: coefficients of the LP function
    """

    B = A[:,assts]
    N = A[:,!assts]

    x_B = x[assts]
    x_N = x[!assts]


    # solve B^T lambda = c_B for lambda
    lambda = B' \ c[assts]

    # compute s_N = c_N - N^\top \lambda
    s_N = c[!assts] - N' * lambda

    # check if all elements of s_N are >= 0
    if all(s_N .>= 0)
        throw(OptimalPointFound())
    end

    # select q \in N with s_q < 0 as entering index
    q = get_q(s_N, assts) # here we always choose the first negative element

    # solve Bd = A_q for d
    d = B \ A[:,q]

    # check d
    if all(s_N .>= 0)
        throw(UnboundedFunctionException())
    end

    # calculate x_q+
    x_q, p = get_min(x, d, assts)

    # update x
    x[assts] = x[assts] - d * x_q
    x[q] = x_q

    # update the assts
    assts[p] = !assts[p]
    assts[q] = !assts[q]

    # returns new assts, x
    return assts, x

end

function run_simplex(assts, x, A, b, c, num_iters, verbose=false)

    for i in 1:num_iters
        #println("Starting iteration: ", i)

        try
            assts, x = simplex_iteration(assts, x, A, b, c)

            if verbose
                println("Resulting x, assts: \n\t", (x,assts))
            end

        catch err
            if isa(err, UnboundedFunctionException)

                println("Unbounded function exception!")
                return assts, x

            elseif isa(err, OptimalPointFound)

                println("Optimal point found!")

                if verbose
                    println("Optimal x, assts: \n\t", (x,assts))
                end

                return assts, x
            else
                println(err)

                throw(err)
            end
        end

    end
    return assts, x
end

function get_starting_values(A, b)
    """
    given a LP min c'x s.t. Ax = b, we want to create the new LP e'z s.t. Ax + Ez = b, where x=0, e=ones(.)

    A: the original constraint matrix
    b: the original b

    returns new A, x, c, assts to pass into simplex algorithm
    """
    nrows, ncols = size(A)
    # init x, c
    x = zeros(ncols + nrows)
    c = zeros(ncols + nrows)

    # add z to x
    x[ncols+1:ncols+nrows,:] = abs(b')
    c[ncols+1:ncols+nrows,:] = 1.

    # assignments are non-zero elements of x
    assts = (x .!= 0)

    E = zeros(nrows, nrows)
    vals = b .> 0

    for i in 1:nrows
        if vals[i]
            E[i,i] = 1
        else
            E[i,i] = -1
        end
    end

    return [A E], x, c, assts

end

function run_two_stage_simplex(A, b, c, num_iters)
    time1 = time()
    A0, x0, c0, assts0 = get_starting_values(A, b)

    nrows, ncols = size(A)
    l = length(x0)

    println("\nRunning Phase I of simplex algorithm\n")
    # run phase I
    assts_start, x_start = run_simplex(assts0,x0,A0,b,c0,num_iters)

    # run phase II
    try
        @assert x_start[l] == 0
    catch err
        println("No feasible solution for m=", size(A)[2])
    end

    # discard z
    xnew = x_start[1:l-nrows]
    assts_new = assts_start[1:l-nrows]

    println("\nRunning Phase II of simplex algorithm \n")

    x, assts = run_simplex(assts_new,xnew,A,b,c,num_iters)

    time2 = time()

    @printf("\nTime: %.5f (s)", time2-time1)

    return x, assts, (time2-time1)

end

#run_two_stage_simplex(A, b, c, 10)




function solve_subproblem(assts, x_k, G, c, A, b)
    g_k = G * x_k + c

    ## if all assignments are false
    if all(assts .== false)
        return G \ -g_k

    else
        dimx = length(x_k)
        A_w = A[assts,:]
        k,k = size(G)
        m,n = size(A_w)

        # construct matrix
        mat = [G A_w'; A_w zeros(k+m - n,k+m - n)]

        # pad with 0's
        d, = size(g_k)
        answ = [-g_k; zeros(k+m - d)]

        # compute solution and get x
        sol = mat \ answ
        x = sol[1:dimx,:]

        return x
    end
end

function compute_multipliers(assts, x_k, G, c, A)
    g = G * x_k + c
    lambda = A[assts,:]' \ g

    n = length(assts)
    return lambda, (1:n)[assts]
end

function compute_step_length(A, b, x_k, p_k, assts)
    """
    Returns step length alpha_k and blocking constraint of choice -- if the latter is 0, then alpha_k = 1
    """
    n = length(b)

    # by default, return 1 with no blocking constraints
    min = 1
    blocking = 0

    for i in 1:n
        # if i is in the working set, continue
        if assts[i]
            continue
        end

        a_i = A[i,:] ## this is a row vector so no need to transpose
        prod = a_i * p_k
        @assert length(prod) == 1

        # if a_i^ p_k >= 0, continue
        if (prod)[1] >= 0
            continue
        end

        val = (b[i] - a_i * x_k) / (prod)
        @assert length(val) == 1
        if val[1] < min
            min = val[1]
            blocking = i
        end
    end

    return min, blocking
end


function active_set_iteration(assts, x_k, G, c, A, b)
    """
    assts: assignments of constraint indices to the working set -- 1's are in the working set, 0's are not
    x_k: the current value of x_k obtained from the previous iteration
    G: the matrix in the quadratic function, i.e., x^T G x + x^T c
    c: the c from the quadratic function above
    A: matrix, where each row is the a_i from the constraint
    b: vector where each index is the b_i corresponding to constraint i
    """
    # solve subproblem to find p_k
    p_k = solve_subproblem(assts, x_k, G, c, A, b)

    #if all(p_k == 0) sufficient close to 0
    if all(abs(p_k-0) .<= 1e-6)

        if all(assts .== false)
            throw(OptimalPointFound())
        else

            # compute lagrange multipliers lambda_i that satisfy
            lambda, inds = compute_multipliers(assts, x_k, G, c, A)
        end

        # if lambda_i >= 0 for all i in W_k \cap I
        if all(lambda .>= 0)
            # stop with solution x* = x_k
            throw(OptimalPointFound())

        # (not all lambda_i >= 0)
        else
            # index of most negative multiplier lambda_j
            j = inds[indmin(lambda)]

            # remove j from working set
            assts[j] = false
            x_new = x_k
        end

    # (p_k != 0)
    else
        # compute step length alpha_k
        alpha_k, blocking = compute_step_length(A, b, x_k, p_k, assts)

        # update x_{k} += alpha_k * p_k
        x_new = x_k + alpha_k * p_k

        # if there are blocking constraints, update W_k (assts) by adding one of them to W_k
        if blocking != 0
            assts[blocking] = true
        end
    end

    return assts, x_new

end


function run_active_set(num_iters, assts, x_0, G, c, A, b, verbose=true)
    x_k = x_0
    for i in 1:num_iters
        if verbose
            println("Iteration:\t", i)
        end
        try
            assts, x_k = active_set_iteration(assts, x_k, G, c, A, b)
            if verbose
                println("Resulting x, assts: \n\t", (x_k,assts))
            end
        catch err
            if isa(err, OptimalPointFound)

                println("Optimal point found!")

                if verbose
                    println("Optimal x, assts: \n\t", (x_k,assts))
                end

                return assts, x_k
            else
                println(err)
                throw(err)
            end
        end

    end
    return assts, x_k
end

function run_active_set_full(num_iters, assts, x_0, G, c, A, b, verbose=true)
    x_k = x_0
    vals = []
    for i in 1:num_iters
        if verbose
            println("Iteration:\t", i)
        end
        try
            assts, x_k = active_set_iteration(assts, x_k, G, c, A, b)

            val = 1/2 * x_k' * G * x_k + c' * x_k
            push!(vals, val)

            if verbose
                println("Resulting x, assts: \n\t", (x_k,assts))
            end
        catch err
            if isa(err, OptimalPointFound)

                println("Optimal point found!")

                if verbose
                    println("Optimal x, assts: \n\t", (x_k,assts))
                end

                return assts, x_k, vals
            else
                println(err)
                throw(err)
            end
        end

    end
    return assts, x_k, vals
end
