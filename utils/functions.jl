using Calculus


################################################################################
# Function 1
################################################################################
f = function(x)
    n = length(x)
    total = 0
    for i in 1:n-1
        total += (x[i+1]-x[i])^2 + (exp(x[i]) - 1)^2
    end
    return(total + (exp(x[n]) - 1)^2)
end

grad_f = function(x)
    n = length(x)
    J = zeros(n)
    for i in 1:n
        if i == 1
            J[1] = -2 * (x[2]-x[1]) + 2*(exp(2*x[1]) - exp(x[1]))
        elseif i == n
            J[n] = 2 * (x[n]-x[n-1]) + 2*(exp(2*x[n]) - exp(x[n]))
        else
            J[i] = -2 * (x[i+1]-x[i]) + 2 * (x[i]-x[i-1]) + 2*(exp(2*x[i])-exp(x[i]))
        end
    end
    return(J)
end

hess_f = function(x)
    n = length(x)
    H = zeros(n,n)
    for i in 1:n
        for j in 1:n
            if i == j
                H[i,j] = 2 + 4*exp(2*x[i]) - 2*exp(x[i])
                if (i != n) && (i != 1)
                    H[i,j] += 2
                end

            elseif abs(i-j) == 1
                H[i,j] = -2
            else
                H[i,j] = 0
            end
        end
    end
    return(H)
end

# test that the numerical gradients equal the analytically computed gradients
#x = [1.;2.;3.]
#println(sum(grad_f(x)-Calculus.gradient(f,x)))
#println(sum(hess_f(x)-Calculus.hessian(f,x)))

f_g = Calculus.gradient(f,x)
f_h = Calculus.hessian(f,x)

################################################################################
# Rosenbrock
################################################################################

function rosenbrock(x)
    """
    Assumes x has an even number of elemnts.
    """
    n = length(x)
    @assert n%2 == 0
    total = 0
    for i in 1:Integer(n/2)
        total += (1-x[2*i-1])^2 + 10*(x[2*i] - x[2*i-1]^2)^2
    end
    return total
end

rosenbrock_g = Calculus.gradient(rosenbrock)
rosenbrock_h = Calculus.hessian(rosenbrock)


################################################################################
# Cute
################################################################################

function cute(x)
    n = length(x)
    total = 0
    for i in 1:n-4
        total += (-4x[i]+3)^2 + (x[i]^2 + 2*x[i+1]^2 + 3*x[i+2]^2 + 4*x[i+3]^2 + 5*x[n]^2)^2
		return total
    end
end

cute_g = Calculus.gradient(cute)
cute_h = Calculus.hessian(cute)


################################################################################
# Fenton
################################################################################

function fenton(x)
    """
    Using Newton's method, the starting point [3;2] converges but [3;4] diverges
    """
    x1 = x[1]; x2 = x[2];
    return((12 + x1*x1 + (1+x2*x2)/(x1*x1) + (x1*x1*x2*x2+100)/((x1*x2)^4)) / 10)
end

fenton_g = Calculus.gradient(fenton)
fenton_h = Calculus.hessian(fenton)
