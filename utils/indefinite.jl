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
