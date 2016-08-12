function cholesky_mod(beta, H)
    if minimum(H) > 0
        tau = 0
    else
        tau = -minimum(H) + beta
        while true
            try
                L = chol(H + eye(size(H)[1]) * tau)
                return(L)
            end
        end
    end
end
