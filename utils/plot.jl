

function compute_grad_ratio(grads)
    nsamps = length(grads)
    return grads[2:nsamps-1,:]./grads[1:nsamps-2,:]
end
