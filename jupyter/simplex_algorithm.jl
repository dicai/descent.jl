
include("../utils/functions.jl")
include("../optimizers/activeset.jl")

using Gadfly

## Toy example 

A = [1 1   1 0;
     2 1/2 0 1]
b = Vector{Float64}([5, 8])
c = Vector{Float64}([-4, -2, 0, 0])

#assts = [true,false,true,false]
assts = [false,false,true,true]

#x = [4.,0.,1.,0.]
x = [0.,0.,5.,8.]
run_simplex(assts,x,A,b,c,2)
#simplex_iteration(assts, x, A, b, c)

# phase I of simplex 
x0 = [0.,0.,0.,0.,5.,8.] # last 2 entries are z1, z2
A0 = [1 1   1 0    1 0;
     2 1/2 0 1    0 1]
c0 = [0., 0., 0., 0., 1., 1.]

assts0 = [false,false,false,false,true,true]

assts_start,x_start = run_simplex(assts0,x0,A0,b,c0,5)

function run_program(m, num_iters=100)
    n = 3*m
    # generate random values for A
    A = randn(m,n)
    b = randn(m) + 1
    c = randn(n) * 2 + 2
    
    return run_two_stage_simplex(A, b, c, num_iters)
end

function run_experiment(num_rounds, num_iters)
    times = []
    for m in range(10,10,num_rounds)
        x, assts, time = run_program(m, num_iters)
        push!(times,time)
    end
    return times
end

times = run_experiment(5,100)

num_rounds = 20

length(times)

#Gadfly.plot(x=range(10,10,num_rounds), y=times, Guide.XLabel("m"),
#    Guide.YLabel("time (s)"),)


