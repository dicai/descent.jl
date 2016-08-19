# descent.jl

# Overview
Julia implementation of various basic optimization methods.

1. Unconstrained optimization
    1. Line search (`optimizers/linesearch.jl`)
        1. Steepest (gradient) descent
        2. Newton's method
        3. Newton-CG (conjugate gradient)
        3. quasi-Newton BFGS
    2. Trust region (`optimizers/trustregion.jl`)
        1. Newton's method
            1. Cauchy point algorithm
            2. Dogleg algorithm
        2. CG-Steinhaug
        3. quasi-Newton SR1 (symmetric rank 1 updating)
2. Constrained optimization
    1. Linear constraints
    2. Active set methods
    3. Interior point methods

For the line search algorithms, step sizes were chosen using the Armijo
backtracking algorithm.

Basic indefinite matrix handling for Newton-type methods, based on iteratively adding a small
multiple of the identity matrix until the matrix of interest is sufficiently
positive definite.

All optimization methods are located in the folder `optimizers/`.


# Examples

To run a line search algorithm, type

```julia
xvals = line_search(rosenbrock, ones(100)*3, rosenbrock_g, rosenbrock_h, "newton", 2000)
```
Here we are optimizing the Rosenbrock function using Newton's method, starting with a vector of all 3's, and the max number of iterations to run is 2000.

To run a trust region algorithm, type
```julia
xvals = trust_region(ones(50)*10, 6, 3, 0.1, cute, cute_g, cute_h, 2000, "dogleg")
```
Here (6,3,0.1) are trust-region parameters.

Examples of the code on various functions (e.g., Rosenbrock and cute) are in the
folder jupyter, which contains various Jupyter notebooks:

1. `line_search.ipynb`
2. `trust_region.ipynb`
2. `linear_constraints.ipynb`

For each method, we try optimizing several functions, reporting the value of the
function at each iteration, the ratio of the gradient norms of the current and
previous iterations, and the gradient norms at each iteration.

