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

For the line search algorithms, step sizes were chosen using the Armijo
backtracking algorithm.

Basic indefinite matrix handling for Newton-type methods, based on iteratively adding a small
multiple of the identity matrix until the matrix of interest is sufficiently
positive definite.

All optimization methods are located in the folder `optimizers/`.


# Examples

Examples of the code on various functions (e.g., Rosenbrock and cute) are in the
folder jupyter, which contains various Jupyter notebooks:

1. `line_search.ipynb`
2. `trust_region.ipynb`

For each method, we try optimizing several functions, reporting the value of the
function at each iteration, the ratio of the gradient norms of the current and
previous iterations, and the gradient norms at each iteration.

