# Optimization as Root-Finding

Training a machine-learning model means minimizing a loss $L(w)$, and the minimum occurs where the gradient vanishes: $\nabla L(w) = 0$. Finding that point is a root-finding problem, so classical root-finding methods and their convergence theory apply directly to optimization.

This project implements three optimizers for logistic regression, estimates each method's empirical order of convergence, and compares their computational cost.

## Methods

- **Gradient descent** with backtracking line search
- **Newton's method** (solving the linear system $Hd = \nabla L$ rather than inverting $H$)
- **BFGS**, a quasi-Newton method that approximates the Hessian from successive gradients

## Results

| Method | Measured order | Type |
|---|---|---|
| Gradient descent | ≈ 1.0 | linear |
| BFGS | superlinear | intermediate |
| Newton's method | ≈ 2.0 | quadratic |

The order of convergence is estimated from the slope of $\log e_{k+1}$ versus $\log e_k$. Newton's method reaches the minimum in about 8 iterations versus hundreds for gradient descent, but each step costs $O(d^3)$ to solve the linear system, so the most efficient method depends on the problem's dimension and conditioning.

## Contents

Full implementation, convergence analysis, and plots: `Optimization as Root-Finding.ipynb`

## Tools

Python, NumPy, SciPy, Matplotlib.

