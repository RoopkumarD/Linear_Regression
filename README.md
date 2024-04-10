# Simple Least Squares General Linear Regression Model

This repository contains a Python implementation of a general linear regression model using the method of least squares,
implemented with NumPy.

## Overview

In the `linear_model.py` file, you'll find the `general_linear_model` function, which returns the coefficients of the
equation of a hyperplane. The hyperplane equation is in the form:

$z = a_{1}x_{1} + a_{2}x_{2} + ... + a_{n}x_{n} + d$

Here, $z$ is the target variable (dependent variable), and $x_{1}$ through $x_{n}$ are the features (independent variables).
$a_{1}$ through $a_{n}$ are the coefficients corresponding to each feature, and $d$ is a constant term.

## Methodology

### Derivation

The algorithm is derived from the method of least squares. Initially, a linear equation in a plane ($y = mx + c$) was
considered, and the sum of squares was taken as the cost function. By finding the derivative of the cost function with
respect to the coefficients and the constant term, and equating it to zero, the minima were obtained. This approach was
extended to a general equation ($z = ax + by + c$, $k = ax + by + cz + d$), resulting in a linear equation with variable
coefficients and constants.

### Matrix Representation

The problem was then formulated in matrix form: $Mx = b$, where:
- $M = A^T \cdot A$, and $A$ is the feature matrix with columns representing features and rows representing observations,
  with an additional column of ones appended for the constant term.
- $b = A^T \cdot y$

Finally, the solution was obtained as $x = M^{-1} \cdot b$.

## Usage

To use the `general_linear_model` function, pass in the feature matrix `x` and the target variable `y`. Ensure that `y` is
of shape $(n, 1)$, where $n$ is the number of observations.

```python
import numpy as np
from linear_model import general_linear_model

# Example usage
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([[3], [7], [11]])

coefficients = general_linear_model(x, y)
print("Coefficients:", coefficients)
```

This will output the coefficients of the hyperplane equation.

## Contributing

Feel free to contribute to this project by forking the repository, making changes, and submitting a pull request.
