# June 9, 2022

# The purpose of this is to introduce a method of calculating derivatives and integrals symbolically in order to avoid brute force and pattern matching.

# Import sympy
import sympy
# Set up pretty printing
sympy.init_printing()
# Set up symbolic variables used in both functions
t, s = sympy.symbols('t, s')
# This function returns the derivative of f
def derivative(f):
  # Define the Laplace Transform of f as L
  L = sympy.laplace_transform(f, t, s, noconds=True)
  # The derivative will be the Inverse Laplace Transform of sL(s)-f(0)
  sLf = s*L-f.subs(t, 0)
  sLf_simplified = sympy.simplify(sLf)
  derivative = sympy.inverse_laplace_transform(sLf_simplified, s, t, noconds=True)
  return derivative
# This function returns the integral of f
def integral(f):
  # Define the Laplace Transform of f as L
  L = sympy.laplace_transform(f, t, s, noconds=True)
  # The integral will be the Inverse Laplace Transform of L(s)/s-f(0)
  Lsf= L/s-f.subs(t, 0)
  Lsf_simplified = sympy.simplify(Lsf)
  integral = sympy.inverse_laplace_transform(Lsf_simplified, s, t, noconds=True)
  return integral
# Test cases
print(derivative(3*t**2))
print(integral(3*t**2))
