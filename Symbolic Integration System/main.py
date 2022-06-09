import sympy
sympy.init_printing()
t, s = sympy.symbols('t, s')
def derivative(f):
  a = sympy.symbols('a', real=True, positive=True)
  F = sympy.laplace_transform(f, t, s, noconds=True)
  sFf = s*F-f.subs(t, 0)
  simplified = sympy.simplify(sFf)
  fprime = sympy.inverse_laplace_transform(simplified, s, t, noconds=True)
  return fprime
def integral(f):
  a = sympy.symbols('a', real=True, positive=True)
  F = sympy.laplace_transform(f, t, s, noconds=True)
  sFf = F/s-f.subs(t, 0)
  simplified = sympy.simplify(sFf)
  bigF = sympy.inverse_laplace_transform(simplified, s, t, noconds=True)
  return bigF
print(derivative(3*t**2))
print(integral(3*t**2))
