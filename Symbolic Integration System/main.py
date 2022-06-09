import sympy
sympy.init_printing()
def derivative(f):
  t, s = sympy.symbols('t, s')
  a = sympy.symbols('a', real=True, positive=True)
  F = sympy.laplace_transform(f, t, s, noconds=True)
  sFf = s*F-f.subs(t, 0)
  simplified = sympy.simplify(sFf)
  fprime = sympy.inverse_laplace_transform(simplified, s, t, noconds=True)
  return fprime
print(derivative(3*t**2))
