import sympy
sympy.init_printing()
t, s = sympy.symbols('t, s')
a = sympy.symbols('a', real=True, positive=True)
#f = sympy.exp(2*t)*sympy.sin(5*t)
f = 4*t+1
F = sympy.laplace_transform(f, t, s)
print(F)
print(s*F-1)
