from z3 import *

x = Int('x')
y = Int('y')
eq = [x>2, y<10]
solve(eq)
#solve(x > 2, y < 10, x + 2*y == 7)