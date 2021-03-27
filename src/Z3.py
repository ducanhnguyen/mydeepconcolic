from z3 import Int, solve

from Z3 import *
if __name__ == '__main__':
    x = Int('x')
    y = Int('y')
    s = solve(x > 2, y < 10, x + 2 * y == 7)
    print(s)