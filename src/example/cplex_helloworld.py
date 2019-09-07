'''
Example of linear constraints. The linear constraints are solved by cplex.

Module cplex comes from the academic version of IBM ILOG CPLEX Optimization Studio. Link download: https://www.ibm.com/products/ilog-cplex-optimization-studio

This source code is modified from the link: https://github.com/cswaroop/cplex-samples/blob/master/lpex1.py

Reference: https://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.3/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/Python/eg_lpex1.html
'''
import cplex
my_colnames = ["x1", "x2", "x3"]

# Maximize x1  + 2x2 + 3x3
my_obj      = [1.0, 2.0, 3.0]

# with these bounds 0 <= x1 <= 40 and 0 <= x2 <= infinity and 0 <= x3 <= infinity
my_ub       = [40.0, +1000000, +1000000]
my_lb       = [0, 0, 0]

# subject to –x1 + x2 + x3 <= 20 and  x1 – 3x2 + x3 <= 30
my_rhs      = [20.0, 30.0]

my_rownames = ["c1", "c2"]
my_sense    = "LL"

def define_linear_function(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.variables.add(obj = my_obj, ub = my_ub, lb = my_lb, names = my_colnames)

    rows = [[[0,"x2","x3"],[-1.0, 1.0,1.0]],
            [["x1",1,2],[ 1.0,-3.0,1.0]]]

    # Adds linear constraints to the problem.
    prob.linear_constraints.add(lin_expr = rows, senses = my_sense,
                                rhs = my_rhs, names = my_rownames)
    return prob

def print_result(prob):
    print()
    # solution.get_status() returns an integer code
    print(f"Solution status = {prob.solution.get_status()}")

    # the following line prints the corresponding string
    print(prob.solution.status[prob.solution.get_status()])
    print(f"Maximum objective  = {prob.solution.get_objective_value()}")

    x     = prob.solution.get_values()

    for j in range(prob.variables.get_num()):
        print(f'Column {j}:  Value = {x[j]}')

if __name__ == "__main__":
    prob = cplex.Cplex()
    prob = define_linear_function(prob)
    prob.solve()
    print_result(prob)