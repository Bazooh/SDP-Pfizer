from gurobipy import Model, Var, LinExpr
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-6

reference_universities = np.array(
    [
        [27.5, 30, 8, 83, 55],
        [32.5, 37.5, 45, 45, 91.5],
        [25, 32.5, 16, 90, 25],
        [30, 35, 4, 75, 85],
        [25, 32.5, 24, 100, 100],
        [39, 40, 8, 100, 15],
    ]
)

new_universities = np.array(
    [
        [27.5, 30, 8, 83, 55],
        [32.5, 37.5, 45, 45, 91.5],
        [25, 32.5, 16, 90, 25],
        [30, 35, 4, 75, 85],
        [25, 32.5, 24, 100, 100],
        [39, 40, 8, 100, 15],
        [32, 35, 25, 85, 12],
    ]
)

n_uni, n = reference_universities.shape
L: list[int] = [4 for _ in range(n)]

x_min: np.ndarray = np.min(reference_universities, axis=0)
x_max: np.ndarray = np.max(reference_universities, axis=0)


def score_i(x: np.ndarray, i: int, vars: list[list[Var]]) -> LinExpr:
    x_norm = L[i] * (x - x_min[i]) / (x_max[i] - x_min[i])
    k = int(x_norm)
    if k == L[i]:
        return LinExpr(vars[i][k])
    return vars[i][k] + (x_norm - k) * (vars[i][k + 1] - vars[i][k])


def score(x: np.ndarray, vars: list[list[Var]]) -> LinExpr:
    s = LinExpr()
    for i in range(n):
        s += score_i(x[i], i, vars)
    return s


def print_results(vars: list[list[Var]], errors: list[Var]):
    print("Optimal solution")
    for i in range(n):
        print(f"Criteria {i + 1}:")
        for k in range(L[i] + 1):
            print(f"  s_{i}(x_{i}_{k}) = {vars[i][k].X}")

    print("Errors:")
    for i in range(len(errors)):
        print(f"University {i + 1}: {errors[i].X}")


def plot_results(vars: list[list[Var]]):
    for i in range(n):
        x_vals = np.linspace(0, 1, L[i] + 1)
        y_vals = [vars[i][k].X for k in range(L[i] + 1)]
        plt.plot(x_vals, y_vals, label=f"Criteria {i + 1}")

    plt.xlabel("Normalized Value")
    plt.ylabel("Score")
    plt.title("Linear Curves for Each Criteria")
    plt.legend()
    plt.show()

    new_uni_scores = [score(x, vars) for x in new_universities]
    new_uni_scores_values = [s.getValue() for s in new_uni_scores]

    plt.figure()
    plt.bar(
        range(len(new_uni_scores_values)),
        new_uni_scores_values,
        tick_label=[f"Uni {i + 1}" for i in range(len(new_uni_scores_values))],
    )
    plt.xlabel("Universities")
    plt.ylabel("Scores")
    plt.title("Scores of New Universities")
    plt.show()


def main():
    m = Model("solve")
    vars: list[list[Var]] = [
        [m.addVar(name=f"s_{i}(x_{i}_{k})") for k in range(L[i] + 1)] for i in range(n)
    ]

    for i in range(n):
        m.addConstr(vars[i][0] == 0)

    s = LinExpr()
    for i in range(n):
        s += vars[i][-1]
    m.addConstr(s == 1)

    for i in range(n):
        for k in range(L[i]):
            m.addConstr(vars[i][k + 1] >= vars[i][k])

    errors = [m.addVar(name=f"error_{i}") for i in range(n_uni)]

    uni_scores = [
        score(x, vars) + errors[i] for i, x in enumerate(reference_universities)
    ]
    for i in range(len(uni_scores) - 1):
        m.addConstr(uni_scores[i] <= uni_scores[i + 1] + EPSILON)

    m.setObjective(sum(e * e for e in errors))
    m.optimize()

    print_results(vars, errors)
    plot_results(vars)


if __name__ == "__main__":
    main()
