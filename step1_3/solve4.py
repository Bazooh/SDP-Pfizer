import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../.'))

from gurobipy import Model, GRB, QuadExpr, Var, LinExpr
import csv
import json
import matplotlib.pyplot as plt

epsilon = 0.001
LOWER_WORKLOAD = 0.8
UPPER_WORKLOAD = 1.2

brick_workload: list[float] = []
distance_rp_to_brick: list[list[float]] = []
"""[brick][rp]"""
distance_brick_to_brick: list[list[float]] = []

with open("data/brick_rp_distances.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        distance_rp_to_brick.append(list(map(float, row[1:])))


with open("data/bricks_index_values.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        brick_workload.append(float(row[1]))

with open("data/brick_rp_affectation.json", mode="r") as file:
    initial_repartition_idx = json.load(file)

with open("data/distances22-4.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        distance_brick_to_brick.append(list(map(float, row)))

initial_repartition = {
    brick: int(sr_idx)
    for sr_idx, bricks in initial_repartition_idx.items()
    for brick in bricks
}

N_SR = len(initial_repartition_idx)
N_bricks = len(brick_workload)


def compute_distances(vars: list[list[Var]], main_vars: list[list[Var]]) -> QuadExpr:
    distances = QuadExpr()
    for sr_idx in range(N_SR):
        for main_brick in range(N_bricks):
            for brick in range(N_bricks):
                distances += (
                    main_vars[main_brick][sr_idx]
                    * vars[brick][sr_idx]
                    * distance_brick_to_brick[main_brick][brick]
                )
    return distances


def compute_workloads(vars: list[list[Var]]) -> list[LinExpr]:
    workloads = [LinExpr() for _ in range(N_SR)]
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            workloads[sr_idx] += vars[brick][sr_idx] * brick_workload[brick]
    return workloads


def compute_disruption(vars: list[list[Var]]) -> LinExpr:
    disruption = LinExpr()
    for brick in range(N_bricks):
        sr_idx = initial_repartition[brick]
        disruption += brick_workload[brick] * (1 - vars[brick][sr_idx])
    return disruption


def compute_solutions(
    m: Model, vars: list[list[Var]], main_vars: list[list[Var]]
) -> list:
    best_solutions = []

    m.setObjective(compute_distances(vars, main_vars), GRB.MINIMIZE)

    m.optimize()

    max_workload_error = 0
    for var in compute_workloads(vars):
        max_workload_error = max(max_workload_error, abs(1 - var.getValue()))

    threshold_workload = max_workload_error - epsilon

    while m.Status == GRB.OPTIMAL:
        workloads = compute_workloads(vars)

        best_solutions.append((m.objVal, threshold_workload + epsilon))

        max_workload_error = 0
        for var in workloads:
            max_workload_error = max(max_workload_error, abs(1 - var.getValue()))

        threshold_workload = max_workload_error - epsilon

        for i in range(N_SR):
            m.addConstr(workloads[i] <= 1 + threshold_workload)
            m.addConstr(workloads[i] >= 1 - threshold_workload)

        m.optimize()

    return best_solutions


def main():
    # initialize model
    m = Model("solve")
    vars: list[list[Var]] = []
    main_vars: list[list[Var]] = []

    ###############
    # In our model, the ith variable represent which SR the ith block is allocated to
    ###############

    # create variables
    for brick in range(N_bricks):
        v = []
        main_v = []
        for sr_idx in range(N_SR):
            v.append(m.addVar(vtype=GRB.BINARY))
            main_v.append(m.addVar(vtype=GRB.BINARY))
        vars.append(v)
        main_vars.append(main_v)

    # create constraints
    workloads = compute_workloads(vars)
    for sr_idx in range(N_SR):
        m.addConstr(LOWER_WORKLOAD <= workloads[sr_idx])
        m.addConstr(workloads[sr_idx] <= UPPER_WORKLOAD)
        s = LinExpr()
        for brick in range(N_bricks):
            s += main_vars[brick][sr_idx]
        m.addConstr(s == 1)

    for brick in range(N_bricks):
        s = LinExpr()
        for sr_idx in range(N_SR):
            s += vars[brick][sr_idx]
        m.addConstr(s == 1)

    # Multi-objective, with epsilon-constraint strategy
    # We fix the disruption, and optimize the distance
    best_solutions = compute_solutions(m, vars, main_vars)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        [solution[0] for solution in best_solutions],
        [solution[1] for solution in best_solutions],
    )
    plt.xlabel("Distance")
    plt.ylabel("Workload Loss")
    plt.title("Distance vs Workload Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
