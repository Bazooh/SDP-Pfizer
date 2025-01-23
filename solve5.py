from gurobipy import Constr, Model, GRB, QuadExpr, Var, LinExpr
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

with open("brick_rp_affectation.json", mode="r") as file:
    initial_repartition_idx = json.load(file)

with open("bricks_index_values.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        brick_workload.append(float(row[1]))

N_SR = len(initial_repartition_idx)
N_bricks = len(brick_workload)
initial_offices: list[list[int]] = [[] for _ in range(N_SR)]

with open("brick_rp_distances.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for i, row in enumerate(reader):
        dist = list(map(float, row[1:]))
        distance_rp_to_brick.append(dist)
        if 0.0 in dist:
            initial_offices[dist.index(0.0)] = [
                (1 if i == brick else 0) for brick in range(N_bricks)
            ]

with open("distances22-4.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        distance_brick_to_brick.append(list(map(float, row)))

initial_repartition = {
    brick: int(sr_idx)
    for sr_idx, bricks in initial_repartition_idx.items()
    for brick in bricks
}


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


def compute_disruption(
    main_vars: list[list[Var]], initial_offices: list[list[int]]
) -> LinExpr:
    disruption = LinExpr()
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            disruption += main_vars[brick][sr_idx] * initial_offices[sr_idx][brick]
    return disruption


def compute_workload_error(vars):
    max_workload_error = 0
    for var in compute_workloads(vars):
        max_workload_error = max(max_workload_error, abs(1 - var.getValue()))

    threshold_workload = max_workload_error - epsilon
    return threshold_workload


def compute_solutions(
    m: Model, vars: list[list[Var]], main_vars: list[list[Var]]
) -> list:
    best_solutions = []

    m.setObjective(compute_distances(vars, main_vars), GRB.MINIMIZE)

    m.optimize()

    threshold_workload = compute_workload_error(vars)
    threshold_disruption = (
        compute_disruption(main_vars, initial_offices).getValue() - epsilon
    )

    workload_finished = False
    disruption_finished = False

    i = 0
    while m.Status == GRB.OPTIMAL:
        threshold_workload = compute_workload_error(vars)
        disruption = compute_disruption(main_vars, initial_offices)

        best_solutions.append((m.objVal, threshold_workload, disruption.getValue()))

        constraints: list[Constr] = []
        if i % 2 == 0:
            if workload_finished:
                continue

            workloads = compute_workloads(vars)

            for i in range(N_SR):
                constraints.append(m.addConstr(workloads[i] <= 1 + threshold_workload))
                constraints.append(m.addConstr(workloads[i] >= 1 - threshold_workload))

        else:
            if disruption_finished:
                continue

            threshold_disruption = disruption.getValue() - epsilon
            constraints.append(m.addConstr(disruption <= threshold_disruption))

        m.optimize()
        m.remove(constraints)

        i += 1

        # if m.Status != GRB.OPTIMAL:
        #     if i % 2 == 0:
        #         workload_finished = True
        #     else:
        #         disruption_finished = True

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

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        [solution[0] for solution in best_solutions],
        [solution[1] for solution in best_solutions],
        [solution[2] for solution in best_solutions],
    )
    ax.set_xlabel("Distance")
    ax.set_ylabel("Workload Error")
    ax.set_zlabel("Disruption")  # type: ignore

    plt.title("Distance vs Workload Error vs Disruption")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
