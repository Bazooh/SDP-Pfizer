from gurobipy import Model, GRB, QuadExpr, Var, LinExpr
import csv
import json
import matplotlib.pyplot as plt

epsilon = 0.001

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
    m1: Model,
    vars1: list[list[Var]],
    main_vars1: list[list[Var]],
    m2: Model,
    vars2: list[list[Var]],
    main_vars2: list[list[Var]],
) -> list:
    best_solutions = []

    m1.setObjective(compute_distances(vars1, main_vars1), GRB.MINIMIZE)
    m2.setObjective(compute_distances(vars2, main_vars2), GRB.MINIMIZE)

    m1.optimize()  # 1 is for workload
    m2.optimize()  # 2 is for disruption

    threshold_workload = compute_workload_error(vars1)
    threshold_disruption = (
        compute_disruption(main_vars2, initial_offices).getValue() - epsilon
    )

    workload_finished = False
    disruption_finished = False

    while not (workload_finished and disruption_finished):
        if not disruption_finished:
            disruption = compute_disruption(main_vars2, initial_offices)
            workload = compute_workload_error(vars2)

            threshold_disruption = (
                min(disruption.getValue(), threshold_disruption) - epsilon
            )

            best_solutions.append((m2.objVal, workload, disruption.getValue()))

            m2.addConstr(disruption <= threshold_disruption)

            m2.optimize()
            if m2.Status != GRB.OPTIMAL:
                disruption_finished = True

        if not workload_finished:
            workload = compute_workload_error(vars1)
            disruption = compute_disruption(main_vars1, initial_offices)

            threshold_workload = min(workload, threshold_workload) - epsilon

            best_solutions.append((m1.objVal, workload, disruption.getValue()))

            workloads = compute_workloads(vars1)

            for i in range(N_SR):
                m1.addConstr(workloads[i] <= 1 + threshold_workload)
                m1.addConstr(workloads[i] >= 1 - threshold_workload)

            m1.optimize()
            if m1.Status != GRB.OPTIMAL:
                workload_finished = True

    # remove duplicates
    best_solutions = list(set(best_solutions))

    return best_solutions


def main():
    # initialize model
    m1 = Model("solve")
    vars1: list[list[Var]] = []
    main_vars1: list[list[Var]] = []

    m2 = Model("solve")
    vars2: list[list[Var]] = []
    main_vars2: list[list[Var]] = []

    ###############
    # In our model, the ith variable represent which SR the ith block is allocated to
    ###############

    # create variables
    for brick in range(N_bricks):
        v = []
        main_v = []
        for sr_idx in range(N_SR):
            v.append(m1.addVar(vtype=GRB.BINARY))
            main_v.append(m1.addVar(vtype=GRB.BINARY))
        vars1.append(v)
        main_vars1.append(main_v)

    for brick in range(N_bricks):
        v = []
        main_v = []
        for sr_idx in range(N_SR):
            v.append(m2.addVar(vtype=GRB.BINARY))
            main_v.append(m2.addVar(vtype=GRB.BINARY))
        vars2.append(v)
        main_vars2.append(main_v)

    # create constraints
    for sr_idx in range(N_SR):
        s = LinExpr()
        for brick in range(N_bricks):
            s += main_vars1[brick][sr_idx]
        m1.addConstr(s == 1)

    for brick in range(N_bricks):
        s = LinExpr()
        for sr_idx in range(N_SR):
            s += vars1[brick][sr_idx]
        m1.addConstr(s == 1)

    for sr_idx in range(N_SR):
        s = LinExpr()
        for brick in range(N_bricks):
            s += main_vars2[brick][sr_idx]
        m2.addConstr(s == 1)

    for brick in range(N_bricks):
        s = LinExpr()
        for sr_idx in range(N_SR):
            s += vars2[brick][sr_idx]
        m2.addConstr(s == 1)

    # Multi-objective, with epsilon-constraint strategy
    # We fix the disruption, and optimize the distance
    best_solutions = compute_solutions(m1, vars1, main_vars1, m2, vars2, main_vars2)

    print("NB SOLUTIONS:", len(best_solutions))
    for i in range(len(best_solutions)):
        print(best_solutions[i])

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
