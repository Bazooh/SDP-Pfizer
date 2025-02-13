import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../.'))

from gurobipy import Model, GRB, Var, LinExpr
import csv
import json
import matplotlib.pyplot as plt

epsilon = 0.001
LOWER_WORKLOAD = 0.9
UPPER_WORKLOAD = 1.1

brick_workload: list[float] = []
distance_matrix: list[list[float]] = []

# initial_repartition_idx = {
#     0: [3, 4, 5, 6, 7, 14],
#     1: [9, 10, 11, 12, 13],
#     2: [8, 15, 16, 17],
#     3: [0, 1, 2, 18, 19, 20, 21],
# }

with open("data/brick_rp_distances10-100.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        distance_matrix.append(list(map(float, row[1:])))


with open("data/bricks_index_values10-100.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        brick_workload.append(float(row[1]))

with open("data/brick_rp_affectation10-100.json", mode="r") as file:
    initial_repartition_idx = json.load(file)

initial_repartition = {
    brick: int(sr_idx)
    for sr_idx, bricks in initial_repartition_idx.items()
    for brick in bricks
}

N_SR = len(initial_repartition_idx)
N_bricks = len(brick_workload)


def compute_distances(vars: list[list[Var]]) -> LinExpr:
    halfDistances = LinExpr()
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            halfDistances += vars[brick][sr_idx] * distance_matrix[brick][sr_idx]
    return 2 * halfDistances  # parce que aller-retour


def compute_workloads(vars: list[list[Var]]) -> list[LinExpr]:
    workloads = [LinExpr() for _ in range(N_SR)]
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            workloads[sr_idx] += vars[brick][sr_idx] * brick_workload[brick]
    return workloads


def compute_size_disruption(vars: list[list[Var]]):
    size_disruption = 0
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            if sr_idx != initial_repartition[brick] and vars[brick][sr_idx].X == 1:
                size_disruption += 1
                break
    return size_disruption


def compute_disruption(vars: list[list[Var]]) -> LinExpr:
    halfDisruption = LinExpr()
    for brick in range(N_bricks):
        sr_idx = initial_repartition[brick]
        halfDisruption += brick_workload[brick] * (1 - vars[brick][sr_idx])
    return 2 * halfDisruption


def compute_solutions(m, vars):
    best_solutions = []

    m.setObjective(compute_distances(vars), GRB.MINIMIZE)

    m.optimize()

    while m.Status == GRB.OPTIMAL:
        disruption = compute_disruption(vars)

        best_solutions.append(
            {
                "objVal": m.objVal,
                "disruption": disruption.getValue(),
                "size_disruption": compute_size_disruption(vars),
                "total_distance": compute_distances(vars).getValue(),
                "max_workload": max(
                    [workload.getValue() for workload in compute_workloads(vars)]
                ),
            }
        )

        m.addConstr(disruption <= disruption.getValue() - epsilon)
        m.optimize()

    return best_solutions


def get_non_dominated_solutions(plot=False):
    # initialize model
    m = Model("solve")
    vars: list[list[Var]] = []

    ###############
    # In our model, the ith variable represent which SR the ith block is allocated to
    ###############

    # create variables
    for brick in range(N_bricks):
        v = []
        for sr_idx in range(N_SR):
            v.append(m.addVar(vtype=GRB.BINARY))
        vars.append(v)

    # create constraints

    # Workloads around 1
    workloads = compute_workloads(vars)
    for sr_idx in range(N_SR):
        m.addConstr(LOWER_WORKLOAD <= workloads[sr_idx])
        m.addConstr(workloads[sr_idx] <= UPPER_WORKLOAD)

    # 1 SR per brick
    for brick in range(N_bricks):
        sum = LinExpr()
        for sr_idx in range(N_SR):
            sum += vars[brick][sr_idx]
        m.addConstr(sum == 1)

    # Disruption
    m.setObjective(compute_disruption(vars), GRB.MINIMIZE)
    m.optimize()
    print("Disruption: ", m.objVal)
    # Best disruption : 0.3391

    # Distance
    m.setObjective(compute_distances(vars), GRB.MINIMIZE)
    m.optimize()
    print("Distance: ", m.objVal)
    # Best distance : 309.24

    # Multi-objective, with epsilon-constraint strategy
    # We fix the disruption, and optimize the distance
    non_dominated_solutions = compute_solutions(m, vars)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(
            [solution["objVal"] for solution in non_dominated_solutions],
            [solution["disruption"] for solution in non_dominated_solutions],
            marker="x",
        )
        plt.xlabel("Distance")
        plt.ylabel("Disruption")
        plt.title("Distance vs Disruption")
        plt.grid(True)
        plt.savefig("Non_Dominated_Solutions.png")
    print(non_dominated_solutions)
    return non_dominated_solutions


if __name__ == "__main__":
    get_non_dominated_solutions(plot=True)
