from gurobipy import Model, GRB, Var, LinExpr
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

with open("brick_rp_distances10-100.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        distance_rp_to_brick.append(list(map(float, row[1:])))


with open("bricks_index_values10-100.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        brick_workload.append(float(row[1]))

with open("brick_rp_affectation10-100.json", mode="r") as file:
    initial_repartition_idx = json.load(file)

initial_repartition = {
    brick: int(sr_idx)
    for sr_idx, bricks in initial_repartition_idx.items()
    for brick in bricks
}

N_SR = len(initial_repartition_idx) + 1
N_bricks = len(brick_workload)


def compute_distances(vars: list[list[Var]]) -> LinExpr:
    distances = LinExpr()
    for sr_idx in range(N_SR - 1):
        for brick in range(N_bricks):
            distances += vars[brick][sr_idx] * distance_rp_to_brick[brick][sr_idx]
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
    # print(list(brick_workload[i] * ((vars[i][j] - int(j in initial_repartition_idx[i]))**2) for i in range(N_bricks) for j in range(N_SR)))
    # return sum(brick_workload[i] * ((vars[i][j] - int(i in initial_repartition_idx[j]))**2) for i in range(N_bricks) for j in range(N_SR))


def compute_solutions(m: Model, vars: list[list[Var]]) -> list:
    best_solutions = []

    m.setObjective(compute_distances(vars), GRB.MINIMIZE)

    m.optimize()
    threshold_disruption = compute_disruption(vars).getValue() - epsilon

    while m.Status == GRB.OPTIMAL:
        best_solutions.append((m.objVal, compute_disruption(vars).getValue()))

        threshold_disruption = compute_disruption(vars).getValue() - epsilon
        m.addConstr(compute_disruption(vars) <= threshold_disruption)
        m.optimize()

    return best_solutions


def main():
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
    workloads = compute_workloads(vars)
    for sr_idx in range(N_SR):
        m.addConstr(LOWER_WORKLOAD <= workloads[sr_idx])
        m.addConstr(workloads[sr_idx] <= UPPER_WORKLOAD)
    for brick in range(N_bricks):
        sum = LinExpr()
        for sr_idx in range(N_SR):
            sum += vars[brick][sr_idx]
        m.addConstr(sum == 1)

    # Disruption
    # m.setObjective(compute_disruption(vars), GRB.MINIMIZE)
    # m.optimize()
    # print("Disruption: ", m.objVal)
    # Best disruption : 0.1696

    # Distance
    # m.setObjective(compute_distances(vars), GRB.MINIMIZE)
    # m.optimize()
    # print("Distance: ", m.objVal)
    # Best distance : 154.62

    # Multi-objective, with epsilon-constraint strategy
    # We fix the disruption, and optimize the distance
    best_solutions = compute_solutions(m, vars)

    # print("number of solutions:", len(best_solutions))

    plt.figure(figsize=(10, 6))
    plt.scatter(
        [solution[0] for solution in best_solutions],
        [solution[1] for solution in best_solutions],
    )
    plt.xlabel("Distance")
    plt.ylabel("Disruption")
    plt.title("Distance vs Disruption")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
