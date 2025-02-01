import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../.'))

from gurobipy import Model, GRB, Var, LinExpr
import csv
import json
import matplotlib.pyplot as plt

epsilon = 0.03
bigInt = 1000000
LOWER_WORKLOAD = 0.8
UPPER_WORKLOAD = 1.2

brick_workload: list[float] = []
distance_matrix: list[list[float]] = []

with open("data/brick_rp_distances.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        distance_matrix.append(list(map(float, row[1:])))


with open("data/bricks_index_values.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        brick_workload.append(float(row[1]))

with open("data/brick_rp_affectation.json", mode="r") as file:
    initial_repartition_idx = json.load(file)

initial_repartition = {
    brick: int(sr_idx)
    for sr_idx, bricks in initial_repartition_idx.items()
    for brick in bricks
}

N_SR = len(initial_repartition_idx)
N_bricks = len(brick_workload)


def compute_distances(fullvars: list[list[list[Var]]]) -> LinExpr:
    vars = fullvars[1]
    halfDistances = LinExpr()
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            halfDistances += vars[brick][sr_idx] * distance_matrix[brick][sr_idx]
    return 2 * halfDistances  # parce que aller-retour


def compute_workloads(fullvars: list[list[list[Var]]]) -> list[LinExpr]:
    vars = fullvars[0]
    workloads = [LinExpr() for _ in range(N_SR)]
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            workloads[sr_idx] += vars[brick][sr_idx] * brick_workload[brick]
    return workloads


def compute_disruption(fullvars: list[list[list[Var]]]) -> LinExpr:
    vars = fullvars[0]
    halfDisruption = LinExpr()
    for brick in range(N_bricks):
        sr_idx = initial_repartition[brick]
        halfDisruption += brick_workload[brick] * (1 - vars[brick][sr_idx])
    return 2 * halfDisruption
    # print(list(brick_workload[i] * ((vars[i][j] - int(j in initial_repartition_idx[i]))**2) for i in range(N_bricks) for j in range(N_SR)))
    # return sum(brick_workload[i] * ((vars[i][j] - int(i in initial_repartition_idx[j]))**2) for i in range(N_bricks) for j in range(N_SR))


def compute_solutions(m, vars):
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
    vars: list[list[list[Var]]] = []

    assignations: list[list[Var]] = []  # floats between 0 and 1

    haveToTravel: list[
        list[Var]
    ] = []  # ints : 0 or 1. must be 1 iff the corresponding var in assignations is not 0

    b: list[
        list[Var]
    ] = []  # helping to ensure the correspondence between assignations and haveToTravel

    vars = [assignations, haveToTravel, b]

    ###############
    # In our model, the ith variable represent which SR the ith block is allocated to
    ###############

    # create variables
    for brick in range(N_bricks):
        v = []
        for sr_idx in range(N_SR):
            v.append(
                m.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=1,
                    name=f"assignations{brick},{sr_idx}",
                )
            )
        vars[0].append(v)

    for brick in range(N_bricks):
        v = []
        for sr_idx in range(N_SR):
            v.append(m.addVar(vtype=GRB.BINARY, name=f"haveToTravel{brick},{sr_idx}"))
        vars[1].append(v)

    for brick in range(N_bricks):
        v = []
        for sr_idx in range(N_SR):
            v.append(m.addVar(vtype=GRB.BINARY, name=f"b{brick},{sr_idx}"))
        vars[2].append(v)

    # create constraints

    # Workloads around 1
    workloads = compute_workloads(vars)
    for sr_idx in range(N_SR):
        m.addConstr(LOWER_WORKLOAD <= workloads[sr_idx])
        m.addConstr(workloads[sr_idx] <= UPPER_WORKLOAD)

    # 1 SR equivalent per brick
    for brick in range(N_bricks):
        sum = LinExpr()
        for sr_idx in range(N_SR):
            sum += vars[0][brick][sr_idx]
        m.addConstr(sum == 1)

    # Matching between assignations and haveToTravel
    for brick in range(N_bricks):
        for sr_idx in range(N_SR):
            m.addConstr(
                vars[0][brick][sr_idx] + vars[1][brick][sr_idx]
                <= 0 + bigInt * vars[2][brick][sr_idx]
            )
            m.addConstr(
                vars[0][brick][sr_idx] + vars[1][brick][sr_idx]
                >= 1 + epsilon - bigInt * (1 - vars[2][brick][sr_idx])
            )

    # Disruption
    # m.setObjective(compute_disruption(vars), GRB.MINIMIZE)
    # m.optimize()
    # print("Disruption: ", m.objVal)
    # Best disruption : 0.3391

    # Distance
    # m.setObjective(compute_distances(vars), GRB.MINIMIZE)
    # m.optimize()
    # print("Distance: ", m.objVal)
    # Best distance : 309.24

    # Multi-objective, with epsilon-constraint strategy
    # We fix the disruption, and optimize the distance
    best_solutions = compute_solutions(m, vars)

    print("number of solutions:", len(best_solutions))

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
