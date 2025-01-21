from gurobipy import Model, GRB, Var, LinExpr
import csv
import json
import matplotlib.pyplot as plt

epsilon = 0.001
LOWER_WORKLOAD = 0.8
UPPER_WORKLOAD = 1.2

brick_workload: list[float] = []
distance_matrix: list[list[float]] = []

# initial_repartition_idx = {
#     0: [3, 4, 5, 6, 7, 14],
#     1: [9, 10, 11, 12, 13],
#     2: [8, 15, 16, 17],
#     3: [0, 1, 2, 18, 19, 20, 21],
# }

with open("brick_rp_distances.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        distance_matrix.append(list(map(float, row[1:])))


with open("bricks_index_values.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        brick_workload.append(float(row[1]))

with open("brick_rp_affectation.json", mode="r") as file:
    initial_repartition_idx = json.load(file)

initial_repartition = {
    brick: int(sr_idx)
    for sr_idx, bricks in initial_repartition_idx.items()
    for brick in bricks
}

N_SR = len(initial_repartition_idx)
N_bricks = len(brick_workload)


def compute_distances(vars: list[list[Var]]) -> LinExpr:
    distances = LinExpr()
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            distances += vars[brick][sr_idx] * distance_matrix[brick][sr_idx]
    return distances


def compute_workloads(vars: list[list[Var]]) -> list[LinExpr]:
    workloads = [LinExpr() for _ in range(N_SR)]
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            workloads[sr_idx] += vars[brick][sr_idx] * brick_workload[brick]
    return workloads

def compute_size_disruption(vars: list[list[Var]]) -> LinExpr:
    size_disruption = LinExpr()
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            if sr_idx != initial_repartition[brick]:  # Only consider changes in assignments
                size_disruption += vars[brick][sr_idx]
                break
    return size_disruption

def compute_disruption(vars: list[list[Var]]) -> LinExpr:
    disruption = LinExpr()
    for brick in range(N_bricks):
        sr_idx = initial_repartition[brick]
        disruption += brick_workload[brick] * (1 - vars[brick][sr_idx])
    return disruption
    # print(list(brick_workload[i] * ((vars[i][j] - int(j in initial_repartition_idx[i]))**2) for i in range(N_bricks) for j in range(N_SR)))
    # return sum(brick_workload[i] * ((vars[i][j] - int(i in initial_repartition_idx[j]))**2) for i in range(N_bricks) for j in range(N_SR))


def compute_solutions(m, vars):
    best_solutions = []

    m.setObjective(compute_distances(vars), GRB.MINIMIZE)

    m.optimize()
    threshold_disruption = compute_disruption(vars).getValue() - epsilon

    while m.Status == GRB.OPTIMAL:
        best_solutions.append({
            "objVal": m.objVal, 
            "disruption": compute_disruption(vars).getValue(),
            "size_disruption": compute_size_disruption(vars).getValue(),
            "total_distance": compute_distances(vars).getValue(), 
            "max_workload": max([workload.getValue() for workload in compute_workloads(vars)])
        })

        threshold_disruption = compute_disruption(vars).getValue() - epsilon
        m.addConstr(compute_disruption(vars) <= threshold_disruption)
        m.optimize()

    return best_solutions


def get_non_dominated_solutions(plot = False):
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
    non_dominated_solutions = compute_solutions(m, vars)

    if plot: 
        plt.figure(figsize=(10, 6))
        plt.plot(
            [solution["objVal"] for solution in non_dominated_solutions],
            [solution["disruption"] for solution in non_dominated_solutions],
            marker = "x"
        )
        plt.xlabel("Distance")
        plt.ylabel("Disruption")
        plt.title("Distance vs Disruption")
        plt.grid(True)
        plt.savefig("Non_Dominated_Solutions.png")
    
    return non_dominated_solutions

if __name__ == "__main__":
    get_non_dominated_solutions(plot = False)
