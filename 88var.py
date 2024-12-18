from gurobipy import Model, GRB, Var, LinExpr
import csv

N_bricks = 22
N_SR = 4
LOWER_WORKLOAD = 0.8
UPPER_WORKLOAD = 1.2

brick_workload: list[float] = []
distance_matrix: list[list[float]] = []

with open(
    "/Users/aymeric/Desktop/CS/SDP/SDP-Pfizer/brick_rp_distances.csv", mode="r"
) as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        distance_matrix.append(list(map(float, row[1:])))

with open(
    "/Users/aymeric/Desktop/CS/SDP/SDP-Pfizer/bricks_index_values.csv", mode="r"
) as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        brick_workload.append(float(row[1]))


def compute_distances(vars: list[list[Var]]) -> list[LinExpr]:
    distances = [LinExpr() for _ in range(N_SR)]
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            distances[sr_idx] += vars[brick][sr_idx] * distance_matrix[brick][sr_idx]
    return distances


def compute_workload(vars: list[list[Var]]) -> list[LinExpr]:
    workloads = [LinExpr() for _ in range(N_SR)]
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            workloads[sr_idx] += vars[brick][sr_idx] * brick_workload[brick]
    return workloads


def compute_disruption(vars: list[list[Var]]) -> LinExpr:
    disruption = LinExpr()
    for sr_idx in range(N_SR):
        for brick in range(N_bricks):
            disruption += (
                vars[brick][sr_idx]
                * distance_matrix[brick][sr_idx]
                * brick_workload[brick]
            )
    return disruption


def main():
    model = Model("SR_Assignment")

    # Create binary variables for each combination of bricks and SRs
    assignment_vars = []
    for i in range(N_bricks):
        temp = []
        for j in range(N_SR):
            temp.append(model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}"))
        assignment_vars.append(temp)

    model.update()

    # Add the constraint : workload must be between LOWER_WORKLOAD and UPPER_WORKLOAD


if __name__ == "__main__":
    main()
