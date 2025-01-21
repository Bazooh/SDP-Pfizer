import csv
import json


brick_workload = []
distance_matrix = []
initial_repartition_idx = {}

with open("Pfitzer10-100.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)

    zone_pos = []
    rp_pos = []
    current_rps = []

    for row in reader:
        infos = tuple(map(float, row[1:]))

        zone_pos.append((infos[0], infos[1]))
        brick_workload.append(infos[2])

        if infos[3] == 1:
            rp_pos.append((infos[0], infos[1]))

        current_rps.append(infos[4:].index(1))

    for x, y in zone_pos:
        distance_matrix.append([])
        for rp_x, rp_y in rp_pos:
            distance_matrix[-1].append(((x - rp_x) ** 2 + (y - rp_y) ** 2) ** 0.5)

    for i, rp in enumerate(current_rps):
        if rp not in initial_repartition_idx:
            initial_repartition_idx[rp] = []
        initial_repartition_idx[rp].append(i)

with open("brick_rp_distances10-100.csv", mode="w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["brick"] + [f"rp{i + 1}" for i in list(range(len(rp_pos)))])
    for i, row in enumerate(distance_matrix):
        writer.writerow([i + 1] + row)

with open("bricks_index_values10-100.csv", mode="w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["brick", "index_value"])
    for i, row in enumerate(brick_workload):
        writer.writerow([i + 1, row])

with open("brick_rp_affectation10-100.json", mode="w") as file:
    json.dump(initial_repartition_idx, file)
