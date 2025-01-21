import os, csv

ranked_instances = []
with open("uta_decision_preferences.csv", mode="r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        ranked_instances.append(
            {   
                "rank": int(row[0]),
                "total_distance": float(row[1]),
                "max_workload": float(row[2]),
                "disruption": int(row[3])
            }
        )
print(ranked_instances)