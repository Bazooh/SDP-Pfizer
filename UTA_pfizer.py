
from parse_uta_decision_preferences import ranked_instances
import numpy as np
from gurobipy import Model, Var, LinExpr
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from solve import get_non_dominated_solutions


EPSILON = 1e-6

criterion_names = ["total_distance", "max_workload", "size_disruption"]

def convert_dictionnaries_to_array(
        reference_dict: List[Dict],
        new_instances_dict: List[Dict],
):
    reference_dict = sorted(reference_dict, key = lambda instance: instance["rank"])
    reference_instances_array = np.array(
        [
            [instance["total_distance"], instance["max_workload"], instance["size_disruption"]] for instance in reference_dict
        ]
    )
    new_instances_array = np.array(
        [
            [instance["total_distance"], instance["max_workload"], instance["size_disruption"]] for instance in new_instances_dict
        ]
    )
    return(reference_instances_array, new_instances_array)


def score_i(
    val: np.ndarray,
    i: int, 
    vars: list[list[Var]],
    n:int,
    L:list[int], 
    x_min:np.ndarray, 
    x_max: np.ndarray
) -> LinExpr:
    val_norm = L[i] * (val - x_min[i]) / (x_max[i] - x_min[i])
    k = int(val_norm)
    if k == L[i]:
        return LinExpr(vars[i][k])
    return vars[i][k] + (val_norm - k) * (vars[i][k + 1] - vars[i][k])


def score(
    x: np.ndarray,
    vars: list[list[Var]],
    n:int,
    L:list[int], 
    x_min:np.ndarray, 
    x_max: np.ndarray
) -> LinExpr:
    s = LinExpr()
    for i in range(n):
        s += score_i(val=x[i], i=i, vars=vars, L=L, n=n, x_min=x_min, x_max=x_max)
    return s

def print_results(vars: list[list[Var]], errors: list[Var], n, L):
    print("Optimal solution")
    for i in range(n):
        print(f"Criteria {i + 1}:")
        for k in range(L[i] + 1):
            print(f"  s_{i}(x_{i}_{k}) = {vars[i][k].X}")

    print("Errors:")
    for i in range(len(errors)):
        print(f"Instance {i + 1}: {errors[i].X}")


def plot_results_criterion(vars: list[list[Var]], n, L):
    plt.figure(figsize=((14,6)))
    for i in range(n):
        x_vals = np.linspace(0, 1, L[i] + 1)
        y_vals = [vars[i][k].X for k in range(L[i] + 1)]
        plt.plot(x_vals, y_vals, label=criterion_names[i])
    plt.xlabel("Normalized Value")
    plt.ylabel("Score")
    plt.title("Linear Curves for Each Criteria")
    plt.legend()
    plt.savefig("Criterion_Curves.png")


def comparision(vars: list[list[Var]], n, L, new_instances_array, x_min, x_max, plot):
    # Compute scores
    new_instances_scores = [
        score(x=x, vars=vars, L=L, n=n, x_min=x_min, x_max=x_max) for x in new_instances_array
    ]
    new_instances_scores_values = [s.getValue() for s in new_instances_scores]
    
    # Sort scores and corresponding instances
    sorted_scores_and_instances = sorted(
        enumerate(zip(new_instances_scores_values, new_instances_array)),
        key=lambda x: x[1][0],  # Sort by score value
        reverse=True  # Optional: Reverse for descending order
    )
    sorted_indices = [item[0] for item in sorted_scores_and_instances]
    sorted_scores = [item[1][0] for item in sorted_scores_and_instances]
    sorted_instances = [item[1][1] for item in sorted_scores_and_instances]
    sorted_labels = [f"I({i + 1})" for i in sorted_indices]
    
    if plot:
        # Plot
        plt.figure(figsize=(14,6))
        plt.bar(
            range(len(sorted_scores)),
            sorted_scores,
            tick_label=sorted_labels,
        )
        plt.xlabel("Instances")
        plt.ylabel("Scores")
        plt.xticks(rotation=25)
        plt.title("Scores of New Instances")
        plt.savefig("UTA_Comparision.png")
    
    # Return sorted instances
    return  [
        {
            "rank": i+1,
            "total_distance": instance[0],
            "max_workload": instance[1],
            "size_disruption": instance[2],            
        }
        for i,instance in enumerate(sorted_instances)
    ]



def main(references_dict, new_instances_dict):
    ############### Get the arrays ###############
    new_instances_dict = get_non_dominated_solutions()
    reference_instances_array, new_instances_array = convert_dictionnaries_to_array(references_dict, new_instances_dict)
    n_instances, n = reference_instances_array.shape
    L: list[int] = [3 for _ in range(n)]
    x_min: np.ndarray = np.min(np.concatenate([reference_instances_array, new_instances_array]), axis=0)
    x_max: np.ndarray = np.max(np.concatenate([reference_instances_array, new_instances_array]), axis=0)
    
    ############### Construct the Model Object ###############
    # step 1: init
    m = Model("solve")
    # step 2: define variables
    vars: list[list[Var]] = [
        [m.addVar(name=f"s_{i}(x_{i}_{k})") for k in range(L[i] + 1)] for i in range(n)
    ]
    # step 3: define constraints
    for i in range(n):
        m.addConstr(vars[i][0] == 0)

    s = LinExpr()
    for i in range(n):
        s += vars[i][-1]
    m.addConstr(s == 1)

    for i in range(n):
        for k in range(L[i]):
            m.addConstr(vars[i][k + 1] >= vars[i][k])
    
    # step 4: define objective (relative to the references_instances)
    errors = [m.addVar(name=f"error_{i}") for i in range(n_instances)]

    instances_scores = [
        score(x = x, vars = vars, L = L, n = n, x_min = x_min, x_max = x_max) + errors[i] for i, x in enumerate(reference_instances_array)
    ]
    for i in range(len(instances_scores) - 1):
        m.addConstr(instances_scores[i] <= instances_scores[i + 1] + EPSILON)

    m.setObjective(sum(e * e for e in errors))
    m.optimize()

    ############### Use the fitted vars to compare the new instances ###############
    print("-" *100)
    print("Instances de références:", len(reference_instances_array))
    print("Nouvelles instances:", len(new_instances_array))
    print("-" *100)
    print("Bounds for total_distance:",  (x_min[0], x_max[0]))
    print("Bounds for max_workload:",    (x_min[1], x_max[1]))
    print("Bounds for size_disruption:", (x_min[2], x_max[2]))
    print("-" *100)
    print("(ref) Bounds for total_distance:",  (np.min(reference_instances_array, axis=0)[0],np.max(reference_instances_array, axis=0)[0]))
    print("(ref) Bounds for max_workload:",    (np.min(reference_instances_array, axis=0)[1],np.max(reference_instances_array, axis=0)[1]))
    print("(ref) Bounds for size_disruption:", (np.min(reference_instances_array, axis=0)[2],np.max(reference_instances_array, axis=0)[2]))
    print("-" *100)
    print("(new) Bounds for total_distance:",  (np.min(new_instances_array, axis=0)[0],np.max(new_instances_array, axis=0)[0]))
    print("(new) Bounds for max_workload:",    (np.min(new_instances_array, axis=0)[1],np.max(new_instances_array, axis=0)[1]))
    print("(new) Bounds for size_disruption:", (np.min(new_instances_array, axis=0)[2],np.max(new_instances_array, axis=0)[2]))
    print("-" *100)
    plot_results_criterion(vars = vars, n=n, L=L)
    new_instances_ranked = comparision(vars = vars, n=n, L=L, new_instances_array= new_instances_array, x_max=x_max, x_min=x_min, plot=True)
    for new_instance in new_instances_ranked:
        print(new_instance)

if __name__ == "__main__":
    main(references_dict=ranked_instances, new_instances_dict={})  
