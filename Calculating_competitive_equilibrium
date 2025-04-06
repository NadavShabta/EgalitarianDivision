import numpy as np
import cvxpy as cp


def display_results(valuation_matrix, resource_supply=None, player_budgets=None, example_title=""):
    allocation_matrix, price_vector = solve_market_equilibrium(valuation_matrix, resource_supply, player_budgets)
    total_utilities = np.sum(allocation_matrix * valuation_matrix, axis=1)
    num_players, num_items = valuation_matrix.shape

    print(f"\n--- Example Result: {example_title} ---\n")
    print("Allocation Table (Players × Resources):")
    print("           " + "  ".join([f"Resource {j+1:>2}" for j in range(num_items)]))
    for i in range(num_players):
        row = "  ".join([f"{allocation_matrix[i, j]:>5.2f}" for j in range(num_items)])
        print(f"Player {i + 1:<2}:    {row}")

    print("\nResource Prices:")
    for j in range(num_items):
        print(f"Resource {j + 1:<2}: Price = {price_vector[j]:.4f}")

    print("\nUtility per Player:")
    for i in range(num_players):
        print(f"Player {i + 1:<2}: Utility = {total_utilities[i]:.4f}")
    print("\n" + "-" * 40 + "\n")


def initialize_defaults(supply, budgets, num_players, num_items):
    if supply is None:
        supply = np.ones(num_items)
    if budgets is None:
        budgets = np.ones(num_players)
    return supply, budgets


def solve_market_equilibrium(preferences, supply=None, budgets=None):

    num_players, num_items = preferences.shape
    supply, budgets = initialize_defaults(supply, budgets, num_players, num_items)

    # Allocation variables
    allocation = cp.Variable((num_players, num_items), nonneg=True)

    # Utility = dot product of allocation and valuation
    utilities = cp.sum(cp.multiply(preferences, allocation), axis=1)

    # Objective: maximize sum of log-utilities weighted by budgets
    objective = cp.Maximize(cp.sum(cp.multiply(budgets, cp.log(utilities))))

    # Constraints: total allocation per resource cannot exceed supply
    constraints = [cp.sum(allocation, axis=0) <= supply]

    # Solve the convex optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    return allocation.value, constraints[0].dual_value


# Examples demonstrating fairness and competitive equilibrium
if __name__ == "__main__":
    default_supply = np.array([1, 1, 1], dtype=float)

    # Example 1: Players with different preferences
    values1 = np.array([
        [6, 2, 1],
        [1, 5, 5]
    ], dtype=float)
    budgets1 = np.array([50, 50], dtype=float)
    display_results(values1, default_supply, budgets1, "Different Preferences")

    # Example 2: Identical preferences, unequal budgets
    values2 = np.array([
        [3, 4, 3],
        [3, 4, 3]
    ], dtype=float)
    budgets2 = np.array([20, 80], dtype=float)
    display_results(values2, default_supply, budgets2, "Equal Preferences, Unequal Budgets")

    # Example 3: Extreme preferences
    values3 = np.array([
        [0, 0, 10],
        [10, 0, 0],
        [0, 10, 0]
    ], dtype=float)
    budgets3 = np.array([33, 33, 34], dtype=float)
    display_results(values3, default_supply, budgets3, "Extreme Preferences")

    # Example 4: Each player only values one unique resource
    values4 = np.array([
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10]
    ], dtype=float)
    budgets4 = np.array([1, 1, 1], dtype=float)
    display_results(values4, default_supply, budgets4, "Single Resource Preference")

    # Example 5: One rich player, others poor
    values5 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=float)
    budgets5 = np.array([1, 1, 98], dtype=float)
    display_results(values5, default_supply, budgets5, "One Rich Player")
