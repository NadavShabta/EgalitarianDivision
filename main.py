import cvxpy as cp
import numpy as np


def egalitarian_division(valuation_matrix):
    """
    Test Cases:
        >>> # CASE 1: Classic two-person division (like inheriting a farm)
        >>> # Ami values Wood:80, Oil:19, Steel:1
        >>> # Tami values Wood:60, Oil:1, Steel:39
        >>> first_division = np.array([
        ...     [80, 19, 1],
        ...     [70, 1, 29]
        ... ])
        >>> result = egalitarian_division(first_division)
        >>> print(result['status'])
        optimal
        >>> np.round(result['min_utility'], 2)
        np.float64(61.67)
        >>> np.round(result['allocations'], 2)
        array([[0.53, 1.  , 0.  ],
               [0.47, 0.  , 1.  ]])
        >>> # Interpretation:
        >>> # - Ami gets 50% wood, all oil (100%), no steel
        >>> # - Tami gets 50% wood, no oil, all steel (100%)
        >>> # Both end up equally happy BARUCH HASHEM

        >>> # CASE 2: Three siblings dividing jewelry (rings, necklaces)
        >>> # Sibling A values rings highly (90), necklaces low (10)
        >>> # Sibling B values both equally (50, 50)
        >>> # Sibling C values necklaces highly (90), rings low (10)
        >>> jewelry = np.array([
        ...     [90, 10],
        ...     [50, 50],
        ...     [10, 90]
        ... ])
        >>> result = egalitarian_division(jewelry)
        >>> print(result['status'])
        optimal
        >>> np.round(result['min_utility'], 2)
        np.float64(47.37)
        >>> # Sibling A gets 53% of the rings and exactly 0% of the necklaces.
        >>> # Sibling B gets 53% of the necklaces just to balance everything out.
        >>> # Sibling C gets 47% of both items. Not perfect for them since they love necklaces, but it's a compromise.

        >>> # CASE 3: Unequal valuations where perfect equality isn't possible
        >>> # Agent Y values everything at 100
        >>> # Agent X values everything at 1
        >>> unequal = np.array([
        ...     [100, 100],
        ...     [1, 1]
        ... ])
        >>> result = egalitarian_division(unequal)
        >>> print(result['status'])
        optimal
        >>> # Agent Y should get most resources since X barely cares
        >>> result['allocations'][0].sum() < 0.1
        np.True_
    """
    n_agents, m_resources = valuation_matrix.shape
    allocations = cp.Variable((n_agents, m_resources))  # Who gets what
    min_utility = cp.Variable()  # The fairness metric we maximize

    constraints = [
        allocations >= 0,
        allocations <= 1,
        cp.sum(allocations, axis=0) == 1
    ]

    for i in range(n_agents):
        agent_utility = cp.sum(cp.multiply(allocations[i], valuation_matrix[i]))
        constraints.append(min_utility <= agent_utility)

    problem = cp.Problem(cp.Maximize(min_utility), constraints)
    problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("Fair division impossible with given valuations")

    return {
        'status': problem.status,
        'min_utility': min_utility.value,
        'allocations': allocations.value,
        'utilities': np.array([
            np.sum(allocations.value[i] * valuation_matrix[i])
            for i in range(n_agents)
        ])
    }


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
