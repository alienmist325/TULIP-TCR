import numpy as np

gas_const = 8.3145


def get_artificial_score(
    affinity, exponential_scaling_const=1, dimension_nullifying_const=1, temperature=1
):
    """
    Taking in a binding affinity value, compute an associated score, by deriving a Gibbs Free Energy, then a probability, and finally taking the log of this value.
    """
    return -np.log(
        1
        + exponential_scaling_const
        * np.power(affinity, -dimension_nullifying_const * temperature * gas_const)
    )


def reverse_artificial_score(
    score, exponential_scaling_const=1, dimension_nullifying_const=1, temperature=1
):
    """
    Get an affinity back, given a score.
    """
    return np.power(
        (np.exp(-score) - 1) / exponential_scaling_const,
        -1 / (dimension_nullifying_const * temperature * gas_const),
    )


def solve_for_transformation(source, destination):
    A = (destination[0] - destination[1]) / (source[0] - source[1])
    B = destination[0] - A * source[0]
    return A, B


def reverse_artificial_scores(scores, *args):
    return [reverse_artificial_score(score, *args) for score in scores]


def get_artificial_scores(affinities, *args):
    return [get_artificial_score(affinity, *args) for affinity in affinities]
