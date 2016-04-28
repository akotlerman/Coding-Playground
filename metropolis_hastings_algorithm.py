"""
Metropolis Algorithms for Monte Carlo
https://www.wikiwand.com/en/Metropolis%E2%80%93Hastings_algorithm

This algorithm uses an iterative algorithms to create
desired probability distributions.
Here we try to get the Normal distribution from the uniform
distribution as the "jumping distribution".

"""

from math import sqrt, exp, pi
from random import uniform


def normpdf(x, mu=0, sigma=1):
    """
    This is the desired density to sample.
    It can be anything! The standard normal was chosen.

    Args:
        x: Input value
        mu: Gaussian (normal) mean. Default: 0
        sigma: Gaussian (normal) standard deviation. Default: 1

    Returns:
        Float

    """
    return exp(-((x - mu)**2.0)/(2 * sigma**2.0))/sqrt(2 * pi * sigma**2.0)


def metropolis_hastings(density_func, size, alpha=1, burn=1000):
    """
    General metropolis hasting algorithm to sample from desired densities.
    This function is generalized to include any density function as input.

    Assumptions:
        This is not necessarily the "best" optimized form of this algorithm.
        The jumping distribution is chosen to be uniform and set by the
        alpha parameter.

    Warnings:
        If alpha is too large, python will sometimes spit out a
        "ZeroDivisionError" exception. Try lowering the value of alpha if
        this occures. s

    Args:
        density_func: Function that takes a float and returns a float
          corresponding to the desired density to sample.
        size: Number of samples to return
        alpha: Range of "jumping distribution" used to randomly walk
          sampling candidates.
        burn: Number of samples to throw away before returning valid samples.

    Returns:
        Array of length "size" of randomly generated samples.

    """
    output = []
    x = uniform(-alpha, alpha)
    while len(output) != size:
        candidate = x + uniform(-alpha, alpha)
        try:
            probability_ratio = density_func(candidate) / density_func(x)
        except ZeroDivisionError:
            raise ZeroDivisionError('Division by zero caught. '
                                    'Alpha might be too big.')
        if uniform(0, 1) < probability_ratio:
            x = candidate
            if not burn:
                output.append(x)
            else:
                burn -= 1
    return output


norm_output = metropolis_hastings(lambda x: normpdf(x, 0, 1), size=1000, alpha=1)