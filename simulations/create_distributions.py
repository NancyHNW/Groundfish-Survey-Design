# Functions to create a set of distriutions for stochastic simulations
import numpy as np

# def create_distributions(num_distributions, distribution_type='normal', **kwargs):
#     """
#     Create a set of distributions for stochastic simulations.

#     Parameters:
#     num_distributions (int): The number of distributions to create.
#     distribution_type (str): The type of distribution to create ('normal', 'uniform', etc.).
#     **kwargs: Additional parameters for the distribution (e.g., mean and std for normal).

#     Returns:
#     list: A list of distribution objects.
#     """
#     distributions = []
    
#     for _ in range(num_distributions):
#         if distribution_type == 'normal':
#             mean = kwargs.get('mean', 0)
#             std = kwargs.get('std', 1)
#             dist = np.random.normal(mean, std)
#         elif distribution_type == 'uniform':
#             low = kwargs.get('low', 0)
#             high = kwargs.get('high', 1)
#             dist = np.random.uniform(low, high)
#         else:
#             raise ValueError(f"Unsupported distribution type: {distribution_type}")
        
#         distributions.append(dist)
    
#     return distributions

# Run code
# Example usage:

# Save distributions to a file within a folder