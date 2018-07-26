import csv
import numpy as np
import matplotlib.pyplot as plt

# for creating synthetic data
from regression_samples import arbitrary_function_1
from regression_samples import arbitrary_function_2
from regression_samples import sample_data
# for performing regression
from regression_models import expand_to_monomials
#from regression_models import least_squares_weights
from regression_models import construct_polynomial_approx
# for plotting results
from regression_plot import plot_function_data_and_approximation


def main():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """
    # choose number of data-points and sample a pair of vectors: the input
    # values and the corresponding target values
    N = 20
    degree=1
    true_func = arbitrary_function_1
    inputs, targets = sample_data(N, true_func, seed=29)
    # convert our inputs (we just sampled) into a matrix where each row
    # is a vector of monomials of the corresponding input
    processed_inputs = expand_to_monomials(inputs, degree)
    #
    # find the weights that fit the data in a least squares way
    weights = least_squares_weights(processed_inputs, targets)
    # use weights to create a function that takes inputs and returns predictions
    # in python, functions can be passed just like any other object
    # those who know MATLAB might call this a function handle
    linear_approx = construct_polynomial_approx(degree, weights)
    fig, ax, hs = plot_function_data_and_approximation(
        linear_approx, inputs, targets, true_func)
    #ax.legend(hs, ['true function', 'data', 'linear approx'])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("regression_linear.pdf", fmt="pdf")

    plt.show()

if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()
