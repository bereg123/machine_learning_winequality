import csv
import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import matplotlib.pyplot as plt


# for creating synthetic data
from regression_samples import arbitrary_function_1
from regression_samples import sample_data
# for performing regression
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
from regression_models import calculate_weights_posterior
from regression_models import predictive_distribution
# for plotting results
#from regression_plot import plot_function_and_data
#from regression_plot import plot_function_data_and_approximation
#from regression_plot import plot_train_test_errors
# for evaluating fit
#from regression_train_test import train_and_test


def main():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """

    # specify the centres of the rbf basis functions
    centres = np.linspace(0,1,9)
    # the width (analogous to standard deviation) of the basis functions
    scale = 0.1
    print("centres = %r" % (centres,))
    print("scale = %r" % (scale,))
    # create the feature mapping
    feature_mapping = construct_rbf_feature_mapping(centres,scale)  
    # plot the basis functions themselves for reference
    display_basis_functions(feature_mapping)

    # sample number of data-points: inputs and targets
    N = 9
    # define the noise precision of our data
    beta = (1./0.1)**2
    inputs, targets = sample_data(
        N, arbitrary_function_1, noise=np.sqrt(1./beta), seed=37)
    # now construct the design matrix for the inputs
    designmtx = feature_mapping(inputs)
    # the number of features is the widht of this matrix
    M = designmtx.shape[1]
    # define a prior mean and covaraince matrix
    m0 = np.zeros(M)
    alpha = 100
    S0 = alpha * np.identity(M)
    # find the posterior over weights 
    mN, SN = calculate_weights_posterior(designmtx, targets, beta, m0, S0)
    # the posterior mean (also the MAP) gives the central prediction
    mean_approx = construct_feature_mapping_approx(feature_mapping, mN)
    fig, ax, lines = plot_function_data_and_approximation(
        mean_approx, inputs, targets, arbitrary_function_1)
    # now plot a number of samples from the posterior
    xs = np.linspace(0,1,101)
    print("mN = %r" % (mN,))
    for i in range(20):
        weights_sample = np.random.multivariate_normal(mN, SN)
        sample_approx = construct_feature_mapping_approx(
            feature_mapping, weights_sample)
        sample_ys = sample_approx(xs)
        line, = ax.plot(xs, sample_ys, 'm', linewidth=0.5)
    lines.append(line)
    ax.legend(lines, ['true function', 'data', 'mean approx', 'samples'])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("regression_bayesian_rbf.pdf", fmt="pdf")

    # now for the predictive distribuiton
    new_inputs = np.linspace(0,1,51)
    new_designmtx = feature_mapping(new_inputs)
    ys, sigma2Ns = predictive_distribution(new_designmtx, beta, mN, SN)
    print("(sigma2Ns**0.5).shape = %r" % ((sigma2Ns**0.5).shape,))
    print("np.sqrt(sigma2Ns).shape = %r" % (np.sqrt(sigma2Ns).shape,))
    print("ys.shape = %r" % (ys.shape,))
    fig, ax, lines = plot_function_and_data(
        inputs, targets, arbitrary_function_1)
    ax.plot(new_inputs, ys, 'r', linewidth=3)
    lower = ys-np.sqrt(sigma2Ns)
    upper = ys+np.sqrt(sigma2Ns)
    print("lower.shape = %r" % (lower.shape,))
    print("upper.shape = %r" % (upper.shape,))
    ax.fill_between(new_inputs, lower, upper, alpha=0.2, color='r')


    plt.show()

def display_basis_functions(feature_mapping):
    datamtx = np.linspace(0,1, 51)
    designmtx = feature_mapping(datamtx)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for colid in range(designmtx.shape[1]):
      ax.plot(datamtx, designmtx[:,colid])
    ax.set_xlim([0,1])
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])


if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()
