import csv
import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import matplotlib.pyplot as plt


# for creating synthetic data
from regression_samples import arbitrary_function_1
from regression_samples import sample_data
# for performing regression
from regression_models import expand_to_monomials
from regression_models import ml_weights
from regression_models import regularised_ml_weights
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
# for plotting results
from regression_plot import plot_function_data_and_approximation
from regression_plot import plot_train_test_errors
# for evaluating fit
from regression_train_test import train_and_test


def main():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """

    # specify the centres of the rbf basis functions
    centres = np.linspace(0,1,7)
    # the width (analogous to standard deviation) of the basis functions
    scale = 0.15
    print("centres = %r" % (centres,))
    print("scale = %r" % (scale,))
    feature_mapping = construct_rbf_feature_mapping(centres,scale)  
    datamtx = np.linspace(0,1, 51)
    designmtx = feature_mapping(datamtx)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for colid in range(designmtx.shape[1]):
      ax.plot(datamtx, designmtx[:,colid])
    ax.set_xlim([0,1])
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    # choose number of data-points and sample a pair of vectors: the input
    # values and the corresponding target values
    N = 20
    inputs, targets = sample_data(N, arbitrary_function_1, seed=37)
    # define the feature mapping for the data
    feature_mapping = construct_rbf_feature_mapping(centres,scale)  
    # now construct the design matrix
    designmtx = feature_mapping(inputs)
    #
    # find the weights that fit the data in a least squares way
    weights = ml_weights(designmtx, targets)
    # use weights to create a function that takes inputs and returns predictions
    # in python, functions can be passed just like any other object
    # those who know MATLAB might call this a function handle
    rbf_approx = construct_feature_mapping_approx(feature_mapping, weights)
    fig, ax, lines = plot_function_data_and_approximation(
        rbf_approx, inputs, targets, arbitrary_function_1)
    ax.legend(lines, ['true function', 'data', 'linear approx'])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("regression_rbf.pdf", fmt="pdf")

    # for a single choice of regularisation strength we can plot the
    # approximating function
    reg_param = 10**-3
    reg_weights = regularised_ml_weights(
        designmtx, targets, reg_param)
    rbf_reg_approx = construct_feature_mapping_approx(feature_mapping, reg_weights)
    fig, ax, lines = plot_function_data_and_approximation(
        rbf_reg_approx, inputs, targets, arbitrary_function_1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("regression_rbf_basis_functions_reg.pdf", fmt="pdf")

    # to find a good regularisation parameter, we can performa a parameter
    # search (a naive way to do this is to simply try a sequence of reasonable
    # values within a reasonable range.
    
    # sample some training and testing inputs
    train_inputs, train_targets = sample_data(N, arbitrary_function_1, seed=37)
    # we need to use a different seed for our test data, otherwise some of our
    # sampled points will be the same
    test_inputs, test_targets = sample_data(100, arbitrary_function_1, seed=82)
    # convert the raw inputs into feature vectors (construct design matrices)
    train_designmtx = feature_mapping(train_inputs)
    test_designmtx = feature_mapping(test_inputs)
    # now we're going to evaluate train and test error for a sequence of
    # potential regularisation strengths storing the results
    reg_params = np.logspace(-5,1)
    train_errors = []
    test_errors = []
    for reg_param in reg_params:
        # evaluate the test and train error for this regularisation parameter
        train_error, test_error = train_and_test(
            train_designmtx, train_targets, test_designmtx, test_targets,
            reg_param=reg_param)
        # collect the errors
        train_errors.append(train_error)
        test_errors.append(test_error)
    # plot the results
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors, test_errors)        
    ax.set_xscale('log')


    # we may also be interested in choosing the right number of centres, or
    # the right width/scale of the rbf functions.
    # Here we vary the width and evaluate the performance
    reg_param = 10**-3
    scales = np.logspace(-2,0)
    train_errors = []
    test_errors = []
    for scale in scales:
        # we must construct the feature mapping anew for each scale
        feature_mapping = construct_rbf_feature_mapping(centres,scale)  
        train_designmtx = feature_mapping(train_inputs)
        test_designmtx = feature_mapping(test_inputs)
        # evaluate the test and train error for this regularisation parameter
        train_error, test_error = train_and_test(
            train_designmtx, train_targets, test_designmtx, test_targets,
            reg_param=reg_param)
        # collect the errors
        train_errors.append(train_error)
        test_errors.append(test_error)
    # plot the results
    fig, ax = plot_train_test_errors(
        "scale", scales, train_errors, test_errors)        
    ax.set_xscale('log')

    # Here we vary the number of centres and evaluate the performance
    reg_param = 10**-3
    scale = 0.15
    n_centres_seq = np.arange(3,20)
    train_errors = []
    test_errors = []
    for n_centres in n_centres_seq:
        # we must construct the feature mapping anew for each number of centres
        centres = np.linspace(0,1,n_centres)
        feature_mapping = construct_rbf_feature_mapping(centres,scale)  
        train_designmtx = feature_mapping(train_inputs)
        test_designmtx = feature_mapping(test_inputs)
        # evaluate the test and train error for this regularisation parameter
        train_error, test_error = train_and_test(
            train_designmtx, train_targets, test_designmtx, test_targets,
            reg_param=reg_param)
        # collect the errors
        train_errors.append(train_error)
        test_errors.append(test_error)
    # plot the results
    fig, ax = plot_train_test_errors(
        "Num. Centres", n_centres_seq, train_errors, test_errors)        
    plt.show()


if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()
