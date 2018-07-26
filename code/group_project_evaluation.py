import numpy as np
#import import_explore
#import regression_linear
#import regression_rbf
#import regression_rbf_cross_validation
##import regression_bayesian_rbf
#import logistic_regression
import newtest

def main(name, delimiter, columns):
    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    #import_explore.main(name, delimiter, columns)
    newtest.testError()
    #regression_linear.main(name, delimiter, columns)

    #best_scales, best_reg_params, best_no_centres = regression_rbf.main(name, delimiter, columns)  # default columns

    #regression_rbf_cross_validation.main(name, delimiter, columns)

    #regression_bayesian_rbf.main(name, delimiter, columns)
    #logistic_regression.main(name, delimiter, columns)



if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    if len(sys.argv) == 1:
        # reverting to default parameters (red wine, ; delimiter, all features)
        main('../winequality-red.csv', ";", np.arange(0, 11))
    elif len(sys.argv) == 2:
        # passing the file name as the first argument
        main(sys.argv[1], ";", np.arange(0, 11))
    elif len(sys.argv) == 3:
        # passing the delimiter as the second argument
        main(sys.argv[1], sys.argv[2], np.arange(0, 11))
    elif len(sys.argv) == 4:
        # the third argument is a list of columns to use as input features
        # list is separated by ','
        custom_columns = list(map(int, sys.argv[3].split(",")))
        main(sys.argv[1], sys.argv[2], custom_columns)
