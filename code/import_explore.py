import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt


def import_csv(name, delimiter, columns):
    with open(name, 'r') as file:
        data_reader = csv.reader(file, delimiter=delimiter)

        # importing the header line separately
        # and printing it to screen
        header = next(data_reader)
        # print("\n\nImporting data with fields:\n\t" + ",".join(header))

        # creating an empty list to store each row of data
        data = []

        for row in data_reader:
            # for each row of data 
            # converting each element (from string) to float type
            row_of_floats = list(map(float, row))

            # now storing in our data list
            data.append(row_of_floats)

        # print("There are %d entries." % len(data))

        # converting the data (list object) into a numpy array
        data_as_array = np.array(data)

        n = data_as_array.shape[1]
        # deleting the last column (quality) from inputs
        inputs = np.delete(data_as_array, n - 1, 1)
        # assigning it as targets instead
        targets = data_as_array[:, n - 1]

        column = 0
        while column < inputs.shape[1]:
            if column in columns:
                column += 1
            else:
                inputs = np.delete(inputs, column, 1)

        # returning this array to caller
        return header, inputs, targets


def import_pandas(name):
    with open(name, 'r') as file:
        data_frame = pd.read_csv(file, sep=';')

        return data_frame


def standardise(inputs):
    # let's inspect the data a little more
    fixed_acidity = inputs[:, 0]
    volatile_acidity = inputs[:, 1]
    citric_acid = inputs[:, 2]
    residual_sugar = inputs[:, 3]
    chlorides = inputs[:, 4]
    free_sulfur_dioxide = inputs[:, 5]
    total_sulfur_dioxide = inputs[:, 6]
    density = inputs[:, 7]
    ph = inputs[:, 8]
    sulphates = inputs[:, 9]
    alcohol = inputs[:, 10]

    print("np.mean(fixed_acidity) = %r" % (np.mean(fixed_acidity),))
    print("np.std(fixed_acidity) = %r" % (np.std(fixed_acidity),))
    print("np.mean(volatile_acidity) = %r" % (np.mean(volatile_acidity),))
    print("np.std(volatile_acidity) = %r" % (np.std(volatile_acidity),))
    print("np.mean(citric_acid) = %r" % (np.mean(citric_acid),))
    print("np.std(citric_acid) = %r" % (np.std(citric_acid),))
    print("np.mean(residual_sugar) = %r" % (np.mean(residual_sugar),))
    print("np.std(residual_sugar) = %r" % (np.std(residual_sugar),))
    print("np.mean(chlorides) = %r" % (np.mean(chlorides),))
    print("np.std(chlorides) = %r" % (np.std(chlorides),))
    print("np.mean(free_sulfur_dioxide) = %r" % (np.mean(free_sulfur_dioxide),))
    print("np.std(free_sulfur_dioxide) = %r" % (np.std(free_sulfur_dioxide),))
    print("np.mean(total_sulfur_dioxide) = %r" % (np.mean(total_sulfur_dioxide),))
    print("np.std(total_sulfur_dioxide) = %r" % (np.std(total_sulfur_dioxide),))
    print("np.mean(density) = %r" % (np.mean(density),))
    print("np.std(density) = %r" % (np.std(density),))
    print("np.mean(ph) = %r" % (np.mean(ph),))
    print("np.std(ph) = %r" % (np.std(ph),))
    print("np.mean(sulphates) = %r" % (np.mean(sulphates),))
    print("np.std(sulphates) = %r" % (np.std(sulphates),))
    print("np.mean(alcohol) = %r" % (np.mean(alcohol),))
    print("np.std(alcohol) = %r" % (np.std(alcohol),))

    # normalising inputs
    # meaning radial basis functions are more helpful
    inputs[:, 0] = (fixed_acidity - np.mean(fixed_acidity)) / np.std(fixed_acidity)
    inputs[:, 1] = (volatile_acidity - np.mean(volatile_acidity)) / np.std(volatile_acidity)
    inputs[:, 2] = (citric_acid - np.mean(citric_acid)) / np.std(citric_acid)
    inputs[:, 3] = (residual_sugar - np.mean(residual_sugar)) / np.std(residual_sugar)
    inputs[:, 4] = (chlorides - np.mean(chlorides)) / np.std(chlorides)
    inputs[:, 5] = (free_sulfur_dioxide - np.mean(free_sulfur_dioxide)) / np.std(free_sulfur_dioxide)
    inputs[:, 6] = (total_sulfur_dioxide - np.mean(total_sulfur_dioxide)) / np.std(total_sulfur_dioxide)
    inputs[:, 7] = (density - np.mean(density)) / np.std(density)
    inputs[:, 8] = (ph - np.mean(ph)) / np.std(ph)
    inputs[:, 9] = (sulphates - np.mean(sulphates)) / np.std(sulphates)
    inputs[:, 10] = (alcohol - np.mean(alcohol)) / np.std(alcohol)

    print("\n")

    return inputs


def histogram(fig, bins, header, data, column):
    # creating a single axis on the supplied figure
    ax = fig.add_subplot(2, 2, (column+1) % 5)

    # x coordinates are the specified column of the data
    xs = data[:, column]

    # plotting a histogram with a certain number of bins
    ax.hist(xs, bins)

    # setting appropriate labels
    ax.set_xlabel(header[column])
    ax.set_ylabel("bin frequency")

    # improving spacing/layout
    fig.tight_layout()


def bar_chart(data):
    series = pd.Series(data)
    value_count = series.value_counts()
    value_count = value_count.sort_index()

    return value_count.plot(kind='bar', title='Wine Quality')


def scatter_plot(fig, header, data, column1, column2):
    # creating a single axis on the supplied figure
    ax = fig.add_subplot(1, 1, 1)

    # x coordinates are the first specified column of the data
    xs = data[:, column1]

    # y coordinates are the second specified column of the data
    ys = data[:, column2]

    # plotting
    ax.plot(xs, ys, 'o', markersize=1)
    ax.set_xlabel(header[column1])
    ax.set_ylabel(header[column2])


def main(name, delimiter, columns):
    # importing using csv reader and storing as numpy array
    header, inputs, targets = import_csv(name, delimiter, columns)

    # importing using pandas and storing as data frame
    data_frame = import_pandas(name)
    print("\n")

    print(data_frame.describe())
    print("\n")

    print(data_frame.corr())

    # creating an empty figure object
    acidity_figure = plt.figure()
    histogram(acidity_figure, 20, header, inputs, 0)
    histogram(acidity_figure, 20, header, inputs, 1)
    histogram(acidity_figure, 20, header, inputs, 2)

    # saving as pdf
    acidity_figure.savefig("../plots/exploratory/acidity_histogram.png", fmt="png")

    sulfur_dioxide_figure = plt.figure()
    histogram(sulfur_dioxide_figure, 20, header, inputs, 5)
    histogram(sulfur_dioxide_figure, 20, header, inputs, 6)
    sulfur_dioxide_figure.savefig("../plots/exploratory/sulfur_dioxide_histogram.png", fmt="png")

    quality_figure = plt.figure()
    chart = bar_chart(targets)
    chart.set(xlabel="quality rating", ylabel="count")

    quality_figure.savefig("../plots/exploratory/quality_bar_chart.png", fmt="png")

    plt.show()


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
