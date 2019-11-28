import argparse
import datetime
import numpy as np
import os
import pandas as pd
import sys
import time

from sklearn import datasets
from sklearn import metrics
from sklearn import neighbors
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle


def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        "-s",
        "--seed",
        default=19034,
        help="Random seed."
    )
    parser.add_argument(
        "-od",
        "--outdir",
        default='results/'
    )
    parser.add_argument(
        "-k",
        "--k_neighbours",
        nargs="+",
        help="Values of k for k-NN."
    )

    parser.add_argument(
        "-pi",
        "--perceptron_iterations",
        default=40,
        type=int,
        help="Iterations (epochs) over the data for Perceptron learner.."
    )
    parser.add_argument(
        "-eta",
        "--perceptron_learning_rate",
        default=0.1,
        type=float,
        help="Learning rate for Perceptron learner."
    )
    parser.add_argument(
        "-cv",
        "--kfold",
        default=5,
        type=int,
        help="Specify the number of folds in a `(Stratified)KFold`."
    )

    return parser


def do_experiment(dataset, dataset_name, random_state, k_neighbours, outdir, perceptron_iterations,
                  perceptron_learning_rate,
                  k_fold):
    # Shuffle input data
    data, target = shuffle(dataset.data, dataset.target, random_state=random_state)

    # Prepare a train/test set split
    # Split 2/3 into training set and 1/3 test set
    # Use the random number generator state +1; this will influence how the data is split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=(random_state + 1))

    # Initialize result lists
    classifier_name_list = []
    accuracy_list = []
    precision_list = []
    training_time_list = []
    testing_time_list = []

    classifiers = []

    # Add k-NN classifier for each k given
    for k_neighbour in k_neighbours:
        classifier = neighbors.KNeighborsClassifier(int(k_neighbour))
        classifiers.append(classifier)
        classifier_name_list.append('k-NN (' + k_neighbour + '-NN)')

    # Add Naive Bayes classifier
    classifiers.append(GaussianNB())
    classifier_name_list.append('Naive Bayes')

    # Add Perceptron classifier with the given parameters
    classifiers.append(
        Perceptron(max_iter=perceptron_iterations, eta0=perceptron_learning_rate, random_state=random_state))
    classifier_name_list.append('Perceptron')

    # Add Decision Trees classifier
    classifiers.append(tree.DecisionTreeClassifier())
    classifier_name_list.append('Decision Trees')

    for classifier in classifiers:
        # Train the classifier
        start_train_time = time.time()
        classifier.fit(X_train, y_train)
        end_train_time = time.time()

        # Predict the test set on trained classifier
        start_test_time = time.time()
        y_test_predicted = classifier.predict(X_test)
        end_test_time = time.time()

        # Compute metrics
        accuracy = "{0:.2f}".format(metrics.accuracy_score(y_test, y_test_predicted))
        precision = "{0:.2f}".format(metrics.precision_score(y_test, y_test_predicted, average="micro"))
        training_time = "{0:.4f}".format(end_train_time - start_train_time)
        testing_time = "{0:.4f}".format(end_test_time - start_test_time)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        training_time_list.append(training_time)
        testing_time_list.append(testing_time)

    df = pd.DataFrame()
    df[dataset_name + '/2/3'] = np.array(classifier_name_list)
    df['Accuracy'] = np.array(accuracy_list)
    df['Precision'] = np.array(precision_list)
    df['Training time (s)'] = np.array(training_time_list)
    df['Testing time (s)'] = np.array(testing_time_list)

    df.to_csv(outdir + dataset_name.lower() + '_two_thirds_results.csv', index=False)

    # 5 Folds Split #
    # Re-initialize result lists
    accuracy_list = []
    precision_list = []
    training_time_list = []
    testing_time_list = []

    scoring = ['accuracy', 'precision_micro']

    for classifier in classifiers:
        scores = cross_validate(classifier, data, target, scoring=scoring, cv=k_fold)

        accuracy_list.append(
            "{:.2f} ± {:.2f}".format(np.mean(scores['test_accuracy'], axis=0), np.std(scores['test_accuracy'], axis=0)))
        precision_list.append("{:.2f} ± {:.2f}".format(np.mean(scores['test_precision_micro'], axis=0),
                                                       np.std(scores['test_precision_micro'], axis=0)))
        training_time_list.append("{:.4f} ± {:.4f}".format(np.mean(scores['fit_time'], axis=0),
                                                           np.std(scores['fit_time'], axis=0)))
        testing_time_list.append("{:.4f} ± {:.4f}".format(np.mean(scores['score_time'], axis=0),
                                                          np.std(scores['score_time'], axis=0)))

    df = pd.DataFrame()
    df[dataset_name + '/' + str(k_fold) + '-folds'] = np.array(classifier_name_list)
    df['Accuracy'] = accuracy_list
    df['Precision'] = precision_list
    df['Training time (s)'] = training_time_list
    df['Testing time (s)'] = testing_time_list

    df.to_csv(outdir + dataset_name.lower() + '_5_folds_results.csv', index=False)


def experiments(config_file):
    args = get_args_parser().parse_args(['@' + config_file])

    # Create results directory
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        print("Directory ", args.outdir, " created.")

    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + timestamp + '/'
    os.mkdir(outdir)
    print("Directory", outdir, "created.")

    do_experiment(datasets.load_iris(),
                  "Iris",
                  int(args.seed),
                  list(args.k_neighbours),
                  outdir,
                  int(args.perceptron_iterations),
                  float(args.perceptron_learning_rate),
                  int(args.kfold))

    do_experiment(datasets.load_digits(),
                  "Digits",
                  int(args.seed),
                  list(args.k_neighbours),
                  outdir,
                  int(args.perceptron_iterations),
                  float(args.perceptron_learning_rate),
                  int(args.kfold))


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
