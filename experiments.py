import argparse
import csv
import datetime
import glob
import os
import sys
import time
from collections import deque

import librosa as librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyprind
import scipy.stats.stats as st
import sklearn
import seaborn as sns
from IPython.core.pylabtools import figsize
from sklearn import preprocessing, datasets
from sklearn import metrics
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle

congressional_voting = {"dataset_name": "Congressional_Voting",
                        "label_column": 'class',
                        "label_column_yes": 'republican',
                        "label_column_no": 'democrat'
                        }

amazon_reviews = {"dataset_name": "Amazon_Reviews",
                  "label_column": 'Class',
                  }


def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot
    Returns
    =======
    Quick Stats of the data and also the count plot

    Source:
    https://towardsdatascience.com/a-starter-pack-to-exploratory-data-analysis-with-python-pandas-seaborn-and-scikit-learn-a77889485baf#249d
    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('=' * 80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    plt.show()


def extract_music_data(music_path):
    # Find all songs in that folder
    cwd = os.getcwd()
    os.chdir(music_path)
    fileNames = glob.glob("*/*.mp3")
    numberOfFiles = len(fileNames)
    targetLabels = []

    print("Found " + str(numberOfFiles) + " files\n")

    # The first step - create the ground truth (label assignment, target, ...)
    # For that, iterate over the files, and obtain the class label for each file
    # Basically, the class name is in the full path name, so we simply use that
    for fileName in fileNames:
        pathSepIndex = fileName.index("/")
        targetLabels.append(fileName[:pathSepIndex])

    # sk-learn can only handle labels in numeric format - we have them as strings though...
    # Thus we use the LabelEncoder, which does a mapping to Integer numbers
    le = preprocessing.LabelEncoder()
    le.fit(targetLabels)  # this basically finds all unique class names, and assigns them to the numbers
    print("Found the following classes: " + str(list(le.classes_)))

    # now we transform our labels to integers
    target = le.transform(targetLabels)
    print("Transformed labels (first elements: " + str(target[0:150]))

    # If we want to find again the label for an integer value, we can do something like this:
    # print list(le.inverse_transform([0, 18, 1]))

    print("... done label encoding")

    # This is a helper function that computes the differences between adjacent array values
    def differences(seq):
        iterable = iter(seq)
        prev = next(iterable)
        for element in iterable:
            yield element - prev
            prev = element

    # This is a helper function that computes various statistical moments over a series of values, including mean, median, var, min, max, skewness and kurtosis (a total of 7 values)
    def statistics(numericList):
        return [np.mean(numericList), np.median(numericList), np.var(numericList), np.float64(st.skew(numericList)),
                np.float64(st.kurtosis(numericList)), np.min(numericList), np.max(numericList)]

    print("Extracting features using librosa" + " (" + str(datetime.datetime.now()) + ")")

    # compute some features based on BPMs, MFCCs, Chroma
    data_bpm = []
    data_bpm_statistics = []
    data_mfcc = []
    data_chroma = []

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # This takes a bit, so let's show it with a progress bar
    bar = pyprind.ProgBar(len(fileNames), stream=sys.stdout)
    for indexSample, fileName in enumerate(fileNames):
        # Load the audio as a waveform `y`, store the sampling rate as `sr`
        y, sr = librosa.load(fileName)

        # run the default beat tracker
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # from this, we simply use the tempo as BPM feature
        data_bpm.append([tempo])

        # Then we compute a few statistics on the beat timings
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        # from the timings, compute the time differences between the beats
        beat_intervals = np.array(deque(differences(beat_times)))

        # And from this, take some statistics
        # There might be a few files where the beat timings are not determined properly; we ignore them, resp. give them 0 values
        if len(beat_intervals) < 1:
            print("Errors with beat interval in file " + fileName + ", index " + str(
                indexSample) + ", using 0 values instead")
            data_bpm_statistics.append([tempo, 0, 0, 0, 0, 0, 0, 0])
        else:
            bpm_statisticsVector = []
            bpm_statisticsVector.append(tempo)  # we also include the raw value of tempo
            for stat in statistics(
                    beat_intervals):  # in case the timings are ok, we actually compute the statistics
                bpm_statisticsVector.append(stat)  # and append it to the vector, which finally has 1 + 7 features
            data_bpm_statistics.append(bpm_statisticsVector)

        # Next feature are MFCCs; we take 12 coefficients; for each coefficient, we have around 40 values per second
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        mfccVector = []
        for mfccCoefficient in mfccs:  # we transform this time series by taking again statistics over the values
            mfccVector.append(statistics(mfccCoefficient))

        # Finally, this vector should have 12 * 7 features
        data_mfcc.append(np.array(mfccVector).flatten())

        # Last feature set - chroma (which is roughly similar to actual notes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chromaVector = []
        for chr in chroma:  # similar to before, we get a number of time-series
            chromaVector.append(statistics(chr))  # and we resolve that by taking statistics over the time series

        # Finally, this vector should be be 12 * 7 features
        data_chroma.append(np.array(chromaVector).flatten())

        bar.update(indexSample)

    print(".... done" + " (" + str(datetime.datetime.now()) + ")")

    # Restore original working directory
    os.chdir(cwd)

    # These are our feature sets; we will use each of them individually to train classifiers
    return [Dataset(np.array(data_bpm), target, "data_bpm"),
            Dataset(np.array(data_bpm_statistics), target, "data_bpm_statistics"),
            Dataset(np.array(data_chroma), target, "data_chroma"),
            Dataset(np.array(data_mfcc), target, "data_mfcc")]


def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        "-n",
        "--name",
        default="Congress",
        help="Name of the Kaggle competition to run: Congress, Amazon."
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=1910299034,
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
        "-nt",
        "--n_trees",
        nargs="+",
        help="Values of number of tress for Random Forests."
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
    parser.add_argument(  # TODO: control min_samples_leaf?
        "-mx",
        "--max_depth",
        default=5,
        type=int,
        help="Max depth of Pruned Decision Tree Classifier."
    )
    parser.add_argument(
        "-mp",
        "--musicPath",
        default='./data/GTZANmp3_22khz/'
    )
    parser.add_argument(
        "-dt",
        "--dataTrain",
        default='./data/mse-3-bb-ds-ws19-congressional-voting/CongressionalVotingID.shuf.train.csv',
        help="Kaggle competition training CSV file."
    )
    parser.add_argument(
        "-ds",
        "--dataTest",
        default='./data/mse-3-bb-ds-ws19-congressional-voting/CongressionalVotingID.shuf.test.csv',
        help="Kaggle competition testing CSV file."
    )
    parser.add_argument(
        "-t",
        "--test",
        default='True',
        help="Whether to test for Kaggle competition."
    )

    return parser


class Dataset:
    def __init__(self, data, target, name):
        self.data = data
        self.target = target
        self.name = name


def do_experiment(dataset, dataset_name, random_state, k_neighbours, n_trees, outdir, perceptron_iterations,
                  perceptron_learning_rate, k_fold, max_depth, test_features=None, label_decoder=None,
                  do_k_fold=False, test_size=0.33):
    # Shuffle input data
    data, target = shuffle(dataset.data, dataset.target, random_state=random_state)

    if test_features is not None:
        X_train, X_test, y_train = data, test_features, target
    else:
        # Prepare a train/test set split
        # Split 2/3 into training set and 1/3 test set
        # Use the random number generator state +1; this will influence how the data is split
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size,
                                                            random_state=(random_state + 1))

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

    # Add Full Decision Tree classifier
    classifiers.append(tree.DecisionTreeClassifier(random_state=random_state))
    classifier_name_list.append('Unpruned DT')

    # Add Pruned/Pre-pruned Decision Tree classifier
    classifiers.append(tree.DecisionTreeClassifier(max_depth=max_depth, random_state=random_state))
    classifier_name_list.append('Pruned DT (' + str(max_depth) + ')')

    # Add Random Forests classifier for each setting
    for trees in n_trees:
        classifier = RandomForestClassifier(n_estimators=int(trees), n_jobs=-1, random_state=random_state,
                                            criterion='entropy', max_features='log2')
        classifiers.append(classifier)
        classifier_name_list.append('RF (' + str(trees) + ')')

    # Add SVC classifier
    classifiers.append(svm.SVC(random_state=random_state))
    classifier_name_list.append('SVC')

    # Add LinearSVC classifier
    if dataset_name == 'Congressional_Voting':
        classifiers.append(svm.SVC(random_state=random_state, C=0.6, gamma='scale', kernel='linear', shrinking=True))
    else:
        classifiers.append(svm.LinearSVC(random_state=random_state))
    classifier_name_list.append('LinearSVC')

    print("Training and testing " + dataset_name + " classifiers" + " (" + str(datetime.datetime.now()) + ")")
    for indexSample, classifier in enumerate(classifiers):
        print(classifier_name_list[indexSample] + ' ' + dataset_name + " (2/3) ...")

        # Train the classifier
        start_train_time = time.time()
        classifier.fit(X_train, y_train)
        end_train_time = time.time()

        # Predict the test set on trained classifier
        start_test_time = time.time()
        if test_features is not None:
            y_test_predicted = classifier.predict(X_test.iloc[:, 1:])
        else:
            y_test_predicted = classifier.predict(X_test)
        end_test_time = time.time()

        if test_features is not None:
            output = pd.DataFrame(
                data={"ID": X_test["ID"].astype(int), "\"class\"": label_decoder.inverse_transform(y_test_predicted)})
            output.to_csv(path_or_buf=outdir + classifier_name_list[indexSample].replace(' ', '_') + ".csv",
                          index=False, quoting=3,
                          sep=',')
        else:
            # Compute metrics
            accuracy = "{0:.4f}".format(metrics.accuracy_score(y_test, y_test_predicted))
            precision = "{0:.2f}".format(metrics.precision_score(y_test, y_test_predicted, average="micro"))
            training_time = "{0:.4f}".format(end_train_time - start_train_time)
            testing_time = "{0:.4f}".format(end_test_time - start_test_time)
            if dataset_name == 'Congressional_Voting':
                print(metrics.confusion_matrix(y_test, y_test_predicted))

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            training_time_list.append(training_time)
            testing_time_list.append(testing_time)

    if test_features is None:
        df = pd.DataFrame()
        df[dataset_name + '/2/3'] = np.array(classifier_name_list)
        df['Accuracy'] = np.array(accuracy_list)
        df['Precision'] = np.array(precision_list)
        df['Training time (s)'] = np.array(training_time_list)
        df['Testing time (s)'] = np.array(testing_time_list)

        df.to_csv(outdir + dataset_name.lower() + '_two_thirds_results.csv', index=False)

        if do_k_fold:
            # 5 Folds Split #
            # Re-initialize result lists
            accuracy_list = []
            precision_list = []
            training_time_list = []
            testing_time_list = []

            scoring = ['accuracy', 'precision_micro']

            for classifier in classifiers:
                if dataset_name == 'Congressional_Voting':
                    scores = cross_validate(classifier, data, target, scoring=scoring, cv=k_fold)
                else:
                    scores = cross_validate(classifier, data, target.values.ravel(), scoring=scoring, cv=k_fold)

                accuracy_list.append(
                    "{:.4f} ± {:.4f}".format(np.mean(scores['test_accuracy'], axis=0),
                                             np.std(scores['test_accuracy'], axis=0)))
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


def do_gridsearch_SVC(X_train, y_train, random_state, isAmazon=False):
    # Set the parameters by cross-validation
    print()
    print()
    print("Tuning SVC hyperparameters for accuracy...")
    print()

    if isAmazon:
        tuned_parameters = {
            'C': [5, 10, 15]
        }
        clf = GridSearchCV(LinearSVC(random_state=random_state, max_iter=10000), tuned_parameters, scoring='accuracy')
    else:
        tuned_parameters = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'C': np.arange(0.1, 1.0, 0.1)
        }
        clf = GridSearchCV(SVC(random_state=random_state), tuned_parameters, scoring='accuracy')

    clf.fit(X_train, y_train.values.ravel())

    print()

    svc_std = clf.cv_results_['std_test_score'][clf.best_index_]
    print(f'Best params: {clf.best_params_}')
    print(f'Best score: {clf.best_score_} (+/- {svc_std})')
    print(pd.DataFrame(clf.cv_results_).loc[:,
          ['mean_test_score', 'std_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score').head())


def do_gridsearch_RandomForest(X_train, y_train, random_state):
    # Set the parameters by cross-validation
    tuned_parameters = {
        'criterion': ['gini', 'entropy', 'absolute_error'],
        'n_estimators': np.arange(10, 100, 10),
        'max_features': ['auto', 'log2']
    }

    print()
    print()
    print("Tuning RandomForest hyperparameters for accuracy...")
    print()

    clf = GridSearchCV(RandomForestClassifier(random_state=random_state), tuned_parameters, scoring='accuracy')
    clf.fit(X_train, y_train)

    print()

    svc_std = clf.cv_results_['std_test_score'][clf.best_index_]
    print(f'Best params: {clf.best_params_}')
    print(f'Best score: {clf.best_score_} (+/- {svc_std})')
    print(pd.DataFrame(clf.cv_results_).loc[:,
          ['mean_test_score', 'std_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score').head())


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

    # Set data dictionary
    data_dictionary = {}
    if args.name == 'Congress':
        data_dictionary = congressional_voting
    elif args.name == 'Amazon':
        data_dictionary = amazon_reviews
    else:
        raise ("Data name not found")

    # Read train data file
    train_data_file = os.getcwd() + "/" + args.dataTrain
    if args.name == 'Congress':
        train_df = pd.read_csv(train_data_file, na_values='unknown')
    else:
        train_df = pd.read_csv(train_data_file)

    # Read train data file
    test_data_file = os.getcwd() + "/" + args.dataTest
    if args.name == 'Congress':
        test_df = pd.read_csv(test_data_file, na_values='unknown')
    else:
        test_df = pd.read_csv(test_data_file)

    # Separate train label and ID
    train_input_samples = train_df.drop([data_dictionary['label_column'], "ID"], axis=1)
    train_target = train_df.get(data_dictionary['label_column'])

    # Encode features
    train_input_samples_encoded = pd.get_dummies(train_input_samples)
    test_input_samples_encoded = pd.get_dummies(test_df)

    # Retain NaN values for KNNImputer
    for col in train_input_samples.head():
        train_input_samples_encoded.loc[
            train_input_samples[col].isnull(), train_input_samples_encoded.columns.str.startswith(col + "_")] = np.nan
    for col in test_df.head():
        test_input_samples_encoded.loc[
            test_df[col].isnull(), test_input_samples_encoded.columns.str.startswith(col + "_")] = np.nan

    # KNN train data imputation
    imp = KNNImputer(n_neighbors=10)
    imputed_train_df = pd.DataFrame(imp.fit_transform(train_input_samples_encoded))
    imputed_train_df.columns = train_input_samples_encoded.columns
    imputed_train_df.index = train_input_samples_encoded.index
    train_input_samples_encoded = imputed_train_df

    # KNN test data imputation
    imp = KNNImputer(n_neighbors=10)
    imputed_test_df = pd.DataFrame(imp.fit_transform(test_input_samples_encoded))
    imputed_test_df.columns = test_input_samples_encoded.columns
    imputed_test_df.index = test_input_samples_encoded.index
    test_input_samples_encoded = imputed_test_df

    if args.name == 'Congress':
        # Encode target
        train_target = np.array(train_target.eq(congressional_voting['label_column_yes']).mul(1))

        # Prediction label decoder
        ple = preprocessing.LabelEncoder()
        ple.fit(["republican", "democrat"])
    elif args.name == 'Amazon':
        # Encode target
        train_target = pd.DataFrame(train_df['Class'])

        # Prediction label decoder
        ple = preprocessing.LabelEncoder()
        ple.fit(train_df['Class'].unique())

    # Create dataset from features and targets
    train_dataset = Dataset(np.array(train_input_samples_encoded), train_target, data_dictionary['dataset_name'])

    if args.name == 'Congress':
        print(train_df.info())
        c_palette = ['tab:blue', 'tab:orange']
        categorical_summarized(train_df, y='class', hue='el-salvador-aid', palette=c_palette)
    # else:
    # print(train_df.describe())

    # if args.test == 'True':
    #     Perform predictions on testing set to save to CV
    # do_experiment(train_dataset,
    #               data_dictionary['dataset_name'],
    #               int(args.seed),
    #               list(args.k_neighbours),
    #               list(args.n_trees),
    #               outdir,
    #               int(args.perceptron_iterations),
    #               float(args.perceptron_learning_rate),
    #               int(args.kfold),
    #               int(args.max_depth),
    #               test_input_samples_encoded,
    #               ple)
    # else:
    # Perform training and testing split among training set
    # do_experiment(train_dataset,
    #               data_dictionary['dataset_name'],
    #               int(args.seed),
    #               list(args.k_neighbours),
    #               list(args.n_trees),
    #               outdir,
    #               int(args.perceptron_iterations),
    #               float(args.perceptron_learning_rate),
    #               int(args.kfold),
    #               int(args.max_depth),
    #               do_k_fold=True)

    # # do_experiment(datasets.load_digits(),
    # #               "Digits",
    # #               int(args.seed),
    # #               list(args.k_neighbours),
    # #               list(args.n_trees),
    # #               outdir,
    # #               int(args.perceptron_iterations),
    # #               float(args.perceptron_learning_rate),
    # #               int(args.kfold),
    # #               int(args.max_depth))
    #
    # # for dataset in extract_music_data(args.dataPath):
    # #     do_experiment(dataset,
    # #                   str(dataset.name),
    # #                   int(args.seed),
    # #                   list(args.k_neighbours),
    # #                   list(args.n_trees),
    # #                   outdir,
    # #                   int(args.perceptron_iterations),
    # #                   float(args.perceptron_learning_rate),
    # #                   int(args.kfold),
    # #                   int(args.max_depth))
    #
    # # Gridsearch among best performing models
    X_train, y_train = shuffle(train_dataset.data, train_dataset.target, random_state=int(args.seed))
    if args.name == 'Congress':
        do_gridsearch_SVC(X_train, y_train, random_state=int(args.seed), isAmazon=False)
    else:
        do_gridsearch_SVC(X_train, y_train, random_state=int(args.seed), isAmazon=True)
    # do_gridsearch_RandomForest(X_train, y_train, random_state=int(args.seed))

    # Try to delete outdir directory if empty
    try:
        os.rmdir(outdir)
    except:
        pass


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
