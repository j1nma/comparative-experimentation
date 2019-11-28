## Comparative Experimentation of Machine Learning Classifiers


This work is a experiment with a number of algorithms on several datasets.
The aim is to get a feeling of how well each of these algorithms works, 
and whether there are differences depending on the dataset.

### Datasets
* Iris (https://archive.ics.uci.edu/ml/datasets/Iris,
    for Python, http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) 
* Handwritten digits (https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits, of which
 only the test set of 1797 instances is used;
 for Python, http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

### Classifiers
* k-NN (with 3 different values for k)
* Naive Bayes
* Perceptron
* Decision Trees

For each dataset, each classifier is trained and evaluated (with parameter variations), and then evaluation metrics are
computed.

### Metrics
* Effectiveness: accuracy, precision
* Efficiency: runtime for training and testing

### Splitting technique
The holdout method with 2/3 training and the rest for testing is used once, and cross validation with 5 folds also used once.

### Results
Presented in a tabular form, with one table for each dataset and splitting combination approach. For example:

| Iris/5-folds  | Accuracy | Precision | Training time | Testing time |
|---------------|----------|-----------|---------------|--------------|
| k-NN (3-NN)   | .85      |           | 0.1 sec       |              |
| Naive Bayes   |          |           |               |              |
| Decision Tree |          |           |               |              |

### Description and analysis

#### Which classifiers work best?

#### Are there differences between the datasets?

#### Are the differences in the efficiency measurements?

#### How is the runtime changing with the different data sets?

Your submission shall contain

The textual report
All code samples and
All data sets (if not already included in your software package, e.g. Python)