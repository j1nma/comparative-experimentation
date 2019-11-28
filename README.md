## Comparative Experimentation of Machine Learning Classifiers

### Installation
```sh
$  pip3 install -r requirements.txt
```

### Running
```sh
$  python3 experiments.py ./configs/config.txt
```

A _results_ folder will contain a timestamp directory with the latest results.

This work is a experiment with a number of algorithms on several datasets.
The aim is to get a feeling of how well each of these algorithms works, 
and whether there are differences depending on the dataset.

### Datasets
* Iris (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) 
* Handwritten digits (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

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

#### How is the runtime changing with the different datasets?