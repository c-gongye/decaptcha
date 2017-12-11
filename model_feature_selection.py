import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


####################################################################################
#  changing parameters below is not recommended unless you know what you are doing #
####################################################################################
# for MLP
MAX_NUS = 110
MIN_NUS = 100
# for cross validation
CV = 5
NUM_SEEDS = 10

# for RF
MAX_TREES = 200

# for LG
MAX_C = 50

####################################################################################

class ModelSelector():
    def __init__(self, X, y, clf):
        self.X = X
        self.y = y
        self.clf = clf

    @staticmethod
    def print_scores(score_mean, score_std, p, name):
        print("Accuracy: %0.3f (+/- %0.3f). The %s is %d" % (score_mean, score_std, name, p))

    def model_score(self, p, name):
        scores = cross_val_score(self.clf, self.X, self.y, cv=CV, n_jobs=-1)
        score_mean, score_std = scores.mean(), scores.std()
        self.print_scores(score_mean, score_std, p, name)
        return scores.mean()

def test_rf(X, y, N):
    model = ModelSelector(X, y, RandomForestClassifier())
    scores = []
    indexes = []
    for i in range(MAX_TREES / 10):
        sub_scores = []
        num_trees = i * 10 + 10
        print "Number of trees: ", num_trees
        for j in range(NUM_SEEDS):
            model.clf = RandomForestClassifier(n_estimators=num_trees, random_state=j,n_jobs=N)
            sub_scores.append(model.model_score(j, "SEED"))
        scores.append(sub_scores)
        indexes.append(num_trees)
    scores = list(np.max(np.array(scores), axis=1))
    plot_scores(scores, indexes, "Accuracy", "Number of trees",
                "Accuracy", "Accuracies of different number of estimators of RF", "RF.png")


def plot_scores(scores, indexes, label, xlabel, ylabel, title, filename):
    plt.plot(indexes, scores, label=label, lw=1.5, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig("./graph/" + filename)
    plt.show()
    # plt.clf()


def test_mlp(X, y):
    # MLP needs normalized features
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    model = ModelSelector(X, y, MLPClassifier())
    scores = []
    indexes = []
    for i in range(MAX_NUS / 10):
        sub_scores = []
        num_neurons = i * 10 + 10
        print "Number of neurons: ", num_neurons
        for j in range(NUM_SEEDS):
            model.clf = MLPClassifier(hidden_layer_sizes=(num_neurons, num_neurons), random_state=j)
            sub_scores.append(model.model_score(j, "SEED"))
        scores.append(sub_scores)
        indexes.append(num_neurons)
    scores = list(np.max(np.array(scores), axis=1))
    plot_scores(scores, indexes, "Accuracy", "Number of neurons",
                "Accuracy", "Accuracies of different number of neurons of MLP", "MLP.png")


def test_rg(X, y, lift, N):
    # lifting: we are only lifting mfcc features according to test result
    poly = PolynomialFeatures(lift)
    X = poly.fit_transform(X)

    # sag needs normalized features
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    print "Caclculation scores..."
    for i in range(MAX_C/10):
        C = i * 10 + 1
        model = ModelSelector(X, y, LogisticRegression(solver="sag", n_jobs=N))
        model.model_score(C,'C')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='select classifier using K-Fold cross validation')
    parser.add_argument('--N', default=-1, type=int, help="number of cpu cores")
    parser.add_argument('--classifier', default=1, type=int, help=
    """
    1: Random Forest
    2: Logistic Regression
    3: Multi-layer Perceptron
    """)
    parser.add_argument('--lift', default=1, type=int,
                        help="degree of polynomial features, only works with logistic regression")
    args = parser.parse_args()

    # load processed data
    X = np.load('X.npy')
    y = np.load('y.npy')
    y = y[:,3]
    # test the classifier
    if args.classifier == 1:
        test_rf(X, y, args.N)
    elif args.classifier == 2:
        test_rg(X, y, args.lift, args.N)
    elif args.classifier == 3:
        test_mlp(X, y)
