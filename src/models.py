from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def get_models():

    models = {
        "Naive_Bayes": MultinomialNB(),
        "Logistic_Regression": LogisticRegression(max_iter=2000),
        "Linear_SVM": LinearSVC()
    }

    return models