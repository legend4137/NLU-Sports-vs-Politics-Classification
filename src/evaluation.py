# ------------------------------------------------------------
# evaluation.py
# Responsible for evaluation metrics and plotting
# ------------------------------------------------------------

import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_model(model, X_train, X_test, y_train, y_test):
    
    # Train model on train and Predict results on test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    return acc, f1, cm


def plot_confusion_matrix(cm, model_name, feature_type):

    plt.figure()
    plt.imshow(cm)

    plt.title(f"{model_name} - {feature_type}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0, 1], ["Sport", "Politics"])
    plt.yticks([0, 1], ["Sport", "Politics"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center')

    plt.colorbar()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_name}_{feature_type}_cm.png")
    plt.close()
