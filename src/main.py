from sklearn.model_selection import train_test_split
from data_loader import load_data
from features import get_features
from models import get_models
from evaluation import evaluate_model, plot_confusion_matrix

def main():
    # Load and preprocess dataset
    # X -> Text documents
    # y -> Binary Label (0: Sports & 1: Politics)
    X, y = load_data()

    # Split into training and testing (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y       # to preserve class distribution in the split
    )

    feature_types = ["bow", "tfidf", "tfidf_bigram"]
    models = get_models()

    for feature in feature_types:

        # Convert text to numerical features
        X_train_vec, X_test_vec = get_features(X_train, X_test, feature)

        for model_name, model in models.items():

            # Train model on train and Predict results on test
            acc, f1, cm = evaluate_model(model, X_train_vec, X_test_vec, y_train, y_test)

            plot_confusion_matrix(cm, model_name, feature)

            print(f"\n{model_name} | {feature}")
            print(f"Accuracy : {acc:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print("-" * 40)

if __name__ == "__main__":
    main()