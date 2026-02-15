import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

def load_data():
    # Selecting only sport and politics related categories (out of the 20 topics od data present in the dataset)
    categories = [
        'rec.sport.baseball',
        'rec.sport.hockey',
        'talk.politics.misc',
        'talk.politics.guns',
        'talk.politics.mideast'
    ]

    # Fetch desired dataset (The 20 newsgroups text dataset)
    dataset = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )

    X = dataset.data
    y = dataset.target

    # In this filtered dataset:
    # Index 0,1 correspond to sports
    # Remaining correspond to politics
    sport_indices = [0, 1]

    # COnverting to binary labels
    y_binary = np.array([0 if label in sport_indices else 1 for label in y])

    print("Class Distribution:", Counter(y_binary))
    return X, y_binary