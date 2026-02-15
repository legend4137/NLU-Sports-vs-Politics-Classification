from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_features(X_train, X_test, feature_type):

    if feature_type == "bow":
        # Bag of Words representation
        vectorizer = CountVectorizer(stop_words='english')
    elif feature_type == "tfidf":
        # TF-IDF representation
        vectorizer = TfidfVectorizer(stop_words='english')
    elif feature_type == "tfidf_bigram":
        # TF-IDF with unigrams + bigrams
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    else:
        raise ValueError("Invalid feature type")
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec