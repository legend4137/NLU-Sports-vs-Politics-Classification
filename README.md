# Sports vs Politics Text Classification

This project implements a binary text classifier that distinguishes between **Sports** and **Politics** documents using the **20 Newsgroups dataset**.

The goal is to compare multiple machine learning techniques and feature representations for document classification.

---

## Dataset

We use a subset of the 20 Newsgroups dataset:

### Sports Categories:
- rec.sport.baseball
- rec.sport.hockey

### Politics Categories:
- talk.politics.misc
- talk.politics.guns
- talk.politics.mideast

All documents are preprocessed by removing headers, footers, and quotes.

Binary Labels:
- 0 → Sport
- 1 → Politics

---

## Feature Representations

We compare three feature extraction methods:

1. **Bag of Words (BoW)**
2. **TF-IDF**
3. **TF-IDF with Bigrams**

---

## Machine Learning Models

We evaluate three classification models:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

---

## Evaluation Metrics

For each model and feature combination, we compute:

- Accuracy
- F1 Score
- Confusion Matrix

Confusion matrices are automatically saved in the `results/` folder.

---

## Project Structure

```
NLU-Sports-vs-Politics-Classification/
│
├── src/
│ ├── data_loader.py
│ ├── features.py
│ ├── models.py
│ ├── evaluation.py
│ └── main.py
│
├── results/
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run experiment:

```bash
python src/main.py
```