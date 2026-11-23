"""Train an ANN sentiment classifier for Amazon reviews.

Dataset labels:
    __label__1 = negative
    __label__2 = positive

Example:
    python train.py --train-file path/to/train.ft.txt --test-file path/to/test.ft.txt
"""

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier


N_FEATURES = 2**14


def read_reviews(file_path, batch_size, max_samples=None):
    """Yield review batches without loading the full dataset into memory."""
    texts, labels = [], []
    total_read = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            match = re.match(r"^(__label__[12])\s+(.*)$", line.rstrip())
            if not match:
                continue

            label, review = match.groups()
            texts.append(review)
            labels.append(0 if label == "__label__1" else 1)
            total_read += 1

            if len(texts) >= batch_size:
                yield texts, np.asarray(labels, dtype=np.int64)
                texts, labels = [], []

            if max_samples is not None and total_read >= max_samples:
                break

    if texts:
        yield texts, np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(
        description="Train an ANN for Amazon review sentiment."
    )
    parser.add_argument("--train-file", required=True, help="Path to training data")
    parser.add_argument("--test-file", help="Optional path to test data")
    parser.add_argument(
        "--model-out",
        default="sentiment_model.joblib",
        help="Saved model path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Reviews per training batch",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[128, 64],
        help="ANN hidden layer sizes, e.g. --hidden-layers 128 64",
    )
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    vectorizer = HashingVectorizer(
        n_features=N_FEATURES,
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        dtype=np.float32,
    )

    # MLPClassifier is a feed-forward artificial neural network.
    # partial_fit trains it in small batches.
    model = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu",
        solver="adam",
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=1,
        random_state=42,
    )

    training_count = 0
    for texts, labels in read_reviews(
        train_path,
        args.batch_size,
        args.max_train_samples,
    ):
        features = vectorizer.transform(texts)
        model.partial_fit(features, labels, classes=np.array([0, 1]))
        training_count += len(labels)
        print(f"Trained on {training_count:,} reviews")

    if training_count == 0:
        raise ValueError("No valid reviews found in the training file.")

    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "n_features": N_FEATURES,
            "hidden_layer_sizes": tuple(args.hidden_layers),
            "training_examples": training_count,
        },
        output_path,
    )
    print(f"Saved ANN model to: {output_path.resolve()}")

    if args.test_file:
        test_path = Path(args.test_file)
        if not test_path.is_file():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        true_labels, predicted_labels = [], []
        for texts, labels in read_reviews(
            test_path,
            args.batch_size,
            args.max_test_samples,
        ):
            predictions = model.predict(vectorizer.transform(texts))
            true_labels.extend(labels)
            predicted_labels.extend(predictions)

        if true_labels:
            print(f"Test accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
            print(
                classification_report(
                    true_labels,
                    predicted_labels,
                    target_names=["Negative", "Positive"],
                )
            )


if __name__ == "__main__":
    main()