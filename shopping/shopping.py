import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        data = []
        months = {
            "Jan": 0,
            "Feb": 1,
            "Mar": 2,
            "Apr": 3,
            "May": 4,
            "June": 5,
            "Jul": 6,
            "Aug": 7,
            "Sep": 8,
            "Oct": 9,
            "Nov": 10,
            "Dec": 11
        }

        for row in reader:
            # skip the first row
            # one or zero depending on visitor type
            if row[15] == 'Returning_Visitor':
                row[15] = 1
            else:
                row[15] = 0
            # weekend(one if weekend, zero if not)
            if row[16] == 'TRUE':
                row[16] = 1
            else:
                row[16] = 0
            # label(one if revenue made, zero if no revenue made)
            if row[17] == 'TRUE':
                row[17] = 1
            else:
                row[17] = 0
            # I need to do the months thing
            for key, value in months.items():
                if row[10] == f'{key}':
                    row[10] = value

            # high-key copied and pasted this from the source code
            data.append({
                "evidence": [float(cell) for cell in row[:16]],
                "label": row[17]
            })
        evidence = [row["evidence"] for row in data]
        labels = [row["label"] for row in data]
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier()
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total_positive = 0
    positive_correct = 0
    total_negative = 0
    negative_correct = 0
    for actual, predicted in zip(labels, predictions):
        # counting totals
        if predicted == 1:
            total_positive += 1
        elif predicted == 0:
            total_negative += 1
        # counting number of correct totals
        if predicted == actual and predicted == 0:
            negative_correct += 1
        if predicted == actual and predicted == 1:
            positive_correct += 1
    print(total_negative, negative_correct)

    sensitivity = positive_correct / total_positive
    specificity = negative_correct / total_negative

    return sensitivity, specificity


if __name__ == "__main__":
    main()
