import pandas as pd
import numpy as np


def preprocess_classification_dataset():
    # TRAIN DATASET
    train_df = pd.read_csv("train.csv")
    train_feat_df = train_df.iloc[:, :-1]  # grab all columns except the last one
    train_output = train_df[['output']]
    X_train = train_feat_df.values
    y_train = train_output.values

    # VAL DATASET
    val_df = pd.read_csv("val.csv")
    val_feat_df = val_df.iloc[:, :-1]  # grab all columns except the last one
    val_output = val_df[['output']]
    X_val = val_feat_df.values
    y_val = val_output.values

    # TEST DATASET
    test_df = pd.read_csv("test.csv")
    test_feat_df = test_df.iloc[:, :-1]  # grab all columns except the last one
    test_output = test_df[['output']]
    X_test = test_feat_df.values
    y_test = test_output.values

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def knn_classification(X_train, y_train, x_new, k=5):
    allDiffs = []
    votes = []
    for i in range(len(X_train)):
        diff = (X_train[i] - x_new) ** 2  # find the distance between each feature value for X_train and x_new
        ecludian = 0
        for item in diff:  # find the sum of squares and take the square root
            ecludian += item
        ecludian = ecludian ** (1 / 2)
        allDiffs.append([ecludian, i])

    allDiffs.sort()
    allDiffs = allDiffs[0:k]  # take the first k smallest diffs
    for item, index in allDiffs:
        votes.append(y_train[index])
    values, counts = np.unique(votes,
                               return_counts=True)  # returns the unique values of the array and the frequency of each value
    # print("aaaaa",values, counts)
    ind = np.argmax(counts)

    return values[ind]


def logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):
    onesWeight = np.ones((len(X_train), 1), dtype=float)
    X_trainCopy = np.array(X_train)
    X_trainWeights = np.hstack((onesWeight, X_trainCopy))

    num_of_features = len(X_train[0]) + 1
    np.random.seed(random_seed)  # for reproducibility
    weights = np.random.normal(loc=0.0, scale=1.0, size=(num_of_features, 1))

    for i in range(max_iters):
        weights = weights - alpha * X_trainWeights.T @ (sigmoid(X_trainWeights @ weights) - y_train)

    return weights


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def logistic_regression_prediction(X, weights, threshold=0.5):
    onesWeight = np.ones((len(X), 1), dtype=float)
    X_trainCopy = np.array(X)
    X_trainWeights = np.hstack((onesWeight, X_trainCopy))

    # prob = np.power(y, y) *  (np.power(1 - y, 1 - y)) # to find the probability

    y_preds = sigmoid(X_trainWeights @ weights)
    for i in range(len(y_preds)):
        if y_preds[i] < threshold:
            y_preds[i] = 0
        else:
            y_preds[i] = 1
    return y_preds


def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_classification_dataset()

    prediction1nn = []
    prediction3nn = []
    prediction5nn = []
    for valRow in X_val:
        prediction1nn.append(knn_classification(X_train, y_train, valRow, k=1))
        prediction3nn.append(knn_classification(X_train, y_train, valRow, k=3))
        prediction5nn.append(knn_classification(X_train, y_train, valRow, k=5))

    val_accuracy_list = {"1nn": 0, "3nn": 0, "5nn": 0, "logistic regression": 0}
    sum1nn = (y_val.flatten() == np.array(prediction1nn).flatten()).sum()
    val_accuracy_list["1nn"] = sum1nn / len(y_val)
    sum3nn = (y_val.flatten() == np.array(prediction3nn).flatten()).sum()
    val_accuracy_list["3nn"] = sum3nn / len(y_val)
    sum5nn = (y_val.flatten() == np.array(prediction5nn).flatten()).sum()
    val_accuracy_list["5nn"] = sum5nn / len(y_val)

    weights = logistic_regression_training(X_train, y_train, alpha, max_iters, random_seed)
    logisticPrediction = logistic_regression_prediction(X_val, weights, threshold)
    sumLR = (y_val.flatten() == np.array(logisticPrediction).flatten()).sum()
    val_accuracy_list['logistic regression'] = sumLR / len(y_val)

    best_method = max(val_accuracy_list, key=val_accuracy_list.get)

    X_train_val_merge = np.vstack([X_train, X_val])
    y_train_val_merge = np.vstack([y_train, y_val])

    test_prediction = []
    if best_method == 'logistic regression':
        weights = logistic_regression_training(X_train_val_merge, y_train_val_merge, alpha, max_iters, random_seed)
        test_prediction = logistic_regression_prediction(X_test, weights, threshold)
    else:
        for testRow in X_test:
            test_prediction.append(
                knn_classification(X_train_val_merge, y_train_val_merge, testRow, k=int(best_method[0])))

    test_accuracy = ((y_test.flatten() == np.array(test_prediction).flatten()).sum()) / len(
        y_test)

    return best_method, list(val_accuracy_list.values()), test_accuracy

