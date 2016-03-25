#!/usr/bin/env python

import config

from pyspark.mllib.recommendation import ALS, Rating


RANKS = [10, 20, 30, 40, 50]
LAMBDA_VALUES = [0.01, 0.1, 1.0, 10.0]
ITERATIONS = 10

sc = config.SPARK_CONTEXT


def prepare_data(data):
    return (
        data
        .map(lambda l: l.split(','))
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    )


def prepare_validation(validation):
    return validation.map(lambda p: (p[0], p[1]))


def report_mse_results(rank, lambda_value, mse):
    print("Rank=%d, Lambda=%0.2f, MSE=%s" % (rank, lambda_value, mse))


def evaluate_parameters(train, validation, ranks, lambda_values):
    for r in ranks:
        for l in lambda_values:
            mse = train_evaluate_als(train, validation, r, ITERATIONS, l)
            report_mse_results(r, l, mse)


# Evaluate the model on training data
def train_evaluate_als(train, validation, rank, iterations_num, lambda_val):
    model = ALS.train(train, rank, iterations_num, lambda_val)
    predictions = model.predictAll(prepare_validation(validation)).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    return MSE


def main():
    # Load and parse the data
    ratings_train_text = sc.textFile(config.TRAIN_FILE)
    ratings_train = prepare_data(ratings_train_text)

    ratings_validation_text = sc.textFile(config.VALIDATION_FILE)
    ratings_validation = prepare_data(ratings_validation_text)

    ratings_test_text = sc.textFile(config.TEST_FILE)
    ratings_test = prepare_data(ratings_validation_text)

    evaluate_parameters(ratings_train, ratings_validation, RANKS, LAMBDA_VALUES)


if __name__ == "__main__":
    main()