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


def report_mse_results(outfile, rank, lambda_value, mse):
    print("Rank=%d, Lambda=%0.2f, MSE=%s" % (rank, lambda_value, mse))
    outfile.write("%d,%f,%f\n" % (rank, lambda_value, mse))


def evaluate_parameters(train, validation, ranks, lambda_values):
    for r in ranks:
        for l in lambda_values:
            model = ALS.train(train, r, ITERATIONS, l)
            yield {
                "rank": r,
                "lambda": l,
                "mse": evaluate_model(model, train, validation),
                "model": model
            }


# Evaluate the model on training data
def evaluate_model(model, train, validation):
    predictions = (
        model
        .predictAll(prepare_validation(validation))
        .map(lambda r: ((r[0], r[1]), r[2])))
    ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    return MSE


def evaluate_models():
    # Load and parse the data
    ratings_train_text = sc.textFile(config.TRAIN_FILE)
    ratings_train = prepare_data(ratings_train_text)

    ratings_validation_text = sc.textFile(config.VALIDATION_FILE)
    ratings_validation = prepare_data(ratings_validation_text)

    ratings_test_text = sc.textFile(config.TEST_FILE)
    ratings_test = prepare_data(ratings_validation_text)

    best_model = None
    min_mse = None
    with open(config.RESULTS_FILE, "w") as outfile:
        for result in evaluate_parameters(ratings_train, ratings_validation,
                                          RANKS, LAMBDA_VALUES):
            report_mse_results(
                outfile,
                result.get("rank"),
                result.get("lambda"),
                result.get("mse"))

            if best_model is None or result.get("mse") < min_mse:
                best_model = result.get("model")
                min_mse = result.get("mse")

    return best_model


def main():
    model = evaluate_models()
    model.save(sc, config.ALS_MODEL_FILE)

if __name__ == "__main__":
    main()
