#!/usr/bin/env python

import math
import config
import configspark as spark

from pyspark.mllib.recommendation import ALS, Rating


RANKS = [10, 20, 30, 40, 50]
LAMBDA_VALUES = [0.01, 0.1, 1.0, 10.0]
ITERATIONS = 10

sc = spark.SPARK_CONTEXT


def prepare_data(data):
    return (
        data
        .map(lambda l: l.split(','))
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    )


def prepare_validation(validation):
    return validation.map(lambda p: (p[0], p[1]))


def report_mse_results(outfile, rank, lambda_value, mse, rmse):
    print("Rank=%d, Lambda=%0.2f, MSE=%f, RMSE=%f" % (
          rank, lambda_value, mse, rmse))
    outfile.write("%d,%f,%f,%f\n" % (rank, lambda_value, mse, rmse))


def evaluate_parameters(train, validation, ranks, lambda_values):
    for r in ranks:
        for l in lambda_values:
            model = ALS.train(train, r, ITERATIONS, l)
            mse, rmse = evaluate_model(model, train, validation)
            yield {
                "rank": r,
                "lambda": l,
                "mse": mse,
                "rmse": rmse,
                "model": model
            }


# Evaluate the model on training data
def evaluate_model(model, train, validation):
    predictions = (
        model
        .predictAll(prepare_validation(validation))
        .map(lambda r: ((r[0], r[1]), r[2])))
    ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    return mse, math.sqrt(mse)


def evaluate_models():
    # Load and parse the data
    ratings_train_text = sc.textFile(config.TRAIN_FILE)
    ratings_train = prepare_data(ratings_train_text)

    ratings_validation_text = sc.textFile(config.VALIDATION_FILE)
    ratings_validation = prepare_data(ratings_validation_text)

    ratings_test_text = sc.textFile(config.TEST_FILE)
    ratings_test = prepare_data(ratings_validation_text)

    min_rmse = None
    best_result = None
    with open(config.RESULTS_FILE, "w") as outfile:
        for result in evaluate_parameters(ratings_train, ratings_validation,
                                          RANKS, LAMBDA_VALUES):
            report_mse_results(
                outfile,
                result.get("rank"),
                result.get("lambda"),
                result.get("mse"),
                result.get("rmse")
            )

            if best_result is None or result.get("rmse") < min_rmse:
                best_result = result
                min_rmse = result.get("rmse")

    return best_result


def main():
    best_model = evaluate_models()
    with open(config.ALS_BEST_PARAMS_FILE, "w") as outfile:
        outfile.write("%s,%s\n" % ("rank", "lambda"))
        outfile.write("%s,%s" % (best_model.get("rank"), best_model.get("lambda")))

if __name__ == "__main__":
    main()
