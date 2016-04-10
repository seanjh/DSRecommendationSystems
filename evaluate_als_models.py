#!/usr/bin/env python

import math
import config
import configspark as spark
import movielens_parse as mlparse
from pyspark.mllib.recommendation import ALS

RANKS = [10, 20, 30, 40, 50]
LAMBDA_VALUES = [0.01, 0.1, 1.0, 10.0]

sc = spark.SPARK_CONTEXT


def report_mse_results(outfile, rank, lambda_value, mse, rmse):
    print("\nRank=%2d, Lambda=%4.3f\n\tMSE=%0f, RMSE=%0f\n" % (
          rank, lambda_value, mse, rmse))
    outfile.write("%d,%f,%f,%f\n" % (rank, lambda_value, mse, rmse))


def evaluate_parameters(train, validation, ranks, iterations, lambda_values):
    for rank in ranks:
        for lambda_val in lambda_values:
            model = ALS.train(train, rank, iterations, lambda_val, nonnegative=True)
            mse, rmse = evaluate_model(model, validation)
            yield {
                "rank": rank,
                "lambda": lambda_val,
                "mse": mse,
                "rmse": rmse,
                "model": model
            }


# Evaluate the model on test data
def evaluate_model(model, validation):
    users_products = validation.map(lambda row: mlparse.user_product(row))
    users_products_ratings = validation.map(mlparse.user_product_rating)

    predictions = (
        model
        .predictAll(users_products)
        .map(mlparse.user_product_rating))

    # RDD of [((user, movie), (real_rating, predicted_rating)), ...]
    ratesAndPreds = users_products_ratings.join(predictions).values()
    print("\nRatings and predictions (sample 10):\n%s" % ratesAndPreds.take(10))

    mse = ratesAndPreds.map(lambda result: (result[0] - result[1]) ** 2).mean()
    return mse, math.sqrt(mse)


def evaluate(train, validation):
    min_rmse = None
    best_result = None
    best_model = None
    with open(config.RESULTS_FILE, "w") as outfile:
        # CSV header
        outfile.write("%s\n" % ",".join(["rank", "lambda", "mse", "rmse"]))

        for result in evaluate_parameters(train, validation, RANKS,
                                          config.ITERATIONS, LAMBDA_VALUES):
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
    ratings_train_text = sc.textFile(config.ML_RATINGS_TRAIN)
    ratings_train = (
        ratings_train_text
        .map(mlparse.parse_line)
        .map(mlparse.rating_convert))

    ratings_validation_text = sc.textFile(config.ML_RATINGS_VALIDATION)
    ratings_validation = (
        ratings_validation_text
        .map(mlparse.parse_line)
        .map(mlparse.rating_convert))

    best_result = evaluate(ratings_train, ratings_validation)
    with open(config.ALS_BEST_PARAMS_FILE, "w") as outfile:
        outfile.write("%s,%s\n" % ("rank", "lambda"))
        outfile.write("%s,%s" % (
            best_result.get("rank"), best_result.get("lambda")))
    best_model = best_result.get("model")
    best_model.save(sc, config.ML_MODEL)


if __name__ == "__main__":
    main()
