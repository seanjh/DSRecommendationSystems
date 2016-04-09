#!/usr/bin/env python

import math
import config
import configspark as spark

from pyspark.mllib.recommendation import ALS, Rating


RANKS = [10, 20, 30, 40, 50]
LAMBDA_VALUES = [0.01, 0.1, 1.0, 10.0]
ITERATIONS = 10

USERID_INDEX = config.ML_USERID_INDEX
MOVIEID_INDEX = config.ML_MOVIEID_INDEX
RATING_INDEX = config.ML_RATING_INDEX

sc = spark.SPARK_CONTEXT


def convert_to_rating(row):
    return Rating(
        int(row[USERID_INDEX]),
        int(row[MOVIEID_INDEX]),
        float(row[RATING_INDEX])
    )


def prepare_data(data):
    return (
        data
        .map(lambda row: row.strip().split("::"))
        .map(convert_to_rating)
    )


def user_product(row):
    return (row[USERID_INDEX], row[MOVIEID_INDEX])


def user_product_rating(row):
    return (user_product(row), row[RATING_INDEX])


def prepare_test(validation):
    return validation.map(lambda row: user_product(row))


def report_mse_results(outfile, rank, lambda_value, mse, rmse):
    print("\nRank=%2d, Lambda=%4.3f\n\tMSE=%0f, RMSE=%0f\n" % (
          rank, lambda_value, mse, rmse))
    outfile.write("%d,%f,%f,%f\n" % (rank, lambda_value, mse, rmse))


def evaluate_parameters(train, validation, ranks, iterations, lambda_values):
    for rank in ranks:
        for lambda_val in lambda_values:
            model = ALS.train(train, rank, iterations, lambda_val)
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
    users_products = prepare_test(validation)
    print("\nValidation U+P:\n%s" % users_products.take(10))
    users_products_ratings = validation.map(user_product_rating)
    # test_grouped = validation.map(user_product_rating)
    print("\nTest Validation U+P+R:\n%s" % users_products_ratings.take(10))

    # train_ratings = train.map(user_product_rating)
    # print("\nTrain ((user, movie), rating):\n%s" % train_ratings.take(10))

    predictions = model.predictAll(users_products).map(user_product_rating)
    print("\nPredictions U+P+R:\n%s" % predictions.take(10))

    # RDD of [((user, movie), (real_rating, predicted_rating)), ...]
    ratesAndPreds = users_products_ratings.join(predictions)
    print("\nRates and predictions joined:\n%s" % ratesAndPreds.take(10))

    mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    return mse, math.sqrt(mse)


def evaluate():
    # Load and parse the data
    ratings_train_text = sc.textFile(config.ML_RATINGS_TRAIN)
    ratings_train = prepare_data(ratings_train_text)

    ratings_validation_text = sc.textFile(config.ML_RATINGS_VALIDATION)
    ratings_validation = prepare_data(ratings_validation_text)

    # ratings_test_text = sc.textFile(config.TEST_FILE)
    # ratings_test = prepare_data(ratings_validation_text)

    min_rmse = None
    best_result = None
    with open(config.RESULTS_FILE, "w") as outfile:
        for result in evaluate_parameters(ratings_train, ratings_validation,
                                          RANKS, ITERATIONS, LAMBDA_VALUES):
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
    best_model = evaluate()
    with open(config.ALS_BEST_PARAMS_FILE, "w") as outfile:
        outfile.write("%s,%s\n" % ("rank", "lambda"))
        outfile.write("%s,%s" % (
            best_model.get("rank"), best_model.get("lambda")))


if __name__ == "__main__":
    main()
