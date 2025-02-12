import os
import math
import config
import configspark
import ml_parse
from pyspark.mllib.recommendation import ALS

RANKS = [10, 20, 30, 40, 50]
LAMBDA_VALUES = [0.01, 0.1, 1.0, 10.0]
ITERATIONS = 10

sc = configspark.SPARK_CONTEXT


def report_mse_results(outfile, rank, lambda_value, mse, rmse):
    print("\nRank=%2d, Lambda=%4.3f\n\tMSE=%0f, RMSE=%0f\n" % (
          rank, lambda_value, mse, rmse))
    outfile.write("%d,%f,%f,%f\n" % (rank, lambda_value, mse, rmse))


def evaluate_parameters(train, validation, ranks, iterations, lambda_values,
                        implicit):
    print("\n")
    if implicit:
        print("Training with implicit feedback")
        trainFunc = ALS.trainImplicit
    else:
        print("Training with explicit feedback")
        trainFunc = ALS.train
    print("\n")

    for rank in ranks:
        for lambda_val in lambda_values:
            model = trainFunc(train, rank, iterations, lambda_val,
                              nonnegative=True)

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
    users_products = validation.map(lambda row: ml_parse.user_product(row))
    users_products_ratings = validation.map(ml_parse.user_product_rating)

    predictions = (
        model
        .predictAll(users_products)
        .map(ml_parse.user_product_rating))

    # RDD of [((user, movie), (real_rating, predicted_rating)), ...]
    ratesAndPreds = users_products_ratings.join(predictions).values()
    print("\nRatings and predictions (sample 10):\n%s" % ratesAndPreds.take(10))

    mse = ratesAndPreds.map(lambda result: (result[0] - result[1]) ** 2).mean()
    return mse, math.sqrt(mse)


def evaluate(train, validation, results_filename, implicit=False):
    min_rmse = None
    best_result = None
    best_model = None
    with open(results_filename, "w") as outfile:
        # CSV header
        outfile.write("%s\n" % ",".join(["rank", "lambda", "mse", "rmse"]))

        for result in evaluate_parameters(train, validation, RANKS,
                                          ITERATIONS, LAMBDA_VALUES, implicit):
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


def load_best_params(filename):
    if not os.path.exists(filename):
        raise RuntimeError("Cannot locate best ALS parameters file %s"
                           % filename)

    with open(filename) as infile:
        lines = [line for line in infile]

    parts = lines[1].strip().split(",")
    return parts[0], parts[1]
