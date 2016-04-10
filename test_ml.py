#!/usr/bin/env python

import sys
import os
import config
import movielens_parse as mlparse

from pyspark.mllib.recommendation import MatrixFactorizationModel, ALS
from evaluate_als_models import evaluate_model


def prepare_model(sc):
    if filename is None and os.path.exists(config.ML_MODEL):
        # load the trained model
        print("\n\nLoading existing recommendation model from %s\n\n"
              % config.ML_MODEL)
        model = MatrixFactorizationModel.load(sc, config.ML_MODEL)
    else:
        # train a new model
        print("\n\nRetraining recommendation model for User #%s\n\n" % user_id)
        rank, lambda_val = load_best_params()
        rank, lambda_val = int(rank), float(lambda_val)
        model = ALS.train(ratings_train, rank, config.ITERATIONS, lambda_val,
                          nonnegative=True)

    return model


def main():
    import configspark
    sc = configspark.SPARK_CONTEXT

    print("\nLoading MovieLens test dataset\n")
    test_text = sc.textFile(config.ML_RATINGS_TEST)
    ratings_test = (
        test_text.map(mlparse.parse_line).map(mlparse.rating_convert))

    if os.path.exists(config.ML_MODEL):
        print("\n\nLoading existing recommendation model from %s\n\n"
              % config.ML_MODEL)
        model = MatrixFactorizationModel.load(sc, config.ML_MODEL)
    else:
        raise RuntimeError("Failed to load ALS model from %s" % config.ML_MODEL)

    mse, rmse = evaluate_model(model, ratings_test)
    print("\nALS model performance: MSE=%0.3f RMSE=%0.3f\n" % (mse, rmse))


if __name__ == "__main__":
    main()
