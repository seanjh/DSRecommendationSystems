#!/usr/bin/env python

import sys
import os
import config
import evaluate
import ml_parse

from pyspark.mllib.recommendation import MatrixFactorizationModel, ALS


def main():
    import configspark
    sc = configspark.SPARK_CONTEXT

    print("\nLoading MovieLens test dataset\n")
    test_text = sc.textFile(config.ML_RATINGS_TEST)
    ratings_test = (
        test_text.map(ml_parse.parse_line).map(ml_parse.rating_convert))

    if os.path.exists(config.ML_MODEL):
        print("\n\nLoading existing recommendation model from %s\n\n"
              % config.ML_MODEL)
        model = MatrixFactorizationModel.load(sc, config.ML_MODEL)
    else:
        raise RuntimeError("Failed to load ALS model from %s" % config.ML_MODEL)

    mse, rmse = evaluate.evaluate_model(model, ratings_test)
    print("\nML ALS model performance: MSE=%0.3f RMSE=%0.3f\n" % (mse, rmse))


if __name__ == "__main__":
    main()
