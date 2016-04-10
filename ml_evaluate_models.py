#!/usr/bin/env python

import math
import config
import configspark
import ml_parse
import evaluate

RANKS = [10, 20, 30, 40, 50]
LAMBDA_VALUES = [0.01, 0.1, 1.0, 10.0]

sc = configspark.SPARK_CONTEXT


def clean():
    config.clean_path(config.ML_MODEL)


def main():
    clean()

    ratings_train_text = sc.textFile(config.ML_RATINGS_TRAIN)
    ratings_train = (
        ratings_train_text
        .map(ml_parse.parse_line)
        .map(ml_parse.rating_convert))

    ratings_validation_text = sc.textFile(config.ML_RATINGS_VALIDATION)
    ratings_validation = (
        ratings_validation_text
        .map(ml_parse.parse_line)
        .map(ml_parse.rating_convert))

    best_result = evaluate.evaluate(ratings_train, ratings_validation,
                                    config.ML_RESULTS_FILE)
    with open(config.ML_BEST_PARAMS_FILE, "w") as outfile:
        outfile.write("%s,%s\n" % ("rank", "lambda"))
        outfile.write("%s,%s" % (
            best_result.get("rank"), best_result.get("lambda")))
    best_model = best_result.get("model")
    best_model.save(sc, config.ML_MODEL)


if __name__ == "__main__":
    main()