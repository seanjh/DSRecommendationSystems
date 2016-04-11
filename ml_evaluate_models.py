#!/usr/bin/env python

import math
import config
import configspark
import ml_parse
import evaluate

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

    ratings_train.cache()
    ratings_validation.cache()

    best_result = evaluate.evaluate(ratings_train, ratings_validation,
                                    config.ML_RESULTS_FILE)
    with open(config.ML_BEST_PARAMS_FILE, "w") as outfile:
        outfile.write("%s,%s\n" % ("rank", "lambda"))
        outfile.write("%s,%s" % (
            best_result.get("rank"), best_result.get("lambda")))
    best_model = best_result.get("model")
    best_model.save(sc, config.ML_MODEL)

    sc.stop()


if __name__ == "__main__":
    main()
