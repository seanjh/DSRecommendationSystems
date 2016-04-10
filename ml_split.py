#!/usr/bin/env python

import os
import sys
import config
import ml_parse
from configspark import SPARK_CONTEXT as sc


TRAIN_UPPER_LIMIT = 6000000
VALIDATION_LOWER_LIMIT = 6000000
VALIDATION_UPPER_LIMIT = 8000000
TEST_LOWER_LIMIT = 8000000
RATINGS_SORTED_DATA = 0
RATINGS_SORTED_INDEX = 1


def clean():
    config.clean_path(config.ML_RATINGS_TRAIN)
    config.clean_path(config.ML_RATINGS_VALIDATION)
    config.clean_path(config.ML_RATINGS_TEST)


def row_timestamp(row):
    return row[ml_parse.TIMESTAMP_INDEX]


def train_row(row):
    return row[RATINGS_SORTED_INDEX] < TRAIN_UPPER_LIMIT


def validation_row(row):
    return (
        row[RATINGS_SORTED_INDEX] >= VALIDATION_LOWER_LIMIT and
        row[RATINGS_SORTED_INDEX] < VALIDATION_UPPER_LIMIT
    )


def test_row(row):
    return row[RATINGS_SORTED_INDEX] > TEST_LOWER_LIMIT


def drop_index(row):
    return row[RATINGS_SORTED_DATA]


def main():
    clean()

    ratings_file = sc.textFile(config.ML_RATINGS)
    ratings_full = ratings_file.map(ml_parse.parse_line)
    ratings_full_sorted = ratings_full.sortBy(row_timestamp).zipWithIndex()

    ratings_train = (
        ratings_full_sorted
        .filter(train_row)
        .map(drop_index)
        .map(ml_parse.parsed_string)
    )
    print("\nTraining data sample:\n%s" % ratings_train.take(5))
    ratings_train.saveAsTextFile(config.ML_RATINGS_TRAIN)

    ratings_validation = (
        ratings_full_sorted
        .filter(validation_row)
        .map(drop_index)
        .map(ml_parse.parsed_string)
    )
    print("\nValidation data sample:\n%s" % ratings_validation.take(5))
    ratings_validation.saveAsTextFile(config.ML_RATINGS_VALIDATION)

    ratings_test = (
        ratings_full_sorted
        .filter(test_row)
        .map(drop_index)
        .map(ml_parse.parsed_string)
    )
    print("\nTest data sample:\n%s" % ratings_test.take(5))
    ratings_test.saveAsTextFile(config.ML_RATINGS_TEST)


if __name__ == "__main__":
    main()
