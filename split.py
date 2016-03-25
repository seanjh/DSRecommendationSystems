#!/usr/bin/env python

import os
import sys

DATA_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "data"))
MOVIE_LENS_10M_PATH = os.path.join(DATA_PATH, "ml-10M100K")
MOVIE_LENS_10M_RATINGS = os.path.join(MOVIE_LENS_10M_PATH, "ratings.dat")
MOVIE_LENS_10M_RATINGS_TRAIN = os.path.join(MOVIE_LENS_10M_PATH,
                                            "ratings-train.dat")
MOVIE_LENS_10M_RATINGS_VALIDATE = os.path.join(MOVIE_LENS_10M_PATH,
                                               "ratings-test.dat")
MOVIE_LENS_10M_RATINGS_TEST = os.path.join(MOVIE_LENS_10M_PATH,
                                           "ratings-validation.dat")


ML_USERID_INDEX = 0
ML_MOVIEID_INDEX = 1
ML_TAG_INDEX = 2
ML_TIMESTAMP_INDEX = 3


def convert_ml_10m_line(line):
    parts = line.strip().split("::")
    return [
        int(parts[ML_USERID_INDEX]),
        int(parts[ML_MOVIEID_INDEX]),
        float(parts[ML_TAG_INDEX]),
        int(parts[ML_TIMESTAMP_INDEX])
    ]


def sort_ml_10m(data):
    return sorted(
        data,
        cmp=lambda x, y: cmp(x[ML_TIMESTAMP_INDEX], y[ML_TIMESTAMP_INDEX])
    )


def stringify(item):
    return ",".join([("%s" % val) for val in item]) + "\n"


def write_ml_line(index, line, train_file, validation_file, test_file):
    if index < 6000000:
        train_file.write(stringify(line))
    elif index >= 6000000 and index < 8000000:
        validation_file.write(stringify(line))
    else:
        test_file.write(stringify(line))
    sys.stdout.write("Writing line # %s\t\t\t\t\t\r" % "{:,}".format(index))
    sys.stdout.flush()


def main():
    print("Loading %s" % MOVIE_LENS_10M_RATINGS)
    with open(MOVIE_LENS_10M_RATINGS) as infile:
        data = [convert_ml_10m_line(line) for line in infile]
        print("Sorting %s" % MOVIE_LENS_10M_RATINGS)
        data = sort_ml_10m(data)

    train_out = open(MOVIE_LENS_10M_RATINGS_TRAIN, 'w')
    validation_out = open(MOVIE_LENS_10M_RATINGS_VALIDATE, 'w')
    test_out = open(MOVIE_LENS_10M_RATINGS_TEST, 'w')

    print("Writing train/validation/test files")
    for index, item in enumerate(data):
        write_ml_line(index, item, train_out, validation_out, test_out)

    print("\n\nFinished writing train %s" % MOVIE_LENS_10M_RATINGS_TRAIN)
    print("Finished writing validation %s" % MOVIE_LENS_10M_RATINGS_VALIDATE)
    print("Finished writing test %s\n\n" % MOVIE_LENS_10M_RATINGS_TEST)
    train_out.close()
    validation_out.close()
    test_out.close()


if __name__ == "__main__":
    main()
