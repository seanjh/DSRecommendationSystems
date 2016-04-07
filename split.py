#!/usr/bin/env python

import os
import sys
import config


def convert_ml_10m_line(line):
    parts = line.strip().split("::")
    return [
        int(parts[config.ML_USERID_INDEX]),
        int(parts[config.ML_MOVIEID_INDEX]),
        float(parts[config.ML_RATING_INDEX]),
        int(parts[config.ML_TIMESTAMP_INDEX])
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
    print("Loading %s" % config.ML_RATINGS)
    with open(config.ML_RATINGS) as infile:
        data = [convert_ml_10m_line(line) for line in infile]
        print("Sorting %s" % config.ML_RATINGS)
        data = sort_ml_10m(data)

    train_out = open(config.ML_RATINGS_TRAIN, 'w')
    validation_out = open(config.ML_RATINGS_VALIDATION, 'w')
    test_out = open(config.ML_RATINGS_TEST, 'w')

    print("Writing train/validation/test files")
    for index, item in enumerate(data):
        write_ml_line(index, item, train_out, validation_out, test_out)

    print("\n\nFinished writing train %s" % config.ML_RATINGS_TRAIN)
    print("Finished writing validation %s" % config.ML_RATINGS_VALIDATION)
    print("Finished writing test %s\n\n" % config.ML_RATINGS_TEST)
    train_out.close()
    validation_out.close()
    test_out.close()


if __name__ == "__main__":
    main()
