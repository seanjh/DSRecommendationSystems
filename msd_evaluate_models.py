#!/usr/bin/env python

import sys
import os
import json
import config
import msd_parse
import ml_parse
import evaluate

from pyspark.mllib.recommendation import MatrixFactorizationModel


def clean():
    config.clean_path(config.MSD_MODEL)


def main():
    clean()

    import configspark
    sc = configspark.SPARK_CONTEXT

    # user/song string ID to int ID mappings
    full_text = sc.textFile(config.MSD_DATA)
    full_raw = full_text.map(msd_parse.parse_line)
    users, songs = msd_parse.get_user_song_maps(full_raw)

    train_parsed = (
        sc.textFile(config.MSD_VALIDATION)
        .map(msd_parse.parse_line))
    train_prepped = msd_parse.replace_raw_ids(train_parsed, users, songs)
    train = train_prepped.map(ml_parse.rating_convert)

    validation_parsed = (
        sc.textFile(config.MSD_VALIDATION)
        .map(msd_parse.parse_line))
    validation_prepped = msd_parse.replace_raw_ids(validation_parsed, users,
                                                   songs)
    validation = validation_prepped.map(ml_parse.rating_convert)

    best_result = evaluate.evaluate(train, validation, config.MSD_RESULTS_FILE,
                                    implicit=True)

    with open(config.MSD_BEST_PARAMS_FILE, "w") as outfile:
        outfile.write("%s,%s\n" % ("rank", "lambda"))
        outfile.write("%s,%s" % (
            best_result.get("rank"), best_result.get("lambda")))
    best_model = best_result.get("model")
    best_model.save(sc, config.MSD_MODEL)

    sc.stop()


if __name__ == "__main__":
    main()
