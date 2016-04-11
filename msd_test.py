#!/usr/bin/env python

import sys
import os
import config
import evaluate
import msd_parse

from pyspark.mllib.recommendation import MatrixFactorizationModel, ALS


def main():
    import configspark
    sc = configspark.SPARK_CONTEXT

    # user/song string ID to int ID mappings
    full_text = sc.textFile(config.MSD_DATA)
    full_raw = full_text.map(msd_parse.parse_line)
    users, songs, _ = msd_parse.get_user_song_maps(full_raw)

    print("\nLoading MovieLens test dataset\n")
    test_parsed = (
        sc.textFile(config.MSD_TEST)
        .map(msd_parse.parse_line))
    test_prepped = msd_parse.replace_raw_ids(test_parsed, users, songs)
    test = test_prepped.map(msd_parse.rating_convert)

    if os.path.exists(config.MSD_MODEL):
        print("\n\nLoading existing recommendation model from %s\n\n"
              % config.MSD_MODEL)
        model = MatrixFactorizationModel.load(sc, config.MSD_MODEL)
    else:
        raise RuntimeError("Failed to load ALS model from %s"
                           % config.MSD_MODEL)

    mse, rmse = evaluate.evaluate_model(model, test)
    print("\nMSD ALS model performance: MSE=%0.3f RMSE=%0.3f\n" % (mse, rmse))


if __name__ == "__main__":
    main()
