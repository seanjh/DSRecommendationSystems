#!/usr/bin/env python

import sys
import os
import json
import config
import msd_parse

from pyspark.mllib.recommendation import MatrixFactorizationModel


def load_id_mappings(train):
    users_map, songs_map = None, None
    if not os.path.exists(config.MSD_USERID_MAP):
        print("Generating new user ID map")
        users_map = msd_parse.make_users_map(train)
    else:
        print("Loading user ID map from %s" % config.MSD_USERID_MAP)
        with open(config.MSD_USERID_MAP) as infile:
            users_map = json.load(infile)

    if not os.path.exists(config.MSD_SONGID_MAP):
        print("Generating new song ID map")
        songs_map = msd_parse.make_songs_map(train)
    else:
        print("Loading song ID map from %s" % config.MSD_SONGID_MAP)
        with open(config.MSD_SONGID_MAP) as infile:
            songs_map = json.load(infile)

    return users_map, songs_map


def main():
    import configspark
    sc = configspark.SPARK_CONTEXT

    train_text = sc.textFile(config.MSD_TRAIN)
    train_raw = train_text.map(msd_parse.parse_line)

    users, songs = load_id_mappings(train_raw)

    train = msd_parse.replace_raw_ids(train_raw, users, songs)


if __name__ == "__main__":
    main()
