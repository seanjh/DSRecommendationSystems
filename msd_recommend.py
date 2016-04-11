#!/usr/bin/env python

import sys
import os
import config
import msd_parse
import evaluate

from pyspark.mllib.recommendation import MatrixFactorizationModel, ALS
from parse_args import parse_args


HELP_PROMPT = """Million Song Dataset Recommendation Engine

usage:
    ./msd_recommend.py <USER_RATINGS_FILE>
    ./msd_recommend.py -u <USER_ID>
"""

RECOMMEND_NUM = 20
sc = None  # initialized within main()


def parse_user_id(data):
    user_id = None

    for d in data:
        parsed_user_id = d[msd_parse.USERID_INDEX]
        if user_id is None:
            user_id = parsed_user_id
        elif user_id != parsed_user_id:
            raise RuntimeError("Multiple user IDs detected input file")

    return user_id


def parse_file(filename):
    if not os.path.exists(filename):
        raise RuntimeError("Could not open file %s" % filename)

    with open(filename) as infile:
        plays = [msd_parse.parse_user_input(line) for line in infile]

    user_id = parse_user_id(ratings)
    return user_id, plays


def get_training_data(sc, filename, user_id, users, songs):
    ratings_train_text = sc.textFile(config.MSD_TRAIN)
    ratings_train = ratings_train_text.map(msd_parse.parse_line)

    if filename is not None:
        print("Loading new user ratings from %s" % filename)
        # parse new ratings from file
        user_id, new_ratings = parse_file(filename)
        print("New user ratings: %s" % new_ratings)
        # add new ratings to the existing dataset
        ratings_train = sc.union([ratings_train, sc.parallelize(new_ratings)])

    print("Done getting training data")
    return user_id, msd_parse.replace_raw_ids(ratings_train, users, songs)


def user_songs_heard(ratings_train, user_id):
    return (
        ratings_train
        .filter(lambda row: row[msd_parse.USERID_INDEX] == user_id)
        .map(lambda row: row[msd_parse.SONGID_INDEX])
        .collect())


def unheard_songs(user_id, ratings_train, songs, songs_reverse_map):
    # Songs already heard by the user
    heard_song_ids = user_songs_heard(ratings_train, user_id)
    print("\nSongs heard by UserID %s:" % user_id)
    for song_id in heard_song_ids:
        original_song_id = songs_reverse_map.get(str(song_id))
        print("Song ID %s" % original_song_id)

    # filter full songs vector to those not in songs heard by user
    user_songs_unheard = [(user_id, song_id) for song_id in songs.values()
                          if song_id not in heard_song_ids]
    print("\n")

    return user_songs_unheard


def prepare_model(sc, filename, user_id, ratings_train):
    if filename is None and os.path.exists(config.MSD_MODEL):
        # load the trained model
        print("\n\nLoading existing recommendation model from %s\n\n"
              % config.MSD_MODEL)
        model = MatrixFactorizationModel.load(sc, config.MSD_MODEL)
    else:
        # train a new model
        print("\n\nRetraining recommendation model for User %s\n\n" % user_id)
        rank, lambda_val = (
            evaluate.load_best_params(config.MSD_BEST_PARAMS_FILE))
        rank, lambda_val = int(rank), float(lambda_val)
        model = ALS.trainImplicit(ratings_train, rank, evaluate.ITERATIONS,
                                  lambda_val, nonnegative=True)

    return model


def make_recommendations(user_id, model, user_songs, songs_reverse_map):
    predictions = model.predictAll(user_songs)
    top_picks = (
        predictions
        .sortBy(lambda row: row[msd_parse.PLAYCOUNT_INDEX], ascending=False)
        .take(RECOMMEND_NUM))

    print("\nNew song recommendations for User #%s:" % user_id)
    for i, result in enumerate(top_picks):
        predicted_song_id = result[msd_parse.SONGID_INDEX]
        song_id = str(predicted_song_id)
        play_count = result[msd_parse.PLAYCOUNT_INDEX]

        print("#%3d: %s, predicted play count: %d"
              % (i + 1, songs_reverse_map.get(song_id), play_count))

    print("\n")


def main():
    import configspark
    sc = configspark.SPARK_CONTEXT

    args = parse_args(HELP_PROMPT, sys.argv)
    filename = args.get("filename")
    user_id = args.get("user_id")

    # Load the Users/Songs ID maps
    full_text = sc.textFile(config.MSD_DATA)
    full_raw = full_text.map(msd_parse.parse_line)
    users, songs, songs_reverse_map = msd_parse.get_user_song_maps(full_raw)

    # Load the new ratings (if any) and replace raw IDs with int IDs
    user_id, raw_plays = get_training_data(sc, filename, user_id, users, songs)
    converted_user_id = users.get(user_id)
    ratings_train = raw_plays.map(msd_parse.rating_convert)

    user_songs_unheard = unheard_songs(converted_user_id, ratings_train,
                                       songs, songs_reverse_map)

    model = prepare_model(sc, filename, converted_user_id, ratings_train)

    make_recommendations(user_id, model, sc.parallelize(user_songs_unheard),
                         songs_reverse_map)

    sc.stop()


if __name__ == "__main__":
    main()
