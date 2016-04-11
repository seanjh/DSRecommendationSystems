#!/usr/bin/env python

import sys
import os
import config
import ml_evaluate_models as model_evaluate
import ml_parse
import evaluate

from pyspark.mllib.recommendation import MatrixFactorizationModel, ALS
from parse_args import parse_args


RECOMMEND_NUM = 20
sc = None  # initialized within main()


HELP_PROMPT = """MovieLens Recommendation Engine

usage:
    ./ml_recommend.py <USER_RATINGS_FILE>
    ./ml_recommend.py -u <USER_ID>
"""


def movies_unwatched(user_id, train, movies):
    # Movies already seen by the user
    seen_movie_ids = user_movies_seen(train, user_id)
    print("\nMovies seen by UserID %s:" % user_id)
    for movie_id in seen_movie_ids:
        print("Movie %d: %s"
              % (movie_id, ml_parse.movie_name(movie_id, movies)))

    # filter full movies vector to those not in movies rated by user
    user_movies_unwatched = [(user_id, movie_id) for movie_id in movies.keys()
                             if int(movie_id) not in seen_movie_ids]
    print("\n")

    return user_movies_unwatched


def user_movies_seen(ratings_train, user_id):
    return (
        ratings_train
        .filter(lambda row: row[ml_parse.USERID_INDEX] == user_id)
        .map(lambda row: row[ml_parse.MOVIEID_INDEX])
        .collect())


def parse_user_id(data):
    user_id = None

    for d in data:
        parsed_user_id = d[ml_parse.USERID_INDEX]
        if user_id is None:
            user_id = parsed_user_id
        elif user_id != parsed_user_id:
            raise RuntimeError("Multiple user IDs detected input file")

    return user_id


def parse_file(filename):
    if not os.path.exists(filename):
        raise RuntimeError("Could not open file %s" % filename)

    with open(filename) as infile:
        ratings = [ml_parse.parse_user_input(line) for line in infile]

    user_id = parse_user_id(ratings)
    return user_id, ratings


def get_training_data(sc, filename, user_id):
    ratings_train_text = sc.textFile(config.ML_RATINGS_TRAIN)
    ratings_train = ratings_train_text.map(ml_parse.parse_line)

    if filename is not None:
        print("Loading new user ratings from %s" % filename)
        # parse new ratings from file
        user_id, new_ratings = parse_file(filename)
        print("New user ratings: %s" % new_ratings)
        # add new ratings to the existing dataset
        ratings_train = sc.union([ratings_train, sc.parallelize(new_ratings)])

    print("Done getting training data")
    return user_id, ratings_train


def prepare_model(sc, filename, user_id, ratings_train):
    if filename is None and os.path.exists(config.ML_MODEL):
        # load the trained model
        print("\n\nLoading existing recommendation model from %s\n\n"
              % config.ML_MODEL)
        model = MatrixFactorizationModel.load(sc, config.ML_MODEL)
    else:
        # train a new model
        print("\n\nRetraining recommendation model for User #%s\n\n" % user_id)
        rank, lambda_val = evaluate.load_best_params(config.ML_BEST_PARAMS_FILE)
        rank, lambda_val = int(rank), float(lambda_val)
        model = ALS.train(ratings_train, rank, evaluate.ITERATIONS, lambda_val,
                          nonnegative=True)

    return model


def make_recommendations(user_id, model, user_movies, movies):
    predictions = model.predictAll(user_movies)
    top_picks = (
        predictions
        .sortBy(lambda row: row[ml_parse.RATING_INDEX], ascending=False)
        .take(RECOMMEND_NUM))
    print("\nNew movie recommendations for User #%s:" % user_id)
    for i, result in enumerate(top_picks):
        print("#%3d: Movie %d - %s" % (
              i + 1,
              result[ml_parse.MOVIEID_INDEX],
              ml_parse.movie_name(result[ml_parse.MOVIEID_INDEX], movies)))
    print("\n")


def main():
    import configspark
    sc = configspark.SPARK_CONTEXT

    args = parse_args(HELP_PROMPT, sys.argv)
    filename = args.get("filename")
    user_id = int(args.get("user_id")) if args.get("user_id") else None

    user_id, raw_ratings = get_training_data(sc, filename, user_id)
    ratings_train = raw_ratings.map(ml_parse.rating_convert)

    movies = ml_parse.load_movies()

    user_movies_unwatched = movies_unwatched(user_id, ratings_train, movies)

    model = prepare_model(sc, filename, user_id, ratings_train)

    make_recommendations(user_id, model, sc.parallelize(user_movies_unwatched),
                         movies)

    sc.stop()


if __name__ == "__main__":
    main()
