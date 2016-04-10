#!/usr/bin/env python

import sys
import os
import config
import evaluate_als_models as model_evaluate
import movielens_parse as mlparse

from pyspark.mllib.recommendation import MatrixFactorizationModel, ALS


RECOMMEND_NUM = 20
sc = None  # initialized within main()


HELP_PROMPT = """MovieLens Recommendation Engine

usage:
    ./recommend.py <USER_RATINGS_FILE>
    ./recommend.py -u <USER_ID>
"""


def movies_unwatched(user_id, train, movies):
    # Movies already seen by the user
    seen_movie_ids = user_movies_seen(train, user_id)
    print("\nMovies seen by UserID %s:" % user_id)
    for movie_id in seen_movie_ids:
        print("Movie %d: %s" % (movie_id, mlparse.movie_name(movie_id, movies)))

    # filter full movies vector to those not in movies rated by user
    user_movies_unwatched = [(user_id, movie_id) for movie_id in movies.keys()
                             if movie_id not in seen_movie_ids]
    print("\n")

    return user_movies_unwatched


def user_movies_seen(ratings_train, user_id):
    return (
        ratings_train
        .filter(lambda row: row[mlparse.USERID_INDEX] == user_id)
        .map(lambda row: row[mlparse.MOVIEID_INDEX])
        .collect())


def parse_user_id(data):
    user_id = None

    for d in data:
        parsed_user_id = d[mlparse.USERID_INDEX]
        if user_id is None:
            user_id = parsed_user_id
        elif user_id != parsed_user_id:
            raise RuntimeError("Multiple user IDs detected input file")

    return user_id


def parse_file(filename):
    if not os.path.exists(filename):
        raise RuntimeError("Could not open file %s" % filename)

    with open(filename) as infile:
        ratings = [mlparse.parse_user_input(line) for line in infile]

    user_id = parse_user_id(ratings)
    return user_id, ratings


def get_training_data(sc, filename, user_id):
    ratings_train_text = sc.textFile(config.ML_RATINGS_TRAIN)
    ratings_train = ratings_train_text.map(mlparse.parse_line)

    if filename is not None:
        print("Loading new user ratings from %s" % filename)
        # parse new ratings from file
        user_id, new_ratings = parse_file(filename)
        print("New user ratings: %s" % new_ratings)
        # add new ratings to the existing dataset
        ratings_train = sc.union([ratings_train, sc.parallelize(new_ratings)])

    print("Done getting training data")
    return user_id, ratings_train


def load_best_params():
    best_params_file = os.path.join("results", "als_params.csv")
    if not os.path.exists(best_params_file):
        raise RuntimeError("Cannot locate best ALS parameters file %s"
                           % best_params_file)

    with open(best_params_file) as infile:
        lines = [line for line in infile]

    parts = lines[1].strip().split(",")
    return parts[0], parts[1]


def prepare_model(sc, filename, user_id, ratings_train):
    if filename is None and os.path.exists(config.ML_MODEL):
        # load the trained model
        print("\n\nLoading existing recommendation model from %s\n\n"
              % config.ML_MODEL)
        model = MatrixFactorizationModel.load(sc, config.ML_MODEL)
    else:
        # train a new model
        print("\n\nRetraining recommendation model for User #%s\n\n" % user_id)
        rank, lambda_val = load_best_params()
        rank, lambda_val = int(rank), float(lambda_val)
        model = ALS.train(ratings_train, rank, config.ITERATIONS, lambda_val,
                          nonnegative=True)

    return model


def make_recommendations(user_id, model, user_movies, movies):
    predictions = model.predictAll(user_movies)
    top_picks = (
        predictions
        .sortBy(lambda row: row[mlparse.RATING_INDEX], ascending=False)
        .take(RECOMMEND_NUM))
    print("\nNew movie recommendations for User #%s:" % user_id)
    for i, result in enumerate(top_picks):
        print("#%3d: %s" %
              (i, mlparse.movie_name(result[mlparse.MOVIEID_INDEX], movies)))
    print("\n")


def main(filename=None, user_id=None):
    import configspark
    sc = configspark.SPARK_CONTEXT

    user_id, raw_ratings = get_training_data(sc, filename, user_id)
    ratings_train = raw_ratings.map(mlparse.rating_convert)

    movies = mlparse.load_movies()

    user_movies_unwatched = movies_unwatched(user_id, ratings_train, movies)

    model = prepare_model(sc, filename, user_id, ratings_train)

    make_recommendations(user_id, model, sc.parallelize(user_movies_unwatched),
                         movies)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print(HELP_PROMPT)
            sys.exit(0)
        elif len(sys.argv) == 3 and sys.argv[1] == '-u':
            main(user_id=int(sys.argv[2]))
            sys.exit(0)
        elif len(sys.argv) == 2:
            main(filename=sys.argv[1])
            sys.exit(0)

    print(HELP_PROMPT)
    sys.exit("Error: Invalid arguments\n")
