#!/usr/bin/env python

import sys
import os
import config
import movielens_parse as mlparse

from pyspark.mllib.recommendation import MatrixFactorizationModel


RECOMMEND_NUM = 20


HELP_PROMPT = """MovieLens Recommendation Engine

usage: ./recommend.py <USER_RATINGS_FILE>
"""


def user_movies_seen(ratings_train, user_id):
    return (
        ratings_train
        .filter(lambda row: row[mlparse.USERID_INDEX] == user_id)
        .map(lambda row: row[mlparse.MOVIEID_INDEX])
        .collect())


def main():
    import configspark
    sc = configspark.SPARK_CONTEXT

    # parse user ratings from file
    # TODO

    ratings_train_text = sc.textFile(config.ML_RATINGS_TRAIN)
    ratings_train = ratings_train_text.map(mlparse.parse_line)

    TEMP_USER_ID = 1

    # load all movies
    movies = mlparse.load_movies()

    # Movies already seen by the user
    seen_movie_ids = user_movies_seen(ratings_train, TEMP_USER_ID)
    print("Movies seen by UserID %d:" % TEMP_USER_ID)
    for movie_id in seen_movie_ids:
        print("MovieID %d -- %s" % (movie_id,
                                    mlparse.movie_name(movie_id, movies)))

    # filter full movies vector to those not in movies rated by user
    new_movies = sc.parallelize(
        [(TEMP_USER_ID, movie_id) for movie_id in movies.keys()
         if movie_id not in seen_movie_ids]
    )
    print(new_movies.take(10))

    # load the trained model
    if os.path.exists(config.ML_MODEL):
        print("Loading existing recommendation model from %s" % config.ML_MODEL)
        model = MatrixFactorizationModel.load(sc, config.ML_MODEL)
    else:
        print("Recommendation model cannot be loaded from %s" % config.ML_MODEL)
        sys.exit(1)

    # predictAll ratings for the user
    predictions = model.predictAll(new_movies)
    top_picks = (
        predictions
        .sortBy(lambda row: row[mlparse.RATING_INDEX], ascending=False)
        .take(RECOMMEND_NUM))
    for i, result in enumerate(top_picks):
        print("#%3d: %s" %
              (i, mlparse.movie_name(result[mlparse.MOVIEID_INDEX], movies)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print(HELP_PROMPT)
            sys.exit(0)
        else:
            main()
    else:
        print("Error: Missing required <USER_RATINGS_FILE> argument\n")
        print(HELP_PROMPT)
        sys.exit(1)
