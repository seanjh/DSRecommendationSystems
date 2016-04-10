import time
import config

from pyspark.mllib.recommendation import Rating

USERID_INDEX = 0
MOVIEID_INDEX = 1
RATING_INDEX = 2
TIMESTAMP_INDEX = 3


def parse_user_input(line):
    parts = line.strip().split(",")
    current_time = time.time()
    return (
        int(parts[USERID_INDEX]),
        int(parts[MOVIEID_INDEX]),
        float(parts[RATING_INDEX]),
        int(current_time)
    )


def parse_line(line):
    parts = line.strip().split("::")
    return (
        int(parts[USERID_INDEX]),
        int(parts[MOVIEID_INDEX]),
        float(parts[RATING_INDEX]),
        int(parts[TIMESTAMP_INDEX])
    )


def rating_convert(row):
    return Rating(
        int(row[USERID_INDEX]),
        int(row[MOVIEID_INDEX]),
        float(row[RATING_INDEX])
    )


def parsed_string(row):
    return "::".join([str(val) for val in row])


def user_product(row):
    return (row[USERID_INDEX], row[MOVIEID_INDEX])


def user_product_rating(row):
    return (user_product(row), row[RATING_INDEX])


MOVIES_FIELDS = ["movieId", "title", "genres"]
MOVIES_MOVIEID_INDEX = 0
MOVIES_TITLE_INDEX = 1


def load_movies(delimiter="::"):
    result = dict()
    with open(config.ML_MOVIES) as infile:
        for row in infile:
            parts = row.strip().split(delimiter)
            result[parts[MOVIES_MOVIEID_INDEX]] = parts[MOVIES_TITLE_INDEX]
    return result


def movie_name(movie_id, movies):
    return movies.get(str(movie_id), "Unrecognized")
