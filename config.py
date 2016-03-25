import os

DATA_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "data"))

ML_USERID_INDEX = 0
ML_MOVIEID_INDEX = 1
ML_RATING_INDEX = 2
ML_TIMESTAMP_INDEX = 3

ML_DATA_PATH = os.path.join(DATA_PATH, "ml-10M100K")
ML_RATINGS = os.path.join(ML_DATA_PATH, "ratings.dat")
ML_MOVIES = os.path.join(ML_DATA_PATH, "movies.dat")
ML_TAGS = os.path.join(ML_DATA_PATH, "tags.dat")

ML_RATINGS_TRAIN = os.path.join(DATA_PATH, "ratings-train.dat")
ML_RATINGS_TEST = os.path.join(DATA_PATH, "ratings-test.dat")
ML_RATINGS_VALIDATION = os.path.join(DATA_PATH, "ratings-validation.dat")

DATA_PATH = os.path.join(os.path.realpath(__file__), "..", "data")

TEST_DATA_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "data-test"))

TRAIN_FILE = os.path.join(DATA_PATH, "ratings-train.dat")
VALIDATION_FILE = os.path.join(DATA_PATH, "ratings-validation.dat")
TEST_FILE = os.path.join(DATA_PATH, "ratings-test.dat")
# TRAIN_FILE = os.path.join(TEST_DATA_PATH, "ratings-train-1000.dat")
# VALIDATION_FILE = os.path.join(TEST_DATA_PATH, "ratings-validation-1000.dat")
# TEST_FILE = os.path.join(TEST_DATA_PATH, "ratings-test-1000.dat")

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "results"))

if not os.path.exists(RESULTS_PATH):
    print("Creating directory %s" % RESULTS_PATH)
    os.mkdir(RESULTS_PATH)

RESULTS_FILE = os.path.join(RESULTS_PATH, "als_model_evaluation.csv")
ALS_BEST_PARAMS_FILE = os.path.join(RESULTS_PATH, "als_params.csv")
