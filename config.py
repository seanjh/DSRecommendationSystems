import os
import shutil


def clean_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)

DATA_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "data"))

# ML-10M100K source files
ML_DATA_PATH = os.path.join(DATA_PATH, "ml-10M100K")
ML_RATINGS = os.path.join(ML_DATA_PATH, "ratings.dat")
ML_MOVIES = os.path.join(ML_DATA_PATH, "movies.dat")
ML_TAGS = os.path.join(ML_DATA_PATH, "tags.dat")

# ML-10M100K split files
ML_RATINGS_TRAIN = os.path.join(DATA_PATH, "ratings-train.dat")
ML_RATINGS_TEST = os.path.join(DATA_PATH, "ratings-test.dat")
ML_RATINGS_VALIDATION = os.path.join(DATA_PATH, "ratings-validation.dat")

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "results"))

if not os.path.exists(RESULTS_PATH):
    print("Creating directory %s" % RESULTS_PATH)
    os.mkdir(RESULTS_PATH)

# Evaluate results
RESULTS_FILE = os.path.join(RESULTS_PATH, "als_model_evaluation.csv")
ALS_BEST_PARAMS_FILE = os.path.join(RESULTS_PATH, "als_params.csv")
ML_MODEL = os.path.join(RESULTS_PATH, "movielens.mllib.model")

MSD_TRAIN = os.path.join(DATA_PATH, "train_triplets.txt")
MSD_USERID_MAP = os.path.join(RESULTS_PATH, "msd_user_ids.json")
MSD_SONGID_MAP = os.path.join(RESULTS_PATH, "msd_song_ids.json")
