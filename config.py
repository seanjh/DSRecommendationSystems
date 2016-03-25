import os
import findspark

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

findspark.init()
SPARK_CONF = SparkConf().setAppName("dshw2")
SPARK_CONTEXT = SparkContext(conf=SPARK_CONF)
SPARK_SQL_CONTEXT = SQLContext(SPARK_CONTEXT)

DATA_PATH = os.path.join(os.path.realpath(__file__), "..", "data")
TRAIN_FILE = os.path.join(DATA_PATH, "ratings-train.dat")
VALIDATION_FILE = os.path.join(DATA_PATH, "ratings-validation.dat")
TEST_FILE = os.path.join(DATA_PATH, "ratings-test.dat")

# DATA_PATH = os.path.join(os.path.realpath(__file__), "..", "data-test")
# TRAIN_FILE = os.path.join(DATA_PATH, "ratings-train-1000.dat")
# VALIDATION_FILE = os.path.join(DATA_PATH, "ratings-validation-1000.dat")
# TEST_FILE = os.path.join(DATA_PATH, "ratings-test-1000.dat")

RESULTS_FILE="results.txt"
