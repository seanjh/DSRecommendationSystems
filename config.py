import os
import findspark

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

findspark.init()
SPARK_CONF = SparkConf().setAppName("dshw2")
SPARK_CONTEXT = SparkContext(conf=SPARK_CONF)
SPARK_SQL_CONTEXT = SQLContext(SPARK_CONTEXT)

DATA_PATH = os.path.join(os.path.realpath(__file__), "data")
