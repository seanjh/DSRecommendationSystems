#!/usr/bin/env python

import time
import random
import config
import configspark


sc = configspark.SPARK_CONTEXT


def main():
    full_text = sc.textFile(config.MSD_DATA)

    random.seed(time.time())
    seed = int(random.random())
    train, validation, test = full_text.randomSplit([0.6, 0.2, 0.2], seed=seed)

    train.saveAsTextFile(config.MSD_TRAIN)
    validation.saveAsTextFile(config.MSD_VALIDATION)
    test.saveAsTextFile(config.MSD_TEST)

    sc.stop()


if __name__ == "__main__":
    main()
