# Large Scale Recommendation Systems
### Data Science Assignment 2
### Daniel Reidler Haiwei Su, Sean Herman

##Setup

    pip install -r requirements.txt

## Instructions

### ALS Recommendation Model
#### Data splits
To generate the ALS recommendation Model, first download the [MovieLens dataset](http://grouplens.org/datasets/movielens/10m/). This dataset should be split into training-validation-test partitions (60-20-20). Update the configuration variables in `split.py` to point at the files extracted from the MovieLens dataset download. To generate the splits, run the `split.py` script.

    $ ./split.py

#### Model parameter evaluation
Given the train-validation-test splits, run the `evaluate_als_models.py` script.

    $ ./evaluate_als_models.py

This script will build ALS models from the training set using a range of parameters. The ALS Model parameters and Mean Squared Error (MSE) are written to `results/als_model_evaluation.txt`. The model with the lowest MSE is saved to disk at `results/als_recommended_model`.

### Task 1 - User Recommendations
To generate new recommendations for a user, execute the `recommend.py` script. This script's CLI supports either an input CSV file, or simply a userId from the existing MovieLens dataset. In both cases, the script will produce 20 new recommendations for the user.

To make recommendations for a single existing MovieLens user, use the `-u` CLI flag.

        $ ./recommend.py -u 355

To add new ratings for an existing MovieLens user, provide a CSV file.

        $ ./recommend.py test_users/user2.csv

The CSV file provided to `recommend.py` must consist of lines of exactly `userId,movieId,rating`. No header row should be included. See `test_users/user2.csv` for an example.

### Task 2 - Dimensionality Reduction
TODO

### Task 3 - Million Song Data Set Recommendations
TODO
