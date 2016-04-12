# Large Scale Recommendation Systems
### Data Science Assignment 2
### Daniel Reidler Haiwei Su, Sean Herman

##Setup

    pip install -r requirements.txt

## Instructions

### Task 1 - Data Splits
To generate the ALS recommendation Model, first download the [MovieLens dataset](http://grouplens.org/datasets/movielens/10m/). This dataset should be split into training-validation-test partitions (60-20-20). Update the configuration variables in `split.py` to point at the files extracted from the MovieLens dataset download. To generate the splits, run the `ml_split.py` script.

    $ ./ml_split.py


### Task 1 - ALS Model Parameter Evaluation
Given the train-validation-test splits, run the `ml_evaluate_models.py` script.

    $ ./ml_evaluate_models.py

This script will build ALS models from the training set using a range of parameters. The ALS Model parameters and Mean Squared Error (MSE) are written to `results/als_model_evaluation.txt`. The model with the lowest MSE is saved to disk at `results/als_recommended_model`.


### Task 1 - Testing Model Performance
The `ml_evaluate_models.py` script will save the best model to disk in `results/`. To evaluate the performance of this model on the test dataset split, execute the `ml_test.py` script.

    $ ./ml_test.py


The mean squared error (MSE) and root mean squared error (RMSE) will be printed to the console. For example:

    ALS model performance: MSE=0.651 RMSE=0.807


### Task 1 - MovieLens User Recommendations
To generate new recommendations for a user, execute the `ml_recommend.py` script. This script's CLI supports either an input CSV file, or simply a userId from the existing MovieLens dataset. In both cases, the script will produce 20 new recommendations for the user.

To make recommendations for a single existing MovieLens user, use the `-u` CLI flag.

        $ ./ml_recommend.py -u 355

To add new ratings for an existing MovieLens user, provide a CSV file.

        $ ./ml_recommend.py test_users/ml_test.csv

The CSV file provided to `ml_recommend.py` must consist of lines of exactly `userId,movieId,rating`. No header row should be included. See `test_users/ml_test.csv` for an example.

### Task 2 - Dimensionality Reduction

We are curious if reducing bias from dataset can improve recommendation performance. Therefore, we calculate two kinds of bias: item-specific bias and user-specific bias. We want to recalculate the ratings by reducing these bias and compare its performance with previous method in task 1.

To calculate two types of bias, run the file " hw2_task2.ipynb".

This will give a new dataset containing the ratings after deduct two types of bias from original ratings values.

After getting the new ratings without bias, we repeated the parameter optimization process in task1 to find the rank and labmda values with minimum RMSE.

### Task 3 - Million Song Data Set Recommendations

The datasplit for Task 3 was performed with the `msd_split.py` script. For recommendations, the interface for Task 3 is identical to the recommendation interface for Task 1, except the `msd_recommend.py` file should be executed. The original User IDs from the Million Song Dataset (MSD) should be passed to the CLI.

To make recommendations for a single existing MSD user, use the `-u` CLI flag.

        $ ./msd_recommend.py -u b80344d063b5ccb3212f76538f3d9e43d87dca9e

To add new ratings for an existing MovieLens user, provide a CSV file.

        $ ./msd_recommend.py test_users/msd_test.csv

## Test Results

### Task 1 - MovieLens

Our MovieLens evalution script, `ml_evaluate_models.py`, trained models on all combinations of ranks `10`, `20`, `30`, `40`, and `50` and λ values `0.01`, `0.1`, `1.0`, `10.0`. When tested on the validation set, the best performing MovieLens model was:

|Rank|λ|MSE|RMSE|
|----|------|---|----|
|20|0.100000|0.584875|0.764771|

The full evaluation results are available in `results/ml_als_model_evaluation.csv`.

The `ml_test.py` script makes predictions and evaluates the performance of the test dataset using the ALS model with the best parameters discovered during evaluation. These best parameters are loaded from `results/ml_als_params.csv`. The results of the test run were:

|RMSE|
|----|
|0.808|


### Task 2 - MovieLens with dimensionality reduction

In task 2, after we get the new training set with new ratings value, we had some difficulties optimizing the rank and lambda parameters. The issue we had is during model build-up process, the model prediciton is empty. During debugging process, we found that the dataset type we passed into funciton are not the compatible, namely, dataframe against RDD file. Then, we also noticed that the column names between new training dataset and validation dataset does not match. After we fixed this, we rerun the code but still gives us MRE = 0.0 which is impossible in real life. But we assumed that the best parameter is similar to the one chosen from task 1. 

### Task 3 - Million Song Dataset

The `msd_test.py` script re-runs the ALS model with these best params, saved in `results/msd_als_params.csv`, against the test dataset. The results of the test run were:

|RMSE|
|----|
|7.436|

Again, the full results of the validation runs are available in `results/msd_als_model_evaluation.csv`.
