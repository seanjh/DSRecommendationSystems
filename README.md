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
