import os
import argparse
import joblib
import tarfile
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--criterion', type=str, default='gini')
    parser.add_argument('--max_depth', type=int, default=45)
    parser.add_argument('--max_features', type=str, default='log2')
    parser.add_argument('--n_estimators', type=int, default=270)
    parser.add_argument('--random_state', type=int, default=42)

    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # model directory
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()

def get_train_data(train_dir):
    train_data = pd.read_csv(os.path.join(train_dir, "train.csv"))
    x_train = train_data.iloc[:, :-1].to_numpy()
    y_train = train_data.iloc[:, -1].to_numpy()
    print("x train", x_train.shape, "y train", y_train.shape)

    return x_train, y_train

def get_test_data(test_dir):
    test_data = pd.read_csv(os.path.join(test_dir, "test.csv"))
    x_test = test_data.iloc[:, :-1].to_numpy()
    y_test = test_data.iloc[:, -1].to_numpy()
    print("x test", x_test.shape, "y test", y_test.shape)

    return x_test, y_test

if __name__ == "__main__":
    args, _ = parse_args()

    print("Training data location: {}".format(args.train))
    print("Test data location: {}".format(args.test))
    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)

    model_RF = RandomForestClassifier(n_jobs=args.n_jobs, 
                                      criterion=args.criterion, 
                                      max_depth=args.max_depth, 
                                      max_features=args.max_features, 
                                      n_estimators=args.n_estimators, 
                                      random_state=args.random_state)
    model_RF.fit(x_train, y_train)
    print("End Train RandomForestClassifier")

    # evaluate on test set
    scores = model_RF.score(x_test, y_test)
    print("Testing Accuracy: {:.3f}".format(scores))

    # save model
    joblib.dump(model_RF, os.path.join(args.model_dir, "model.joblib"))