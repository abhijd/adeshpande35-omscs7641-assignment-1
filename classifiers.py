import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.model_selection import LearningCurve, ValidationCurve

m_depth = list(range(1, 40))
m_depth.append(None)

n_estimator = list(range(100, 2100, 100))
n_estimator.append(50)

cv = 10

oh = preprocessing.OneHotEncoder(sparse=False)

normalizer = preprocessing.MinMaxScaler()

scaler = preprocessing.StandardScaler()

idTransformer = preprocessing.FunctionTransformer(None)

# assumed ordinals
education = ["unknown", "primary", "secondary", "tertiary"]
credit_default = ["unknown", "yes", "no"]
# job = ['unknown','unemployed',]

education_encoder = preprocessing.OrdinalEncoder(categories=[education])
credit_default_encoder = preprocessing.OrdinalEncoder(categories=[credit_default])

liv_train, liv_test, liv_ans_train, liv_ans_test = None, None, None, None

bm_train, bm_test, bm_ans_train, bm_ans_test = None, None, None, None

np.random.seed(65)

classifier_list = ["ada_boost", "dt", "knn", "mlp", "svm"]

ada_boost_shuffle_params = {
    "n_estimators": n_estimator,
    "base_estimator": [None, DecisionTreeClassifier(max_depth=1)],
}
dt_shuffle_params = {
    "max_depth": m_depth,
    "min_samples_leaf": range(1, 25),
    "ccp_alpha": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
    "criterion": ["gini"],
}
knn_shuffle_params = {
    "n_neighbors": range(1, 40),
    "metric": ["euclidean", "manhattan", "minkowski"],
    "weights": ["uniform", "distance"],
    "n_jobs": [10],
}
mlp_shuffle_params = {
    "learning_rate_init": [
        0.0000000001,
        0.000000001,
        0.00000001,
        0.0000001,
        0.000001,
        0.00001,
        0.0001,
        0.001,
        0.01,
        0.1,
        1,
    ],
    "solver": [
        "adam",
        # "lbfgs"
    ],
    "max_iter": [10000],
    "hidden_layer_sizes": [(100,), (100, 100)],
    "activation": ["relu", "logistic"],
}
svm_rbf_shuffle_params = {"C": np.logspace(-8, 8, 17)}
svm_sigmoid_shuffle_params = {"C": np.logspace(-8, 8, 17)}

classifier_shuffler = {
    "ada_boost": {
        "classifier": AdaBoostClassifier,
        "shuffle_params": ada_boost_shuffle_params,
        "default_params": {},
    },
    "dt": {
        "classifier": DecisionTreeClassifier,
        "shuffle_params": dt_shuffle_params,
        "default_params": {},
    },
    "knn": {
        "classifier": KNeighborsClassifier,
        "shuffle_params": knn_shuffle_params,
        "default_params": {"n_jobs": 10},
    },
    "mlp": {
        "classifier": MLPClassifier,
        "shuffle_params": mlp_shuffle_params,
        "default_params": {"verbose": False},
    },
    "svm_rbf": {
        "classifier": SVC,
        "shuffle_params": svm_rbf_shuffle_params,
        "default_params": {"kernel": "rbf", "verbose": False},
    },
    "svm_sigmoid": {
        "classifier": SVC,
        "shuffle_params": svm_sigmoid_shuffle_params,
        "default_params": {"kernel": "sigmoid", "verbose": False},
    },
}


def run_classifiers(classifier=None, full=False):
    create_intake()
    loop = [
        [liv_train, liv_test, liv_ans_train, liv_ans_test, "liver"],
        [bm_train, bm_test, bm_ans_train, bm_ans_test, "bank marketing"],
    ]
    for x in loop:
        X_train, X_test, y_train, y_test, name = x[0], x[1], x[2], x[3], x[4]
        classifiers = []
        if full:
            classifiers = classifier_list
        elif not full and classifier in classifier_list:
            classifiers = [classifier]
        for classifier_n in classifiers:
            if classifier_n == "svm":
                run_model(X_train, X_test, y_train, y_test, name, "svm_sigmoid")
                run_model(X_train, X_test, y_train, y_test, name, "svm_rbf")
            else:
                run_model(X_train, X_test, y_train, y_test, name, classifier_n)


def run_model(X_train, X_test, y_train, y_test, name, classifier_n):
    classifier_d = classifier_shuffler[classifier_n]
    classifier_inst = classifier_d["classifier"]
    shuffle_params = classifier_d["shuffle_params"]
    params = classifier_d["default_params"]

    print("{} {}".format(classifier_n, name))

    # set initial accuracy
    classifier_init = classifier_inst(**params)
    classifier_init.fit(X_train, y_train)
    init_score = classifier_init.score(X_test, y_test)

    # Set scoring to accuracy since we are doing classifiers
    scoring = "accuracy"

    # Grid search to get best params
    grid_classifier = classifier_inst(**params)
    grid_params = [shuffle_params]
    start_time = time.time()
    best_classifier = GridSearchCV(
        grid_classifier, grid_params, scoring=scoring, cv=cv, verbose=0, n_jobs=10
    )
    best_classifier.fit(X_train, y_train)
    grid_time = time.time() - start_time
    print("Grid search time: {}".format(grid_time))
    print()

    #print("Grid search results: {}".format(best_classifier.cv_results_))
    print("Grid search best params: {}".format(best_classifier.best_params_))
    print("Grid search accuracy score results: {}".format(best_classifier.score(X_test, y_test)))
    print()

    # get learning curves for fit using the best params
    end_classifier = classifier_inst(**params)
    end_classifier.set_params(**best_classifier.best_params_)
    sizes = np.linspace(0.1, 1.0, 10)
    make_save_learning_curve_chart(
        end_classifier, classifier_n, scoring, sizes, cv, 10, name, X_train, y_train
    )

    # perform time
    end_classifier = classifier_inst(**params)
    end_classifier.set_params(**best_classifier.best_params_)
    start_time = time.time()
    end_classifier.fit(X_train, y_train)
    train_time = time.time() - start_time

    print("Train time: {}".format(train_time))
    print()

    # Run classifier on test data and record time
    start_time = time.time()
    end_score = end_classifier.score(X_test, y_test)
    test_time = time.time() - start_time

    print("Test time: {}".format(test_time))
    print()

    print("Default Score: {}".format(init_score))
    print()
    print("Best Params Score: {}".format(end_score))
    print()
    print()

    write_to_csv(
        name,
        classifier_n,
        best_classifier.best_params_,
        init_score,
        end_score,
        scoring,
        train_time,
        test_time,
        grid_time,
    )


def create_intake():
    global liv_train, liv_test, liv_ans_train, liv_ans_test, bm_train, bm_test, bm_ans_train, bm_ans_test

    liv_dataset = pd.read_csv("indian_liver_patient_dataset.csv")
    liv_answer = liv_dataset["class"]

    liv_dataset = liv_dataset.drop("class", axis=1)

    liv_transformer = make_column_transformer(
        (oh, ["gender"]),
        (
            idTransformer,
            ["age", "TB", "DB", "alkphos", "sgpt", "sgot", "TP", "ALB", "A_G"],
        ),
    )

    liv_full_set = liv_transformer.fit_transform(liv_dataset)

    liv_train, liv_test, liv_ans_train, liv_ans_test = train_test_split(
        liv_full_set, liv_answer, test_size=0.2, random_state=30
    )

    liv_train = scaler.fit_transform(liv_train)
    liv_test = scaler.fit_transform(liv_test)

    bm_dataset = pd.read_csv("bank_marketing_dataset.csv")
    bm_answer = bm_dataset["y"]

    bm_dataset = bm_dataset.drop("y", axis=1)
    # drop predicted outcome
    bm_dataset = bm_dataset.drop("poutcome", axis=1)
    # drop day of month as this is assumed to be a noisy feature
    bm_dataset = bm_dataset.drop("day", axis=1)

    bm_transformer = make_column_transformer(
        (education_encoder, ["education"]),
        (credit_default_encoder, ["default"]),
        (oh, ["job", "marital", "housing", "loan", "contact", "month"]),
        (
            idTransformer,
            ["age", "balance", "duration", "campaign", "pdays", "previous"],
        ),
    )

    bm_full_set = bm_transformer.fit_transform(bm_dataset)

    bm_train, bm_test, bm_ans_train, bm_ans_test = train_test_split(
        bm_full_set, bm_answer, test_size=0.2, random_state=30
    )

    bm_train = scaler.fit_transform(bm_train)
    bm_test = scaler.fit_transform(bm_test)


def make_save_learning_curve_chart(
    model, classifier_n, scoring, sizes, cv, n_jobs, dataset_name, X_train, y_train
):
    v = LearningCurve(model, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=n_jobs)
    v.fit(X_train, y_train)
    v.show("files/{}_learning_curve_{}.png".format(classifier_n, dataset_name))
    plt.clf()


def write_to_csv(
    dataset_name,
    classifier_n,
    best_params,
    init_score,
    improved_score,
    scoring,
    train_time,
    test_time,
    grid_time,
):
    fname = "files/{}_{}_metrics.csv".format(classifier_n, dataset_name)
    try:
        f = open(fname)
    except IOError:
        f = open(fname, "a+")
        f.write(
            "Execution Time,Best Params,Initial Score,Imporved Score,Scoring Type,Training Time,Testing Time,Grid Search Time\n"
        )
    finally:
        f.close()
    with open(fname, "a+") as f:
        f.write(
            "{},{},{},{},{},{},{},{}\n".format(
                time.time(),
                '"{}"'.format(best_params),
                init_score,
                improved_score,
                scoring,
                train_time,
                test_time,
                grid_time,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--full", help="Run all classifiers", action="store_true")
    g.add_argument("-c", help="Run a specific classifier: ada_boost, dt, knn, mlp, svm")
    args = parser.parse_args()
    if args.full:
        print("Running all experiments")
        run_experiments(full=True)
    else:
        if args.c.lower() not in ["ada_boost", "dt", "knn", "mlp", "svm"]:
            raise ValueError(
                "Invalid classifier, choose from: ada_boost, dt, knn, mlp, svm"
            )
        else:
            run_classifiers(classifier=args.c)


if __name__ == "__main__":
    main()
