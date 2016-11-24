import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif, SelectPercentile, VarianceThreshold
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import math
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import Imputer, StandardScaler
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from utils_functions import *


def partial_model_selection(main_path, dataset_type, dataset_subtype, sampling, sampling_timing, fs_step_name, classifier_step_name, chromosome):
    start = time.time()

    print("##### Experiment Info #####")
    print("Chromosome:", chromosome)
    print("Dataset type:", dataset_type)
    print("Dataset subtype:", dataset_subtype)
    print("Sampling:", sampling)
    print("Sampling timing:", sampling_timing)
    print("Filter FS:", fs_step_name)
    print("Classifier:", classifier_step_name)
    print()

    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/train_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    print("Variable names size:", len(variable_names))
    print()

    sampling_seeds = [123, 456, 789]

    print("Loading training data...")
    print()

    train_data = load_dataset(main_path + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/train_train.csv')

    X_train = train_data[:, 1:]
    print("X_train shape:", X_train.shape)
    print()

    y_train = train_data[:, 0]
    print("y_train shape:", y_train.shape)
    print()

    experiment_results = dict()

    param_grid = dict()

    print("Creating pipeline...")
    print()
    pipe = Pipeline([("imputer", Imputer(missing_values=-1)),
                     ("variance", VarianceThreshold()),
                     ("scaler", StandardScaler())])

    if fs_step_name == "anova":
        filter = SelectPercentile(f_classif, percentile=2)

    if sampling_timing == "sampling_before_fs":
        if sampling == "down_sample":
            pipe.steps.append((sampling, RandomUnderSampler(random_state=sampling_seeds[0])))
        elif sampling == "up_sample":
            pipe.steps.append((sampling, RandomOverSampler(random_state=sampling_seeds[1])))
        elif sampling == "smote_sample":
            pipe.steps.append((sampling, SMOTE(n_jobs=-1, random_state=sampling_seeds[2])))

        pipe.steps.append((fs_step_name, filter))
    elif sampling_timing == "sampling_after_fs":
        pipe.steps.append((fs_step_name, filter))

        if sampling == "down_sample":
            pipe.steps.append((sampling, RandomUnderSampler(random_state=sampling_seeds[0])))
        elif sampling == "up_sample":
            pipe.steps.append((sampling, RandomOverSampler(random_state=sampling_seeds[1])))
        elif sampling == "smote_sample":
            pipe.steps.append((sampling, SMOTE(n_jobs=-1, random_state=sampling_seeds[2])))

    if classifier_step_name == "linear_svm":
        C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        param_grid[classifier_step_name + '__C'] = C_OPTIONS

        classifier = SVC(kernel='linear', random_state=123456, probability=True, class_weight='balanced')
    elif classifier_step_name == "rf":
        powers = list(np.arange(1, 3.2, 0.2))
        NUM_TREES_OPTIONS = list(map(math.floor, np.multiply(3, list(map(math.pow, [10] * len(powers), powers)))))

        param_grid[classifier_step_name + '__n_estimators'] = NUM_TREES_OPTIONS

        classifier = RandomForestClassifier(oob_score=True, random_state=123456, n_jobs=-1, bootstrap=True,
                                            class_weight="balanced")
    elif classifier_step_name == "knn":
        max_num_neighbors = 60

        NUM_NEIGHBORS_OPTIONS = list(np.arange(5, max_num_neighbors, 15))

        param_grid[classifier_step_name + '__n_neighbors'] = NUM_NEIGHBORS_OPTIONS

        classifier = KNeighborsClassifier(n_jobs=-1)

    pipe.steps.append((classifier_step_name, classifier))

    print("Performing gridsearch...")
    print()

    pipe_gridsearch = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, scoring='f1_weighted',
                                   cv=StratifiedKFold(n_splits=5, random_state=123456), verbose=10)
    pipe_gridsearch.fit(X_train, y_train)

    end1 = time.time()

    experiment_results['gridsearch_cv_time'] = end1

    print("################################################ Time until GridSearchCV:", str(np.round((end1 - start) / 60)), "minutes")
    print()

    cv_results = dict()
    cv_results['mean_test_score'] = pipe_gridsearch.cv_results_['mean_test_score']
    cv_results['std_test_score'] = pipe_gridsearch.cv_results_['std_test_score']
    cv_results['mean_train_score'] = pipe_gridsearch.cv_results_['mean_train_score']
    cv_results['std_train_score'] = pipe_gridsearch.cv_results_['std_train_score']
    cv_results['params'] = pipe_gridsearch.cv_results_['params']

    experiment_results['cv_results'] = cv_results

    print("GridSearchCV results:")
    print()
    print(cv_results['mean_test_score'])
    print()
    print(cv_results['std_test_score'])
    print()

    print("Best parameters set found on development set:")
    print()
    print(pipe_gridsearch.best_params_)
    print()

    experiment_results['best_estimator'] = pipe_gridsearch.best_estimator_

    cv_score = np.mean(cross_val_score(pipe_gridsearch.best_estimator_, X_train, y_train, n_jobs=-1,
                                       cv=StratifiedKFold(n_splits=5, random_state=789012), scoring='f1_weighted'))

    experiment_results['cv_score'] = cv_score

    print("CV score:")
    print()
    print(cv_score)
    print()

    y_train_pred = pipe_gridsearch.best_estimator_.predict(X_train)
    train_score = f1_score(y_train, y_train_pred, average='weighted')
    experiment_results['train_score'] = train_score

    print("Train score:")
    print()
    print(train_score)
    print()

    result_files_path = os.getcwd() + '/' + fs_step_name + '/classifiers/' + classifier_step_name + '/' + \
                        sampling_timing + '/' + '/' + sampling + '/' + dataset_type + '/' + dataset_subtype + '/' + \
                        chromosome

    if classifier_step_name == "linear_svm":
        save_object(experiment_results, result_files_path + '/train_' + classifier_step_name + '_results.pkl')

        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('linear SVM coefficients', 'f4')])

        features_info['names'] = variable_names

        coefficients = np.zeros(X_train.shape[1])
        coefficients[pipe_gridsearch.best_estimator_.named_steps["variance"].get_support()][
            pipe_gridsearch.best_estimator_.named_steps[fs_step_name].get_support()] = np.absolute(
            pipe_gridsearch.best_estimator_.named_steps[classifier_step_name].coef_[0, :])

        features_info['linear SVM coefficients'] = coefficients

        with open(result_files_path + '/train_coefficients_features_info.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['names', 'linear SVM coefficients'])
            w.writerows(features_info)
    elif classifier_step_name == "rf":
        save_object(experiment_results, result_files_path + '/train_' + classifier_step_name + '_results.pkl')

        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('RF importances', 'f4')])

        features_info['names'] = variable_names

        importances = np.zeros(X_train.shape[1])
        importances[pipe_gridsearch.best_estimator_.named_steps["variance"].get_support()][
            pipe_gridsearch.best_estimator_.named_steps[fs_step_name].get_support()] = pipe_gridsearch.best_estimator_.named_steps[
            classifier_step_name].feature_importances_

        features_info['RF importances'] = importances

        with open(result_files_path + '/train_importances_features_info.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['names', 'RF importances'])
            w.writerows(features_info)
    elif classifier_step_name == "knn":
        save_object(experiment_results, result_files_path + '/train_' + classifier_step_name + '_results.pkl')

    end2 = time.time()
    print("################################################ Time end:", str(np.round((end2 - end1) / 60)))
    print()


def manual_partial_model_selection(main_path, dataset_type, dataset_sub_type, sampling, sampling_timing, fs_step_name, classifier_step_name, chromosome):
    print("##### Experiment Info #####")
    print("Chromosome:", chromosome)
    print("Dataset type:", dataset_type)
    print("Dataset subtype:", dataset_sub_type)
    print("Sampling:", sampling)
    print("Sampling timing:", sampling_timing)
    print("Filter FS:", fs_step_name)
    print("Classifier:", classifier_step_name)
    print()

    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/' + dataset_sub_type + '/' + chromosome + '/train_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    print("Variable names size:", len(variable_names))
    print()

    sampling_seeds = [123, 456, 789]

    print("Loading training data...")
    print()

    train_data = load_dataset(main_path + dataset_type + '/' + dataset_sub_type + '/' + chromosome + '/train_train.csv')

    X_train = train_data[:, 1:]
    print("X_train shape:", X_train.shape)
    print()

    y_train = train_data[:, 0]
    print("y_train shape:", y_train.shape)
    print()

    experiment_results = dict()

    print("Creating pipeline...")
    print()
    pipe = Pipeline([("imputer", Imputer(missing_values=-1)),
                     ("variance", VarianceThreshold()),
                     ("scaler", StandardScaler())])

    if fs_step_name == "anova":
        filter = SelectPercentile(f_classif, percentile=2)

    if sampling_timing == "sampling_before_fs":
        if sampling == "down_sample":
            pipe.steps.append((sampling, RandomUnderSampler(random_state=sampling_seeds[0])))
        elif sampling == "up_sample":
            pipe.steps.append((sampling, RandomOverSampler(random_state=sampling_seeds[1])))
        elif sampling == "smote_sample":
            pipe.steps.append((sampling, SMOTE(n_jobs=-1, random_state=sampling_seeds[2])))

        pipe.steps.append((fs_step_name, filter))
    elif sampling_timing == "sampling_after_fs":
        pipe.steps.append((fs_step_name, filter))

        if sampling == "down_sample":
            pipe.steps.append((sampling, RandomUnderSampler(random_state=sampling_seeds[0])))
        elif sampling == "up_sample":
            pipe.steps.append((sampling, RandomOverSampler(random_state=sampling_seeds[1])))
        elif sampling == "smote_sample":
            pipe.steps.append((sampling, SMOTE(n_jobs=-1, random_state=sampling_seeds[2])))

    classifier = SVC(kernel='linear', random_state=123456, probability=True, class_weight='balanced')

    pipe.steps.append((classifier_step_name, classifier))

    print("Performing manual gridsearch...")
    print()

    C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    cv = StratifiedKFold(n_splits=5, random_state=123456)
    f1_cv = []
    mean_test_score = []
    std_test_score = []

    for C in C_OPTIONS:
        pipe.set_params(linear_svm__C=C)

        for train_indexes, validation_indexes in cv.split(X_train, y_train):
            pipe.fit(X_train[train_indexes, :], y_train[train_indexes])

            y_pred = pipe.predict(X_train[validation_indexes, :])

            f1 = f1_score(y_train[validation_indexes], y_pred, average='weighted')
            f1_cv.append(f1)

        mean_test_score.append(np.mean(f1_cv))
        std_test_score.append(np.std(f1_cv))

    cv_results = dict()
    cv_results['mean_test_score'] = mean_test_score
    cv_results['std_test_score'] = std_test_score
    cv_results['params'] = C_OPTIONS

    experiment_results['cv_results'] = cv_results

    print("Manual gridsearch results:")
    print()
    print(cv_results['mean_test_score'])
    print()
    print(cv_results['std_test_score'])
    print()

    print("Best parameters set found on development set:")
    print()
    print(C_OPTIONS[np.argmax(mean_test_score)])
    print()


if __name__ == '__main__':
    disease = "diabetes"

    main_path = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast-project-diabetes/health-forecast/datasets/' + disease + '/'

    sampling_timings = ["sampling_after_fs"]
    sampling_types = ["raw"]
    dataset_types = ["genomic_epidemiological"]
    dataset_sub_types = ["D2_vs_H"]
    fs_step_names = ["anova"]
    classifier_step_names = ["linear_svm"]

    for fs_step_name in fs_step_names:
        fs_dir = os.getcwd() + '/' + fs_step_name

        if not os.path.exists(fs_dir):
            os.makedirs(fs_dir)

        for classifier_step_name in classifier_step_names:
            classifier_dir = fs_dir + '/classifiers/' + classifier_step_name

            if not os.path.exists(classifier_dir):
                os.makedirs(classifier_dir)

            for sampling_timing in sampling_timings:
                sampling_timing_dir = classifier_dir + '/' + sampling_timing

                if not os.path.exists(sampling_timing_dir):
                    os.makedirs(sampling_timing_dir)

                for sampling in sampling_types:
                    sampling_dir = sampling_timing_dir + '/' + sampling

                    if not os.path.exists(sampling_dir):
                        os.makedirs(sampling_dir)

                    for dataset_type in dataset_types:
                        dataset_dir = sampling_dir + '/' + dataset_type

                        if not os.path.exists(dataset_dir):
                            os.makedirs(dataset_dir)

                        for dataset_subtype in dataset_sub_types:
                            dataset_sub_type_dir = dataset_dir + '/' + dataset_sub_type

                            if not os.path.exists(dataset_sub_type_dir):
                                os.makedirs(dataset_sub_type_dir)

                            for i in range(1, 23):
                                chromosome = "chr" + str(i)
                                chromosome_dir = dataset_sub_type_dir + '/' + chromosome

                                if not os.path.exists(chromosome_dir):
                                    os.makedirs(chromosome_dir)

                                partial_model_selection(main_path, dataset_type, dataset_subtype, sampling, sampling_timing,
                                                        fs_step_name, classifier_step_name, chromosome)
                                stability_feature_selection(main_path, dataset_type, sampling, sampling_timing,
                                                            fs_step_name, classifier_step_name,
                                                            dataset_subtype, chromosome)
                                # manual_partial_model_selection(main_path, dataset_type, dataset_sub_type, sampling,
                                #                                sampling_timing, fs_step_name, classifier_step_name,
                                #                                chromosome)