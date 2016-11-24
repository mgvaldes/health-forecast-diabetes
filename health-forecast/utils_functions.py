import numpy as np
from sklearn.externals import joblib
import os
import csv
import time
from sklearn.utils import shuffle


def load_dataset(filename, delimiter=',', skiprows=1, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        load_dataset.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, load_dataset.rowlength))
    return data


def save_object(obj, filename):
    joblib.dump(obj, filename, compress=1)


def load_object(filename):
    return joblib.load(filename)


def stability_feature_selection(main_path, dataset_type, sampling, sampling_timing, fs_step_name, classifier_step_name,
                                dataset_subtype, chromosome):

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

    print("Loading best estimator...")
    print()

    result_files_path = os.getcwd() + '/' + fs_step_name + '/classifiers/' + classifier_step_name + '/' + sampling_timing + \
                        '/' + sampling + '/' + dataset_type + '/' + dataset_subtype

    experiment_results = load_object(result_files_path + '/' + chromosome + '/train_' + classifier_step_name + '_results.pkl')

    best_estimator = experiment_results['best_estimator']

    print("Loading stability data...")
    print()

    stability_data = load_dataset(
        main_path + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/train_stability.csv')

    X_stability = stability_data[:, 1:]
    print("X_stability shape:", X_stability.shape)
    print()

    y_stability = stability_data[:, 0]
    print("y_stability shape:", y_stability.shape)
    print()

    num_experiments = 100
    feature_ranking = np.zeros((len(variable_names), num_experiments))
    coefficients = np.zeros((len(variable_names), num_experiments))

    for i in range(0, num_experiments):
        print("##### Experiment " + str(i) + " #####")
        print()

        X_stability_shuffled, y_stability_shuffled = shuffle(X_stability, y_stability,
                                                             random_state=i,
                                                             n_samples=int(np.round(0.8 * stability_data.shape[0])))

        print("Re-fitting best estimator...")
        print()
        best_estimator.fit(X_stability_shuffled, y_stability_shuffled)

        msk1 = np.repeat(False, len(variable_names))
        msk1[best_estimator.named_steps["variance"].get_support()] = best_estimator.named_steps[fs_step_name].get_support()

        msk2 = np.logical_and(best_estimator.named_steps["variance"].get_support(), msk1)

        selected_features = np.zeros(len(variable_names))
        selected_features[msk2] = 1

        feature_ranking[:, i] = selected_features

        if classifier_step_name != "knn" or (classifier_step_name == "knn" and fs_step_name == "rlr_l1"):
            selected_coefficients = np.zeros(len(variable_names))

            if classifier_step_name == "linear_svm":
                selected_coefficients[msk2] = \
                    best_estimator.named_steps[classifier_step_name].coef_[0, :]
            elif classifier_step_name == "rf":
                selected_coefficients[msk2] = \
                    best_estimator.named_steps[classifier_step_name].feature_importances_
            elif classifier_step_name == "knn":
                selected_coefficients[msk2] = \
                    best_estimator.named_steps[fs_step_name].estimator_.coef_[0, :]

            coefficients[:, i] = selected_coefficients

    print("Calculating final feature ranking")
    print()

    final_ranking = np.sum(feature_ranking, axis=1)

    save_object(feature_ranking, result_files_path + '/' + chromosome + '/feature_stability.pkl')

    if classifier_step_name != "knn" or (classifier_step_name == "knn" and fs_step_name == "rlr_l1"):
        if classifier_step_name == "linear_svm":
            mean_name = "coefficients_mean"
            abs_mean_name = "abs_coefficients_mean"
            scaled_name = "scaled_coefficients"

            save_object(coefficients, result_files_path + '/' + chromosome + '/feature_coefficients.pkl')
        elif classifier_step_name == "rf":
            mean_name = "importances_mean"
            abs_mean_name = "abs_importances_mean"
            scaled_name = "scaled_importances"

            save_object(coefficients, result_files_path + '/' + chromosome + '/feature_importances.pkl')
        elif classifier_step_name == "knn":
            mean_name = "coefficients_mean"
            abs_mean_name = "abs_coefficients_mean"
            scaled_name = "scaled_coefficients"

            save_object(coefficients, result_files_path + '/' + chromosome + '/feature_coefficients.pkl')

        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)),
                                          np.repeat(0, len(variable_names)), np.repeat(0, len(variable_names)),
                                          np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('stability', '>i4'), (mean_name, 'float64'),
                                        (abs_mean_name, 'float64'), (scaled_name, 'float64')])
    else:
        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('stability', '>i4')])

    features_info['names'] = variable_names
    features_info['stability'] = final_ranking

    if classifier_step_name != "knn" or (classifier_step_name == "knn" and fs_step_name == "rlr_l1"):
        features_info[mean_name] = np.mean(coefficients, axis=1)
        features_info[abs_mean_name] = np.mean(np.abs(coefficients), axis=1)
        features_info[scaled_name] = np.mean((coefficients - np.min(coefficients)) / (np.max(coefficients) - np.min(coefficients)), axis=1)

    with open(result_files_path + '/' + chromosome + '/general_features_info.csv', 'w') as f:
        w = csv.writer(f)

        header = list(['names', 'stability'])

        if classifier_step_name != "knn" or (classifier_step_name == "knn" and fs_step_name == "rlr_l1"):
            header += list([mean_name, abs_mean_name, scaled_name])

        w.writerow(header)
        w.writerows(features_info)

    with open(result_files_path + '/features_selected_by_stability.txt', "a") as f:
        f.write(",".join(map(str, list(np.where(final_ranking == num_experiments)[0]))) + "\n")

