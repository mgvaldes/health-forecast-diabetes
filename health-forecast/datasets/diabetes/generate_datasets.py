import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedShuffleSplit
from utils_functions import load_dataset


def generate_genomic_datasets():
    last_epidemiological_col_index = 255
    dataset_type = "genomic"
    dataset_subtypes = ["D2_vs_H", "D2_vs_CD2F", "CD2W_vs_CD2F"]

    for dataset_subtype in dataset_subtypes:
        os.makedirs(os.getcwd() + '/' + dataset_type + '/' + dataset_subtype)

        for i in range(1, 23):
            chromosome = "chr" + str(i)

            print("################################", chromosome, "################################")
            print()

            print("Loading dataset...")
            print()
            df = load_dataset(os.getcwd() + '/genomic_epidemiological/' + dataset_subtype + '/' + chromosome + '/' + dataset_subtype + '.csv')

            print("Dataset:", df.shape)
            print()

            print("Loading variable names...")
            print()
            with open(os.getcwd() + '/genomic_epidemiological/' + dataset_subtype + '/' + chromosome + '/' + dataset_subtype + '.csv', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    variable_names = np.array(list(row))
                    break

            print("Last epidemiological column:", variable_names[last_epidemiological_col_index])
            print()

            all_cols_without_epidemiological_cols_indexes = [0] + list(np.arange(last_epidemiological_col_index + 1, df.shape[1]))
            print("Deleting column 'ENFERMEDADES_DIABETES_T2DM' from dataset and from variable names. Keeping", len(all_cols_without_epidemiological_cols_indexes), "columns.")
            print()

            dataset_dir = os.getcwd() + '/' + dataset_type + '/' + dataset_subtype + '/' + chromosome

            os.makedirs(dataset_dir)

            print("Saving", dataset_subtype, "dataset.")
            print()
            with open(dataset_dir + '/' + dataset_subtype + '.csv', 'w') as f:
                w = csv.writer(f)

                w.writerow(variable_names[all_cols_without_epidemiological_cols_indexes])
                w.writerows(df[:, all_cols_without_epidemiological_cols_indexes])


def generate_genomic_epidemiological_datasets():
    dataset_type = "genomic_epidemiological"

    for i in range(2, 23):
        chromosome = "chr" + str(i)

        print("################################", chromosome, "################################")
        print()

        print("Loading dataset...")
        print()
        df = load_dataset(os.getcwd() + '/' + dataset_type + '/' + chromosome + '/' + dataset_type + '.csv')

        print("Dataset:", df.shape)
        print()

        print("Loading variable names...")
        print()
        with open(os.getcwd() + '/genomic_epidemiological/' + chromosome + '/genomic_epidemiological.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                variable_names = np.array(list(row))
                break

        diabetes_type2_col_index = [i for i, x in enumerate(variable_names) if x in ["ENFERMEDADES_DIABETES_T2DM"]]

        diabetes_type2_column_name = variable_names[diabetes_type2_col_index]
        diabetes_type2_column = df[:, diabetes_type2_col_index]
        print("Saving column 'ENFERMEDADES_DIABETES_T2DM' with", diabetes_type2_column.shape, "elements.")
        print()

        # Filter to keep all columns except for diabetes type2 column
        all_cols_without_diabetes_type2_col_indexes = list(np.arange(0, diabetes_type2_col_index[0])) + list(np.arange(diabetes_type2_col_index[0] + 1, df.shape[1]))
        print("Deleting column 'ENFERMEDADES_DIABETES_T2DM' from dataset and from variable names. Keeping", len(all_cols_without_diabetes_type2_col_indexes), "columns.")
        print()

        df = df[:, all_cols_without_diabetes_type2_col_indexes]
        print("Dataset shape without 'ENFERMEDADES_DIABETES_T2DM':", df.shape)
        print()

        variable_names = variable_names[all_cols_without_diabetes_type2_col_indexes]
        print("Variable names shape without 'ENFERMEDADES_DIABETES_T2DM':", variable_names.shape)
        print()

        # Move diabetes type2 column to be first column of dataset as target
        df = np.column_stack((diabetes_type2_column, df))
        print("Dataset shape with 'ENFERMEDADES_DIABETES_T2DM' in front:", df.shape)
        print()

        variable_names = np.hstack((diabetes_type2_column_name, variable_names))
        print("Variable names shape with 'ENFERMEDADES_DIABETES_T2DM' in front:", variable_names.shape)
        print()

        ####### Generating D2_vs_H dataset
        dataset_subtype = "D2_vs_H"

        ####### Diabetes Type2
        diabetes_type2_row_indexes = np.where(df[:, 0] == 1)[0]
        print("Number of diabetes type2 individuals:", len(diabetes_type2_row_indexes))
        print()

        # All columns starting with 'ENFERMEDADES' = 0
        diseases_col_names_indexes = [col_name for col_name in variable_names if 'ENFERMEDADES' in col_name]
        diseases_col_indexes = [i for i, x in enumerate(variable_names) if x in diseases_col_names_indexes]

        ####### Healthy
        healthy_row_indexes = np.where(np.all(np.logical_not(df[:, diseases_col_indexes]), axis=1) == True)[0]
        print("Number of healthy individuals:", len(healthy_row_indexes))
        print()

        ####### Check if sets of Diabetes Type2 and Healthy intersect
        print("Check if diabetes type2 and healthy rows intersect:", np.intersect1d(diabetes_type2_row_indexes, healthy_row_indexes))
        print()

        diabetes_col_index = [i for i, x in enumerate(variable_names) if x in ["ENFERMEDADES_DIABETES"]]

        print("Deleting column 'ENFERMEDADES_DIABETES' from dataset and from variable names.")
        print()
        # Deleting column "Diabetes"
        df = np.delete(df, diabetes_col_index, axis=1)
        variable_names = np.delete(variable_names, diabetes_col_index)

        print("Dataset shape:", df.shape)
        print()

        print("Variable names shape:", variable_names.shape)
        print()

        diabetes_type1_col_index = [i for i, x in enumerate(variable_names) if x in ["ENFERMEDADES_DIABETES_T1DM"]]

        print("Deleting column 'ENFERMEDADES_DIABETES_T1DM' from dataset and from variable names.")
        print()
        # Deleting column "Diabetes Type 1"
        df = np.delete(df, diabetes_type1_col_index, axis=1)
        variable_names = np.delete(variable_names, diabetes_type1_col_index)

        print("Dataset shape:", df.shape)
        print()

        print("Variable names shape:", variable_names.shape)
        print()

        dataset_rows = list(diabetes_type2_row_indexes) + list(healthy_row_indexes)

        # Filter only columns i'm interested
        print("Building", dataset_subtype, "dataset.")
        print()
        D2_vs_H_dataset = df[dataset_rows, :]

        print("Shape:", D2_vs_H_dataset.shape)
        print()

        dataset_dir = os.getcwd() + '/' + dataset_type + '/' + dataset_subtype + '/' + chromosome

        os.makedirs(dataset_dir)

        print("Saving", dataset_subtype, "dataset.")
        print()
        with open(dataset_dir + '/' + dataset_subtype + '.csv', 'w') as f:
            w = csv.writer(f)

            w.writerow(variable_names)
            w.writerows(D2_vs_H_dataset)

        dataset_subtype = "D2_vs_CD2F"

        comorbidity_col_indexes = [i for i, x in enumerate(variable_names) if x in ["ENFERMEDADES_HTA", "ENFERMEDADES_HIPERCOLESTEROLEMIA", "ENFERMEDADES_ICTUS", "ENFERMEDADES_INFARTO", "ENFERMEDADES_ANGI-1"]]

        some_comorbidity_row_indexes = np.where(np.any(df[:, comorbidity_col_indexes], axis=1) == True)[0]

        no_diabetes_type2_row_indexes = np.where(df[:, 0] == 0)[0]

        some_comorbidity_no_diabetes_type2_row_indexes = np.intersect1d(no_diabetes_type2_row_indexes, some_comorbidity_row_indexes)
        print("Number of some comorbidity but no diabetes type2 individuals:", len(some_comorbidity_no_diabetes_type2_row_indexes))
        print()

        ####### Check if sets of Diabetes Type2 and Some Comorbidity but No Diabetes Type2 intersect
        print("Check if diabetes type2 and some comorbidity but no diabetes type2 rows intersect:", np.intersect1d(diabetes_type2_row_indexes, some_comorbidity_no_diabetes_type2_row_indexes))
        print()

        dataset_rows = list(diabetes_type2_row_indexes) + list(some_comorbidity_no_diabetes_type2_row_indexes)

        # Filter only columns i'm interested
        print("Building", dataset_subtype, "dataset.")
        print()
        D2_vs_CD2F_dataset = df[dataset_rows, :]

        print("Shape:", D2_vs_CD2F_dataset.shape)
        print()

        dataset_dir = os.getcwd() + '/' + dataset_type + '/' + dataset_subtype + '/' + chromosome

        os.makedirs(dataset_dir)

        print("Saving", dataset_subtype, "dataset.")
        print()
        with open(dataset_dir + '/' + dataset_subtype + '.csv', 'w') as f:
            w = csv.writer(f)

            w.writerow(variable_names)
            w.writerows(D2_vs_CD2F_dataset)

        dataset_subtype = "CD2W_vs_CD2F"

        some_comorbidity_with_diabetes_type2_row_indexes = np.intersect1d(diabetes_type2_row_indexes, some_comorbidity_row_indexes)
        print("Number of some comorbidity and with diabetes type2 individuals:", len(some_comorbidity_with_diabetes_type2_row_indexes))
        print()

        ####### Check if sets of Some Comorbidity with Diabetes Type2 and Some Comorbidity but No Diabetes Type2 intersect
        print("Check if some comorbidity with diabetes type2 and some comorbidity but no diabetes type2 rows intersect:", np.intersect1d(some_comorbidity_with_diabetes_type2_row_indexes, some_comorbidity_no_diabetes_type2_row_indexes))
        print()

        dataset_rows = list(some_comorbidity_with_diabetes_type2_row_indexes) + list(some_comorbidity_no_diabetes_type2_row_indexes)

        # Filter only columns i'm interested
        print("Building", dataset_subtype, "dataset.")
        print()
        CD2W_vs_CD2F_dataset = df[dataset_rows, :]

        print("Shape:", CD2W_vs_CD2F_dataset.shape)
        print()

        dataset_dir = os.getcwd() + '/' + dataset_type + '/' + dataset_subtype + '/' + chromosome

        os.makedirs(dataset_dir)

        print("Saving", dataset_subtype, "dataset.")
        print()
        with open(dataset_dir + '/' + dataset_subtype + '.csv', 'w') as f:
            w = csv.writer(f)

            w.writerow(variable_names)
            w.writerows(CD2W_vs_CD2F_dataset)


def generate_train_test_stability_datasets():
    # dataset_types = ["genomic", "genomic_epidemiological"]
    # dataset_subtypes = ["D2_vs_H", "D2_vs_CD2F", "CD2W_vs_CD2F"]
    dataset_types = ["genomic_epidemiological"]
    dataset_subtypes = ["D2_vs_H"]

    for dataset_type in dataset_types:
        for dataset_subtype in dataset_subtypes:
            for i in range(16, 23):
                chromosome = "chr" + str(i)

                print("################################", chromosome, "################################")
                print()

                print("Loading dataset...")
                print()
                df = load_dataset(os.getcwd() + '/' + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/' + dataset_subtype + '.csv')

                print("Dataset:", df.shape)
                print()

                print("Loading variable names...")
                print()
                with open(os.getcwd() + '/' + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/' + dataset_subtype + '.csv', 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:
                        variable_names = np.array(list(row))
                        break

                print("Spliting dataset into train and test...")
                print()
                train_indexes, test_indexes = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=6547891).split(df[:, 1:], df[:, 0]))[0]

                print("Spliting train into and train and stability...")
                print()
                train_train_indexes, train_stability_indexes = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=5413697).split(df[train_indexes, 1:], df[train_indexes, 0]))[0]

                dataset_dir = os.getcwd() + '/' + dataset_type + '/' + dataset_subtype + '/' + chromosome

                print("Saving", dataset_subtype, "training dataset.")
                print()
                with open(dataset_dir + '/train_train.csv', 'w') as f:
                    w = csv.writer(f)

                    w.writerow(variable_names)
                    w.writerows(df[train_indexes, :][train_train_indexes, :])

                print("Saving", dataset_subtype, "stability dataset.")
                print()
                with open(dataset_dir + '/train_stability.csv', 'w') as f:
                    w = csv.writer(f)

                    w.writerow(variable_names)
                    w.writerows(df[train_indexes, :][train_stability_indexes, :])

                print("Saving", dataset_subtype, "test dataset.")
                print()
                with open(dataset_dir + '/test.csv', 'w') as f:
                    w = csv.writer(f)

                    w.writerow(variable_names)
                    w.writerows(df[test_indexes, :])


def generate_final_train_test_stability_datasets(main_path, dataset_type, dataset_subtype, sampling, sampling_timing,
                                                 fs_step_name, classifier_step_name, fs_type):
    last_epidemiological_col_index = 255

    result_files_path = main_path + fs_type + '/' + fs_step_name + '/classifiers/' + classifier_step_name + '/' + sampling_timing + \
                        '/' + sampling + '/' + dataset_type + '/' + dataset_subtype

    str_chromosome_feature_indexes = [line.rstrip('\n') for line in open(result_files_path + '/features_selected_by_stability.txt')]
    dict_chromosome_feature_indexes = dict()
    epidemiological_features_indexes = np.array([])
    genomic_features_size = 0
    train_train_datasets_size = 844
    test_datasets_size = 517

    for i in range(1, 23):
        chromosome = "chr" + str(i)

        feature_indexes = np.array(map(int, str_chromosome_feature_indexes[i].split(',')))

        epidemiological_features_indexes = np.union1d(epidemiological_features_indexes,
                                                      np.where(feature_indexes <= last_epidemiological_col_index)[0])

        dict_chromosome_feature_indexes[chromosome] = np.where(feature_indexes > last_epidemiological_col_index)[0]

        genomic_features_size += len(dict_chromosome_feature_indexes[chromosome])

    new_train_data = np.zeros((train_train_datasets_size, 1))
    epidem_cols_added = False
    new_train_variable_names = []

    for i in range(1, 23):
        chromosome = "chr" + str(i)

        with open(os.getcwd() + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/train_train.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                train_variable_names = np.array(list(row))
                break

        print(chromosome, "train variable names size:", len(train_variable_names[1:]))
        print()

        train_data = load_dataset(os.getcwd() + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/train_train.csv')

        X_train = train_data[:, 1:]
        print(chromosome, "X_train shape:", X_train.shape)
        print()

        if not epidem_cols_added:
            new_train_variable_names.append(train_variable_names[0])

            train_variable_names = train_variable_names[1:]

            new_train_data[:, 0] = train_data[:, 0]

            new_train_variable_names.append(train_variable_names[epidemiological_features_indexes])

            new_train_data = np.column_stack((new_train_data, X_train[:, epidemiological_features_indexes]))

            epidem_cols_added = True

        new_train_variable_names.append(train_variable_names[dict_chromosome_feature_indexes[chromosome]])

        new_train_data = np.column_stack((new_train_data, X_train[:, dict_chromosome_feature_indexes[chromosome]]))

    print("New train dataset shape:", new_train_data.shape)
    print("New train variable names size:", len(new_train_variable_names))

    print("Saving", dataset_subtype, "new training dataset.")
    print()
    with open(main_path + dataset_type + '/' + dataset_subtype + '/final_train.csv', 'w') as f:
        w = csv.writer(f)

        w.writerow(new_train_variable_names)
        w.writerows(new_train_data)

    new_test_data = np.zeros((test_datasets_size, 1))
    epidem_cols_added = False
    new_test_variable_names = []

    for i in range(1, 23):
        chromosome = "chr" + str(i)

        with open(os.getcwd() + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                test_variable_names = np.array(list(row))
                break

        print(chromosome, "test variable names size:", len(test_variable_names[1:]))
        print()

        test_data = load_dataset(os.getcwd() + dataset_type + '/' + dataset_subtype + '/' + chromosome + '/test.csv')

        X_test = test_data[:, 1:]
        print(chromosome, "X_test shape:", X_test.shape)
        print()

        if not epidem_cols_added:
            new_test_variable_names.append(test_variable_names[0])

            test_variable_names = test_variable_names[1:]

            new_test_data[:, 0] = test_data[:, 0]

            new_test_variable_names.append(test_variable_names[epidemiological_features_indexes])

            new_test_data = np.column_stack((new_test_data, X_test[:, epidemiological_features_indexes]))

            epidem_cols_added = True

        new_test_variable_names.append(test_variable_names[dict_chromosome_feature_indexes[chromosome]])

        new_test_data = np.column_stack((new_test_data, X_test[:, dict_chromosome_feature_indexes[chromosome]]))

    print("New test dataset shape:", new_test_data.shape)
    print("New test variable names size:", len(new_test_variable_names))

    print("Saving", dataset_subtype, "new test dataset.")
    print()
    with open(main_path + dataset_type + '/' + dataset_subtype + '/final_test.csv', 'w') as f:
        w = csv.writer(f)

        w.writerow(new_train_variable_names)
        w.writerows(new_train_data)

if __name__ == '__main__':
    # generate_genomic_epidemiological_datasets()

    # generate_genomic_datasets()

    # generate_train_test_stability_datasets()

    disease = "diabetes"

    main_path = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast-project-diabetes/health-forecast/fs/' + disease + '/'

    sampling_timings = ["sampling_after_fs"]
    sampling_types = ["raw"]
    dataset_types = ["genomic_epidemiological"]
    dataset_sub_types = ["D2_vs_H"]
    fs_step_names = ["anova"]
    classifier_step_names = ["linear_svm"]
    fs_types = ["filter"]

    for fs_step_name in fs_step_names:
        for classifier_step_name in classifier_step_names:
            for sampling_timing in sampling_timings:
                for sampling in sampling_types:
                    for dataset_type in dataset_types:
                        for dataset_subtype in dataset_sub_types:
                            for fs_type in fs_types:
                                generate_final_train_test_stability_datasets(main_path, dataset_type, dataset_subtype,
                                                                             sampling, sampling_timing, fs_step_name,
                                                                             classifier_step_name, fs_type)