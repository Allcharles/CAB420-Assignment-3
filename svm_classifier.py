import numpy as np
import csv
import glob
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

try:
    import cPickle as pickle
except:
    import pickle


def read_data(path, filelist):
    print("Loading classifications")
    family_lables = np.load(
        path + filelist[1].strip())
    feature_vectors = []
    print(filelist[1].strip())

    print("Loading features")
    # Load melspectrogram and chromacqt
    for feature in filelist[2:]:
        feature_vectors.append(
            np.load(path + feature.strip()))
    feature_vectors[0] = np.reshape(
        feature_vectors[0], (feature_vectors[0].shape[0], 128, 126))

    print("Aggregating features")
    aggregated_vectors = []
    for vector in feature_vectors:
        aggregated_vectors.append(np.array(vector).mean(2))
        aggregated_vectors.append(np.array(vector).std(2))

    X = np.concatenate(tuple(aggregated_vectors), axis=1)
    y = np.vectorize(str)(family_lables)
    return [X, y]


def svm_classify(pathToTrainingFeatures, pathToTestingFeatures):
    scaler = StandardScaler()
    batches = []
    with open(pathToTrainingFeatures + "filelist.csv") as csv_file:
        reader = csv.reader(csv_file)
        batches = [row for row in reader]
        print(batches)
    batch = batches[0]

    testing_data = []
    with open(pathToTestingFeatures + "filelist.csv") as csv_file:
        reader = csv.reader(csv_file)
        testing_data = [row for row in reader][0]

    print("Loading test data")
    [x_test, y_test] = read_data(pathToTestingFeatures, testing_data)

    print("Loading batch data")
    [x_train, y_train] = read_data(pathToTrainingFeatures, batch)

    print("Standardising data")
    start_time = time.time()
    pca = PCA(.90)  # Maintain 90% of variance
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    print("Running PCA")
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    end_time = time.time()
    print("Standardisation Time Taken: {}s".format(end_time - start_time))
    print("PCA Variance Ratio:")
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)

    print("Performing cross validation")
    # SVM incremental learning
    model = SVC(max_iter=1000, degree=3, C=0.1, kernel='poly', gamma='auto')
    model.fit(x_train, y_train)

    print('Testing on unknown data')
    print('Generating model predictions')
    outputs = model.predict(x_test)
    # outputs = tune_model.predict(x_test)
    print('Test results')
    print('Accuracy score: ')
    print(accuracy_score(y_test, outputs))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, outputs))
    print('Classification report: ')
    print(classification_report(y_test, outputs))

    # Save the classifier
    with open(pathToTrainingFeatures + "/svm_classifier.pk1", 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == '__main__':
    # svm_classify("data20190604T162847419276validation/",
    #             "data20190604T160328250448testing/")
    #svm_classify("data1_000_training/", "datatesting/")
    #svm_classify("data2_500_training/", "datatesting/")
    #svm_classify("data5_000_training/", "datatesting/")
    svm_classify("data10_000_training/", "datatesting/")
    #svm_classify("data20_000_training/", "datatesting/")
    #svm_classify("data30_000_training/", "datatesting/")
