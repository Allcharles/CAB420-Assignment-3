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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import imageio

try:
    import cPickle as pickle
except:
    import pickle


def read_data(filelist, dataset_path):
    dataset = np.genfromtxt(filelist, delimiter=',', dtype=None, encoding=None)
    X = [list(row)[0] for row in dataset]
    y = [list(row)[1] for row in dataset]
    np.array(X)

    feature_vectors = []
    for feature in X:
        image_path = dataset_path + str(feature)
        try:
            image = imageio.imread(image_path)
            image.reshape(-1)
            feature_vectors.append(image.reshape(-1))
        except:
            print("Failed to find file: " + image_path)

    return feature_vectors, np.array(y)


def svm_classify(pathToTrainingFeatures, pathToTestingFeatures, filelist, model_name):
    print("Running Classification for " + model_name)
    scaler = StandardScaler(copy=False)

    print("Loading train data")
    [x_train, y_train] = read_data(filelist, pathToTrainingFeatures)

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

    print("Performing Grid Search")
    print("Training for: SVM_" + model_name)
    start_time = time.time()
    svm_model = SVC(max_iter=1000, shrinking=True,
                    gamma='auto', kernel='poly', C=0.1)
    svm_model.fit(x_train, y_train)
    end_time = time.time()
    print("SVM Training Time Taken: {}s".format(end_time - start_time))

    print("Training for: KNN_" + model_name)
    start_time = time.time()
    knn_model = KNN(n_neighbors=5,
                    weights='distance', algorithm='auto', metric='euclidean')
    knn_model.fit(x_train, y_train)
    end_time = time.time()
    print("KNN Training Time Taken: {}s".format(end_time - start_time))

    print("Loading test data")
    del x_train
    [x_test, y_test] = read_data(
        pathToTestingFeatures + "filelist.csv", pathToTestingFeatures)

    print("Standardising data")
    x_test = scaler.transform(x_test)
    x_test = pca.transform(x_test)

    print('Testing on unknown data')
    print('Generating SVM model predictions')
    start_time = time.time()
    svm_outputs = svm_model.predict(x_test)
    end_time = time.time()
    print("SVM Predictions Time Taken: {}s".format(end_time - start_time))

    print('Generating KNN model predictions')
    start_time = time.time()
    knn_outputs = knn_model.predict(x_test)
    end_time = time.time()
    print("KNN Predictions Time Taken: {}s".format(end_time - start_time))

    print('SVM Test results')
    print('Accuracy score: ')
    print(accuracy_score(y_test, svm_outputs))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, svm_outputs))
    print('Classification report: ')
    print(classification_report(y_test, svm_outputs))

    print('KNN Test results')
    print('Accuracy score: ')
    print(accuracy_score(y_test, knn_outputs))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, knn_outputs))
    print('Classification report: ')
    print(classification_report(y_test, knn_outputs))

    # Save the classifier
    with open("./models/SVM_" + model_name, 'wb') as fid:
        pickle.dump(svm_model, fid)

    with open("./models/KNN_" + model_name, 'wb') as fid:
        pickle.dump(knn_model, fid)


if __name__ == '__main__':
    """ svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_1_000.csv",
                 "classifier_1_000.pk1")

    svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_2_500.csv",
                 "classifier_2_500.pk1")

    svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_5_000.csv",
                 "classifier_5_000.pk1")

    svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_10_000.csv",
                 "classifier_10_000.pk1")

    svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_20_000.csv",
                 "classifier_20_000.pk1")

    svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_30_000.csv",
                 "classifier_30_000.pk1") """

    """ svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_50_000.csv",
                 "classifier_50_000.pk1")

    svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_100_000.csv",
                 "classifier_100_000.pk1")

     """
    svm_classify("D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/train/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/valid/",
                 "D:/Documents/GitHub/CAB420-Assignment-3/data/nsynth-images/filelist_200_000.csv",
                 "classifier_200_000.pk1")
