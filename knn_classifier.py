import numpy as np
import csv
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import pipes
from scipy import misc
import imageio
from sklearn.preprocessing import StandardScaler
import time

try:
    import cPickle as pickle
except:
    import pickle


def read_data(filelist,dataset_path):
    dataset = np.genfromtxt(filelist,delimiter=',', dtype=None, encoding=None)
    X = [list(row)[0] for row in dataset]
    y = [list(row)[1] for row in dataset]
    np.array(X)
    
    feature_vectors = []
    for feature in X:
        image_path = dataset_path+str(feature)
        image = imageio.imread(image_path)
        # image = image[:,:,:1]
        image.reshape(-1)
        feature_vectors.append(image.reshape(-1))

    return  feature_vectors, np.array(y)


def knn_classify(pathToTrainingFeatures, pathToTestingFeatures):
    scaler = StandardScaler()

    print("Loading train data")
    [x_train, y_train] = read_data(
        pathToTrainingFeatures + "filelist.csv", pathToTrainingFeatures)

    print("Standardising data")
    start_time = time.time()
    pca = PCA(.90)  # Maintain 90% of variance
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    pca.fit(x_train)
    x_train = pca.transform(x_train)
    end_time = time.time()
    print("Standardisation Time Taken: {}s".format(end_time - start_time))
    print("PCA Variance Ratio:")
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)

    print("Training KNN Model")
    start_time = time.time()
    model = KNN(n_neighbors=5,weights='distance',algorithm='auto',metric='euclidean')
    model.fit(x_train, y_train)
    end_time = time.time()
    print("Training KNN Time Taken: {}s".format(end_time - start_time))

    print("Loading test data")
    del x_train
    [x_test, y_test] = read_data(
        pathToTestingFeatures + "filelist.csv", pathToTestingFeatures)

    print("Standardising data")
    x_test = scaler.transform(x_test)
    x_test = pca.transform(x_test)

    print('Testing on unknown data')
    print('Generating model predictions')
    start_time = time.time()
    outputs = model.predict(x_test)
    end_time = time.time()
    print("Predictions Time Taken: {}s".format(end_time - start_time))

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
   knn_classify("/Users/christinawang/Downloads/nsynth-images/train/",  #folder for filelist and melspectrogram
               "/Users/christinawang/Downloads/nsynth-images/test/")

