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
            image = image[:, :, :1]
            image.reshape(-1)
            feature_vectors.append(image.reshape(-1))
        except:
            print("Failed to find file: " + image_path)

    return feature_vectors, np.array(y)

def annot_max(x,y, ax=None):
    '''
    Annotate the max point on the graph to use for the best parameter for cross validation
    '''
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "x="+xmax + "y={:.3f}".format(ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


def svm_classify(pathToTrainingFeatures, pathToTestingFeatures):
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

    print("Performing Grid Search")
    start_time = time.time()
    model = SVC(max_iter=1000)
    para = {
        "C": [10, 1, 0.1],
        "kernel": ["linear", "poly", "rbf"],
        "gamma": [1, 0.1, 'auto'],
        "shrinking": [True, False]
    }
    tune_model = GridSearchCV(
        model, para, scoring='accuracy', cv=3, n_jobs=-1)
    tune_model.fit(x_train, y_train)
    end_time = time.time()
    print("Grid Search Time Taken: {}s".format(end_time - start_time))

    print('best parameter')
    print(tune_model.best_params_)
    means = tune_model.cv_results_['mean_test_score']
    stds = tune_model.cv_results_['std_test_score']
    param_list = []
    for mean, std, params in zip(means, stds, tune_model.cv_results_['params']):
        param_list.append(str(params))
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        print() 

    fig, ax = plt.subplots()
    ax.set_title('Cross Validation for SVM')
    print(param_list)
    ax.plot(param_list,means, '-b')
    plt.ylim(0.2,0.55)
    annot_max(param_list,means)
    plt.xlabel("Parameters")
    plt.ylabel("Mean Test Score")
    plt.grid()
    plt.show()

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
    outputs = tune_model.predict(x_test)
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
    svm_classify("/Users/christinawang/Downloads/nsynth-images/train/",  #folder for filelist and melspectrogram
               "/Users/christinawang/Downloads/nsynth-images/test/")
