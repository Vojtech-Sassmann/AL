import random
import codecs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from sklearn.cluster import DBSCAN
from adjustText import adjust_text
import pandas as pd


def cluster_dbscan(distance_matrix, file_name, output_file_name):
    """
    Provede DBSCAN shlukovani na dane matici
    """

    try:
        db = DBSCAN(metric='precomputed', min_samples=DBSCAN_MIN_SIZE, eps=DBSCAN_EPSILON).fit(distance_matrix)
    except:
        print("Using default values")
        db = DBSCAN(metric='precomputed').fit(distance_matrix)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    for cluster, solution in zip(db.labels_, distance_matrix.index.values):
        if cluster is not -1:
            with codecs.open(path + "eps" + str(DBSCAN_EPSILON) + "min" + str(DBSCAN_MIN_SIZE)  + "/" + file_name[:-4]
                             + "__" + str(cluster) + ".txt", 'a', encoding='UTF-8') as f:
                f.write(solution)
                f.write('\n')

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #

    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()

    project(distance_matrix, labels, output_file_name)


def project(distance_matrix, labels, output_file_name):
    model = PCA(n_components=2)
    results = model.fit(distance_matrix.transpose())

    plt.figure(figsize=(20, 20))
    texts = []

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    number_of_solutions = {}

    for cluster in labels:
        if number_of_solutions.__contains__(cluster):
            number_of_solutions[cluster] += 1
        else:
            number_of_solutions[cluster] = 1

    for key, value in sorted(number_of_solutions.items(), key=lambda x: x[1], reverse=True):
        print("kluster " + str(key) + " has ", str(value))
    print(number_of_solutions)

    print('Estimated number of clusters: %d' % n_clusters_)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for i in range(len(distance_matrix.index)):
        x, y = results.components_[0][i], results.components_[1][i]
        gitter_x = (random.randint(0, 2000001) + 99000000) / 100000000.0
        gitter_y = (random.randint(0, 2000001) + 99000000) / 100000000.0
        x *= gitter_x
        y *= gitter_y
        color = colors[labels[i]]
        if labels[i] == -1:
            color = [0, 0, 0, 1]
        if labels[i] != -1:
            plt.plot(x, y, "o", markerfacecolor=tuple(color), markeredgecolor='k', markersize=18)
        else:
            plt.plot(x, y, "o", markerfacecolor=tuple(color), markeredgecolor='k')
        # texts.append(plt.text(x, y, distance_matrix.index[i], size=10))

    adjust_text(texts)
    plt.savefig(output_file_name + ".svg")
    # plt.show()


def analyze_file(file_name, output_file_name):
    df = pd.read_pickle(path + "distances_l1/" + file_name)
    cluster_dbscan(df, file_name, output_file_name)


path = "./res/solutiongroups/dbscan/correct_v2/"
DBSCAN_MIN_SIZE = 5
DBSCAN_EPSILON = 1

# analyze_file("40.pkl", '47')

if __name__ == '__main__':
    for i in range(1, 75):
        try:
            analyze_file(str(i) + '.pkl', str(i))
        except:
            pass

