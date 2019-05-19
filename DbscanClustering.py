import random
import codecs

import math
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

    db = DBSCAN(metric='precomputed', min_samples=DBSCAN_MIN_SIZE, eps=DBSCAN_EPSILON).fit(distance_matrix)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    example_solution_cluster = labels[labels.size-1]

    # with open('./res/exampleSolutionClusters', mode='a', encoding='utf-8') as output:
    #     clusters = ""
    #     for c in labels:
    #         clusters += str(c) + "/"
    #     print(file_name[:-4] + ";" + clusters, file=output)

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

    return labels

    # project(distance_matrix, labels, output_file_name)


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
    df = pd.read_pickle(path + "distances/" + file_name)
    return cluster_dbscan(df, file_name, output_file_name)


path = "./res/solutiongroups/dbscan/correct_v3/"
DBSCAN_MIN_SIZE = 3
DBSCAN_EPSILON = 2

# analyze_file("40.pkl", '47')


def print_clusters(clusters_c, ii):
    solution_cluster = clusters_c[clusters_c.size - 1]

    data = {}
    for c in clusters_c:
        if not data.__contains__(c):
            data[c] = 1
        else:
            data[c] += 1

    return data
    # labels = []
    # values = []
    #
    # solution_cluster_count = data[solution_cluster]
    #
    # for l in sorted(data.keys()):
    #     labels.append(l)
    #     values.append(data[l])
    #
    # values = sorted(values, reverse=True)
    # colors = []
    #
    # outliers_count = data[-1]
    #
    # for v in values:
    #     if v == solution_cluster_count:
    #         colors.append("green")
    #     else:
    #         if v == outliers_count:
    #             colors.append("red")
    #         else:
    #             colors.append("black")
    #
    # print(data)
    # print(values)
    # print(colors)
    #
    # n_groups = values.__len__()
    #
    # fig, ax = plt.subplots()
    #
    # index = np.arange(n_groups)
    # bar_width = 0.35
    #
    # opacity = 0.4
    # error_config = {'ecolor': '0.3'}
    #
    # rects1 = ax.bar(index, values, bar_width,
    #                 alpha=opacity, color=colors, error_kw=error_config,
    #                 label='Men')
    #
    # ax.set_xlabel('Group')
    # ax.set_ylabel('Scores')
    # ax.set_title('Scores by group and gender')
    #
    # fig.tight_layout()
    ## plt.show()
    # plt.savefig('./res/newfigs/' + str(ii) + ".svg")


def print_data_for_all_tasks(values):
    all_groups = []

    i = -1
    finish = False
    while not finish:
        finish = True
        group = []

        for task_id in values:
            task_data = values[task_id]
            try:
                group.append(task_data[i])
                finish = False
            except Exception as e:
                group.append(0)
        if not finish:
            all_groups.append(group)
        i += 1

    return all_groups


def plot_data(all_groups, solution_clusters):

    barWidth = 1

    max = float(0)

    task_sums = []

    for i in range(0, all_groups[0].__len__()):
        sum = 0
        for j in range(all_groups.__len__()):
            sum += all_groups[j][i]
        if sum > max:
            max = sum
        task_sums.append(sum)

    normalized_groups = []

    for group in all_groups:
        normalized_group = []
        for count, task_sum in zip(group, task_sums):
            ratio = max / task_sum
            normalized_group.append(count * ratio)
        normalized_groups.append(normalized_group)

    for q in all_groups:
        print(q)
    for q in normalized_groups:
        print(q)
    X = []
    for i in range(0, normalized_groups[0].__len__()):
        X.append(i)

    bottom = []

    colors = ['#bbbb00', '#00bbbb', '#bb00bb']
    outliers_color = '#ff0000'
    example_color = '#00ff00'

    for i in range(0, normalized_groups.__len__()):
        line_colors = []
        for j in range(0, len(solution_clusters)):
            if solution_clusters[j]+1 == i:
                if i == 0:
                    line_colors.append('#aaaa00')
                else:
                    line_colors.append(example_color)
            else:
                if i == 0:
                    line_colors.append('#660000')
                else:
                    line_colors.append('#555555')

        heights = []
        for count in task_sums:
            heights.append(math.log(count, 2))

        positions = []
        diff = 0
        for j in range(0, len(X)):
            positions.append(X[j] + diff + 2)
            diff += heights[j]

        if i == 0:
            bottom = normalized_groups[i]
            plt.barh(X, normalized_groups[i], color=line_colors, edgecolor='white', height=barWidth)
        else:
            plt.barh(X, normalized_groups[i], left=bottom, color=line_colors, edgecolor='white', height=barWidth)
            new_bottom = []
            for old, current in zip(bottom, normalized_groups[i]):
                new_bottom.append(old + current)
            bottom = new_bottom
        print(bottom)


    # Show graphic
    plt.show()


if __name__ == '__main__':
    values = {}
    solution_clusters = []

    for i in range(1, 41):
        try:
            clusters = analyze_file(str(i) + '.pkl', str(i))
            print("----------")
            print("Task:", i, "Example solution cluster:", clusters[clusters.size-1])
            solution_clusters.append(clusters[clusters.size-1])
            data = print_clusters(clusters, i)
            values[i] = data

        except Exception as e:
            print(e)
            pass

    print(values)
    print("#############################")
    print(solution_clusters)
    print("--------------------------------------")
    all_groups = print_data_for_all_tasks(values)

    plot_data(all_groups, solution_clusters)



