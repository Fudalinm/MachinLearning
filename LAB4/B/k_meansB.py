from random import random, sample
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt


# USING Davies-Bouldin index

def make_circle(x, y, r, n):
    points_to_ret = []
    i = 0
    while i < n:
        px = ((random() * 4 * r) % (2 * r) - r) + x
        py = ((random() * 4 * r) % (2 * r) - r) + y
        if (x - px) ** 2 + (y - py) ** 2 <= r ** 2:
            i += 1
            points_to_ret.append([px, py])
    return points_to_ret


def make_sample(data_type, r, n):
    if data_type == 'basic':
        return make_basic_sample(r, n)
    elif data_type == 'broken':
        return make_broken_sample(r, n)


def make_broken_sample(r, n):
    whole_points = []
    whole_points_clustered = []

    away = make_circle(15 * r, 15 * r, r, n)

    spindle = []
    spindle.extend(make_circle(6 * r, 6 * r, 1.5 * r, n))
    spindle.extend(make_circle(6 * r - 0.5 * r, 6 * r, r / 2, n))
    spindle.extend(make_circle(6 * r + 0.5 * r, 6 * r, r / 2, n))

    density = make_circle(2 * r, 4 * r, r, 3 * n)

    first_close = make_circle(6 * r, 3 * r, r, n)
    second_close = make_circle(6 * r, 2 * r, r, n)

    one_bigger = make_circle(2 * r, 6 * r, 1.75 * r, 2 * n)

    basic1 = make_circle(2 * r, 2 * r, r, n)
    basic2 = make_circle(4 * r, 2 * r, r, n)
    basic3 = make_circle(4 * r, 6 * r, r, n)

    whole_points.extend(away)
    whole_points.extend(spindle)
    whole_points.extend(density)
    whole_points.extend(first_close)
    whole_points.extend(second_close)
    whole_points.extend(one_bigger)
    whole_points.extend(basic1)
    whole_points.extend(basic2)
    whole_points.extend(basic3)

    whole_points_clustered.append(away)
    whole_points_clustered.append(spindle)
    whole_points_clustered.append(density)
    whole_points_clustered.append(first_close)
    whole_points_clustered.append(second_close)
    whole_points_clustered.append(one_bigger)
    whole_points_clustered.append(basic1)
    whole_points_clustered.append(basic2)
    whole_points_clustered.append(basic3)

    return whole_points, whole_points_clustered


def make_basic_sample(r, n):
    whole_points = []
    whole_points_clustered = []

    for x in [2 * r, 6 * r, 10 * r]:
        for y in [2 * r, 6 * r, 10 * r]:
            circle_points = make_circle(x, y, r, n)
            whole_points.extend(circle_points)
            whole_points_clustered.append(circle_points)
    return whole_points, whole_points_clustered


def scatter_sample(name, whole_points):
    plt.clf()
    whole_points_arr = np.array(whole_points)
    plt.scatter(whole_points_arr[0:len(whole_points_arr), 0], whole_points_arr[0:len(whole_points_arr), 1], s=2)

    plt.savefig(name)
    plt.clf()


def forgy(whole_data, clusters):
    return sample(whole_data, clusters)


# name, mean, standard deviation
def print_fig(fig_name, scores):
    labels = []
    standard_deviations = []
    means = []

    for name, mean, standard_deviation in scores:
        labels.append(name)
        standard_deviations.append(standard_deviation)
        means.append(mean)

    plt.clf()
    width = 0.44
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, means, width, label='Mean_result')
    rects2 = ax.bar(x + width / 2, standard_deviations, width, label='Standard_deviation')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    for tick in ax.get_xticklabels():
        tick.set_rotation(25)

    fig.tight_layout()
    plt.savefig(fig_name)


def run1():
    n_clusters = 9
    repeat = 10

    for data_type in ['basic', 'broken']:
        observations, clustered = make_sample(data_type, 20, 100)
        scatter_sample(data_type + '_sample.png', observations)
        # name, mean, standard deviation
        score_to_present = []
        for init_type in ['k-means++', 'random', 'forgy']:
            for n in [2, 5, 15, 30]:
                whole_score = []
                if init_type == 'forgy':
                    for i in range(repeat):
                        centers = np.array(forgy(observations, n_clusters))
                        kmeans = KMeans(n_clusters=n_clusters, init=centers, max_iter=n, n_init=1, n_jobs=-1). \
                            fit(observations)
                        predicted = kmeans.predict(observations)
                        centers = kmeans.cluster_centers_
                        score = davies_bouldin_score(observations, predicted)
                        whole_score.append(score)
                else:
                    # n_init 1 because we want to have full control of how many and results
                    for i in range(repeat):
                        kmeans = KMeans(n_clusters=n_clusters, init=init_type, max_iter=n, n_init=1, n_jobs=-1). \
                            fit(observations)
                        predicted = kmeans.predict(observations)
                        centers = kmeans.cluster_centers_
                        score = davies_bouldin_score(observations, predicted)
                        whole_score.append(score)
                print(whole_score)
                whole_score = np.array(whole_score)
                score_to_present.append([init_type + '_' + str(n), np.mean(whole_score), np.std(whole_score)])

        print(score_to_present)

        print_fig(data_type + '_score.png', score_to_present)


def run2():
    # TODO:  find elbow
    score_to_present = []
    repeat = 10
    for n_clusters in [2, 3, 5, 7, 9, 11, 14, 19]:
        for data_type in ['basic', 'broken']:
            observations, clustered = make_sample(data_type, 20, 100)
            for init_type in ['k-means++', 'random', 'forgy']:
                score = []
                if init_type == 'forgy':
                    for i in range(repeat):
                        centers = np.array(forgy(observations, n_clusters))
                        kmeans = KMeans(n_clusters=n_clusters, init=centers, max_iter=15, n_init=1, n_jobs=-1). \
                            fit(observations)
                        predicted = kmeans.predict(observations)
                        l = davies_bouldin_score(observations, predicted)
                        score.append(l)
                else:
                    # n_init 1 because we want to have full control of how many and results
                    for i in range(repeat):
                        kmeans = KMeans(n_clusters=n_clusters, init=init_type, max_iter=15, n_init=2, n_jobs=-1). \
                            fit(observations)
                        predicted = kmeans.predict(observations)
                        l = davies_bouldin_score(observations, predicted)
                        score.append(l)

                score = np.array(score)
                score_to_present.append([data_type, n_clusters, init_type, np.mean(score), np.std(score)])
            print(score_to_present)
    print_run2_result(score_to_present)


# presenting only means
# score_to_present.append([data_type, n_clusters, init_type, np.mean(score), np.std(score)])
def print_run2_result(scores):
    basic = []
    broken = []
    for data_type, n_clusters, init_type, mean, std in scores:
        if data_type == 'basic':
            basic.append([n_clusters, init_type, mean, std])
        else:
            broken.append([n_clusters, init_type, mean, std])

    """ print basic """
    basic.sort(key=lambda l: l[0])
    # 'k-means++', 'random', 'forgy'

    plus_plus_init = []
    random_init = []
    forgy_init = []
    labels = []
    for n_clusters, init_type, mean, _ in basic:
        if init_type == 'k-means++':
            plus_plus_init.append(mean)
            labels.append('basic_' + str(n_clusters))
        elif init_type == 'random':
            random_init.append(mean)
        else:
            forgy_init.append(mean)

    plt.clf()
    width = 0.3
    x = np.arange(len(labels))
    fig, ax = plt.subplots()

    print(basic)
    print(plus_plus_init)
    print(random_init)
    print(forgy_init)

    rects1 = ax.bar(x - width / 2, plus_plus_init, width / 3, label='k_plus_plus')
    rects1 = ax.bar(x, random_init, width / 3, label='random')
    rects2 = ax.bar(x + width / 2, forgy_init, width / 3, label='forgy')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    for tick in ax.get_xticklabels():
        tick.set_rotation(25)

    fig.tight_layout()
    plt.savefig('basic_cluster_compare.png')

    """ print broken """
    broken.sort(key=lambda l: l[0])
    # 'k-means++', 'random', 'forgy'

    plus_plus_init = []
    random_init = []
    forgy_init = []
    labels = []
    for n_clusters, init_type, mean, _ in broken:
        if init_type == 'k-means++':
            plus_plus_init.append(mean)
            labels.append('basic_' + str(n_clusters))
        elif init_type == 'random':
            random_init.append(mean)
        else:
            forgy_init.append(mean)

    plt.clf()
    width = 0.3
    x = np.arange(len(labels))
    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, plus_plus_init, width / 3, label='k_plus_plus')
    rects1 = ax.bar(x, random_init, width / 3, label='random')
    rects2 = ax.bar(x + width / 2, forgy_init, width / 3, label='forgy')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    for tick in ax.get_xticklabels():
        tick.set_rotation(25)

    fig.tight_layout()
    plt.savefig('broken_cluster_compare.png')

    # print broken


if __name__ == "__main__":
    run1()
    #run2()
