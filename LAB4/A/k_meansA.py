import cv2
from pyclustering.cluster import kmedoids
import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pyclustering.cluster.silhouette import silhouette

image_path = 'balonik_3.png'  # 3:2\
new_width = 90
new_height = 60

k = 8
# 13
cluster_colors = []


def generate_colors(n):
    rgb_values = []
    hex_values = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for _ in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        r_hex = hex(r)[2:]
        g_hex = hex(g)[2:]
        b_hex = hex(b)[2:]
        hex_values.append('#' + r_hex + g_hex + b_hex)
        rgb_values.append([r, g, b])
    return rgb_values


def prepare_image():
    im = cv2.imread(image_path)
    im_resized = cv2.resize(im, (new_width, new_height))
    return im_resized


def remove_duplicates(im):
    without_duplicates = []
    with_duplicates = []

    nx, ny, _ = im.shape
    for x in range(nx):
        for y in range(ny):
            with_duplicates.append([im[x][y][0], im[x][y][1], im[x][y][2]])
            if list([im[x][y][0], im[x][y][1], im[x][y][2]]) not in without_duplicates:
                without_duplicates.append([im[x][y][0], im[x][y][1], im[x][y][2]])
    return without_duplicates, with_duplicates


# if would be run multiple times we will use random centers
def draw_centers(observations):
    global k
    copy_observations = observations.copy()
    copy_observations.sort(key=lambda x: x[0] * 10000 + x[1] * 100 + x[2])
    centers = []
    centers_indexes = []
    step_length = int(len(copy_observations) / (k + 1)) + 1

    for i in range(step_length, len(copy_observations), step_length):
        centers.append(copy_observations[i])
        centers_indexes.append(i)
    # print("#####")
    # print(k)
    # print(len(centers))
    # print("#####")
    return centers, centers_indexes


def draw_square(im, x, y, center_color, round_color):
    hex_color_center = '#%02x%02x%02x' % (center_color[0], center_color[1], center_color[2])
    hex_round_color = '#%02x%02x%02x' % (round_color[0], round_color[1], round_color[2])
    plt.scatter(x, y, c=hex_color_center, edgecolors=hex_round_color, s=5)


def print_maps(data, clusters, medoids, name, name_pointers, name_silhouette_score):
    map = []
    plt.clf()
    cluster_colors = generate_colors(len(clusters))

    pca_for_map = PCA(n_components=2)
    pca_for_map.fit(data)
    data_after_pca = pca_for_map.transform(data)

    for i in range(len(clusters)):
        print(i)
        cluster = clusters[i]
        cluster_color = cluster_colors[i]
        for data_index in cluster:
            data_color = data[data_index]
            draw_square(map, data_after_pca[data_index][0], data_after_pca[data_index][1], data_color,
                        cluster_color)
    plt.savefig(name, dpi=400)

    for m in medoids:
        plt.scatter(data_after_pca[m][0], data_after_pca[m][1], color='red', s=6)
    plt.savefig(name_pointers, dpi=400)

    plt.clf()
    sil_score = silhouette(data, clusters).process().get_score()

    print(len(data_after_pca))
    print(data_after_pca)
    print(len(sil_score))
    print(sil_score)

    print(data_after_pca[0][0])
    print(sil_score[0])

    # for i in range(len(data_after_pca)):
    #     print(i)
    #     print(data_after_pca[i][0])
    #     print(data_after_pca[i][1])
    #     print(sil_score[i])
    #     plt.scatter(data_after_pca[i][0], data_after_pca[i][1], c=255*sil_score[i], s=3)

    sc = plt.scatter(data_after_pca[0:len(data_after_pca), 0], data_after_pca[0:len(data_after_pca), 1],
                     c=sil_score[0:len(sil_score)], s=3)

    # for m in medoids:
    #     plt.scatter(data_after_pca[m][0], data_after_pca[m][1], color='red', s=6)
    plt.colorbar(sc)
    plt.gray()
    plt.savefig(name_silhouette_score, dpi=400)


def run():
    im = prepare_image()
    without_duplicates, with_duplicates = remove_duplicates(im)
    centers_with_duplicate, indexes_duplicate = draw_centers(with_duplicates)
    centers_without_duplicates, indexes = draw_centers(without_duplicates)

    # DONE: WITH DUPLICATES
    kmedoid_duplicate = kmedoids.kmedoids(with_duplicates, indexes_duplicate)
    kmedoid_duplicate.process()
    clusters = kmedoid_duplicate.get_clusters()

    medoids = kmedoid_duplicate.get_medoids()
    print_maps(with_duplicates, clusters, medoids, "duplicates.png", "duplicates_cluster_pointer.png",
               "duplicates_silhouette_score.png")

    # DONE: WITHOUT DUPLICATES
    kmedoid_no_duplicate = kmedoids.kmedoids(without_duplicates, indexes)
    kmedoid_no_duplicate.process()
    clusters_no_duplicates = kmedoid_no_duplicate.get_clusters()

    medoids_no_duplicate = kmedoid_no_duplicate.get_medoids()
    print_maps(without_duplicates, clusters_no_duplicates, medoids_no_duplicate, "no_duplicates.png",
               "no_duplicates_cluster_pointer.png", "no_duplicates_silhouette_score.png")


if __name__ == '__main__':
    run()
