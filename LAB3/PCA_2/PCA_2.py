import cv2
import os
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer

compress_edge = 50
original_faces = "faces/original/"
compressed_images = "faces/compressed/"


def compress_images():
    for _, _, f in os.walk(original_faces):
        # print(f)
        for picture in f:
            # print(picture)
            im = cv2.imread(original_faces + picture)
            im_compressed = cv2.resize(im, (compress_edge, compress_edge))
            cv2.imwrite(compressed_images + "C_" + picture, im_compressed)


def load_images():
    whole = []
    sad = []
    happy = []
    for _, _, f in os.walk(compressed_images):
        for picture in f:
            im = cv2.imread(compressed_images + picture, cv2.IMREAD_GRAYSCALE)
            if 'w' in picture:
                happy.append(im)
            elif 's' in picture:
                sad.append(im)
            else:
                print("Picture incorrect name")
    whole.extend(happy)
    whole.extend(sad)
    return whole, happy, sad


def calculate_mean_picture(whole):
    pictures_count, x, x = whole.shape
    mean = np.zeros([x, x])
    for i in range(pictures_count):
        mean += whole[i]
    mean = mean / pictures_count
    cv2.imwrite("mean.png", mean)
    return mean


def transform_array_to_vector(arr):
    to_ret = []
    r, _ = arr.shape
    for i in range(r):
        to_ret.extend(arr[i])
    to_ret = np.array(to_ret)
    return to_ret


def transform_all_to_vectors(whole_arr):
    to_ret = []
    x, _, _ = whole_arr.shape
    for i in range(x):
        to_ret.append(transform_array_to_vector(whole_arr[i]))
    to_ret = np.array(to_ret)
    return to_ret


def transform_from_vec_to_pic(vec):
    to_ret = np.zeros([compress_edge, compress_edge])
    for i in range(compress_edge):
        for j in range(compress_edge):
            to_ret[i][j] = vec[i * 50 + j]
    return to_ret


def transform_all_vec_to_pic(whole_vec):
    to_ret = []
    r, _ = whole_vec.shape
    for i in range(r):
        to_ret.append(transform_from_vec_to_pic(whole_vec[i]))
    return to_ret


def pca_on_pictures(whole, happy, sad, dimensions=None):
    whole_as_vector = transform_all_to_vectors(whole)
    pca = PCA(dimensions)
    transformed = pca.fit_transform(whole_as_vector)
    if dimensions == 5 or dimensions == 15 or dimensions == 50:
        inverse = pca.inverse_transform(transformed)
        transformed_inverse = transform_all_vec_to_pic(inverse)
        p = 0
        for pic in transformed_inverse:
            cv2.imwrite("faces/reverse_transform/reverse_D" + str(dimensions) + "_P" + str(p) + ".png", pic)
            p += 1
    if dimensions == 2:
        plt.clf()
        plt.scatter(transformed[0:len(happy), 0], transformed[0:len(happy), 1], alpha=0.5, color="red", s=30)
        plt.scatter(transformed[len(happy):len(happy) + len(sad), 0], transformed[len(happy):len(sad) + len(sad), 1],
                    alpha=0.5, color="blue", s=30)

        plt.axis('off')
        plt.savefig("faces/2dPCA.png")
    # todo: show principal components on picture
    print(pca.explained_variance_ratio_)
    plt.clf()

    plt.imshow(transformed, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig("faces/vectors/D" + str(dimensions) + ".png")


def sparse_pca_on_pictures(whole, dimensions=None):
    whole_as_vector = transform_all_to_vectors(whole)
    pca = SparsePCA(dimensions, alpha=100, n_jobs=-1, normalize_components=True, max_iter=10)
    transformed = pca.fit_transform(whole_as_vector)
    plt.clf()
    plt.imshow(transformed, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig("faces/sparsePCA/D_" + str(dimensions) + ".png")
    # print(pca.components_)


def run_pca(whole_arr, happy, sad):
    for PCA_DIM in [None, 50, 15, 5, 2]:
        print(PCA_DIM)
        pca_on_pictures(whole_arr, happy, sad, PCA_DIM)
        sparse_pca_on_pictures(whole_arr, PCA_DIM)


if __name__ == "__main__":
    compress_images()
    whole, happy, sad = load_images()
    whole_arr = np.array(whole)
    mean = calculate_mean_picture(whole_arr)
    run_pca(whole_arr, happy, sad)
