# for A
import itertools
import random
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# hyper_cube
dimensions = 7
hyper_cube_size = 100
points_per_edge = 50
in_hypersphere = 50
out_hypersphere = 100
corner_color = 'red'
edge_color = 'green'
in_sphere_color = 'blue'
out_sphere_color = 'yellow'


def generate_hypercube_corners():
    corners = list(itertools.product([0, hyper_cube_size], repeat=dimensions))
    to_ret = []
    for c in corners:
        to_ret.append(list(c))
    return to_ret


def generate_hypercube_edges():
    edges_points = []
    steps = []
    for i in range(1, points_per_edge - 1):
        steps.append((hyper_cube_size / points_per_edge) * i)
    for tuple_point in list(itertools.product([0, hyper_cube_size], repeat=dimensions)):
        list_point = list(tuple_point)
        for i in range(len(list_point)):
            if list_point[i] == 0:
                for s in steps:
                    tmp = list(list_point)
                    tmp[i] = s
                    edges_points.append(tmp)
    return edges_points


def generate_inside_hypersphere():
    hypersphere_center = []
    hypersphere_points = []
    for i in range(dimensions):
        hypersphere_center.append(hyper_cube_size / 2.0)

    for i in range(in_hypersphere):
        u = np.random.normal(0, 1, dimensions)  # an array of d normally distributed random variables
        norm = np.sum(u ** 2) ** 0.5
        r = random.random() ** (1.0 / dimensions)
        x = r * u / norm

        for j in range(len(x)):
            x[j] = x[j] * (hyper_cube_size / 2) + (hyper_cube_size / 2)
        hypersphere_points.append(list(x))
    return hypersphere_points


def generate_outside_hypersphere():
    out_points = []
    i = 0
    while i < out_hypersphere:
        tmp_point = []
        sum = 0
        for d in range(dimensions):
            r = (random.random() * hyper_cube_size) % hyper_cube_size
            tmp_point.append(r)
            sum += (r - hyper_cube_size / 2) * (r - hyper_cube_size / 2)
        if sum > (hyper_cube_size / 2) * (hyper_cube_size / 2):
            out_points.append(tmp_point)
            i += 1
        else:
            continue

    return out_points


def generate_data():
    corners = generate_hypercube_corners()
    edges = generate_hypercube_edges()
    out_hypersphere = generate_outside_hypersphere()
    in_hypersphere = generate_inside_hypersphere()
    return corners, edges, out_hypersphere, in_hypersphere


def add_to_sccater(whole, offset, dimensions, color, len, ax):
    for i in range(offset, len + offset):
        if i % 1000 == 0:
            print(i)
            print(len)
        if dimensions == 2:
            # print(i)
            # print(len)
            plt.scatter(whole[i][0], whole[i][1], color=color, s=2)
        else:
            ax.scatter(whole[i][0], whole[i][1], whole[i][2], color=color, s=2)
    offset += len
    return offset


for pca_dim in [2, 3]:
    for d in [2, 3, 4, 5, 7]:  # , 13
        if pca_dim > d:
            print("pca dimension is higher than out dimension go next")
            continue

        print("####")
        print(pca_dim)
        print(d)
        print("####")
        dimensions = d
        corners, edges, out_sphere, in_sphere = generate_data()
        whole = []
        whole.extend(corners)
        whole.extend(edges)
        whole.extend(out_sphere)
        whole.extend(in_sphere)
        print("generated")
        print("****")
        # print("*********************")
        # print("\nCORNERS" + str(corners))
        # print("\nEDGES:" + str(edges))
        # print("\nOUT_SPHERE::" + str(out_sphere))
        # print("\nIN_SPHERE:" + str(in_sphere))
        # print("\nWHOLE:" + str(whole))
        # print("*********************")

        # whole = StandardScaler.fit_transform(whole)
        print("stamp1")
        pca = PCA(n_components=pca_dim)
        print("stamp2")
        pca.fit(whole)
        print("stamp3")
        pca_whole = pca.transform(whole)
        print("stamp4")

        # TODO: jak to narysować
        offset = 0
        print("stamp5")

        plt.cla()
        plt.clf()
        ax = None
        if pca_dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        offset = add_to_sccater(pca_whole, offset, pca_dim, corner_color, len(corners), ax)
        print("step51")
        offset = add_to_sccater(pca_whole, offset, pca_dim, edge_color, len(edges), ax)
        print("step52")
        offset = add_to_sccater(pca_whole, offset, pca_dim, out_sphere_color, len(out_sphere), ax)
        print("step53")
        offset = add_to_sccater(pca_whole, offset, pca_dim, in_sphere_color, len(in_sphere), ax)
        print("stamp6")
        plt.savefig('FROM_' + str(dimensions) + "_TO_" + str(pca_dim) + ".png")
        print("stamp7")

