import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

base_img = "PCA.png"

base_pca_vectors_fig = "basePcaVectors.png"
base_pca_transformed_fig = "basePcaTransformed.png"

cosine_fig = "cosine_fig.png"

c1_color = 'red'
c2_color = 'green'
c3_color = 'blue'
c4_color = 'yellow'

scatter_size = 1


def load_class_image(image_file):
    opened_image = cv2.imread(image_file)
    x, y, z = opened_image.shape
    # [ 36  28 237] red
    class_1 = []
    # [ 76 177  34] green
    class_2 = []
    # [232 162   0] blue
    class_3 = []
    # [  0 242 255] yellow
    class_4 = []
    for ix in range(x):
        for iy in range(y):
            if opened_image[ix][iy][0] == 36 and opened_image[ix][iy][1] == 28 and opened_image[ix][iy][2] == 237:
                class_1.append([iy, ix])
            elif opened_image[ix][iy][0] == 76 and opened_image[ix][iy][1] == 177 and opened_image[ix][iy][2] == 34:
                class_2.append([iy, ix])
            elif opened_image[ix][iy][0] == 232 and opened_image[ix][iy][1] == 162 and opened_image[ix][iy][2] == 0:
                class_3.append([iy, ix])
            elif opened_image[ix][iy][0] == 0 and opened_image[ix][iy][1] == 242 and opened_image[ix][iy][2] == 255:
                class_4.append([iy, ix])
    return class_1, class_2, class_3, class_4


def run_basic():
    c1, c2, c3, c4 = load_class_image(base_img)
    whole = []
    whole.extend(c1)
    whole.extend(c2)
    whole.extend(c3)
    whole.extend(c4)

    copy_whole = []
    copy_whole.extend(whole)
    print("*******")
    print("Points to transform: ")
    print(whole)

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(whole)
    print("Points transformed: ")
    print(transformed)
    print("*******")

    whole = np.array(whole)
    plt.scatter(whole[0:len(c1), 0], whole[0:len(c1), 1], alpha=0.2, color=c1_color, s=scatter_size)
    plt.scatter(whole[len(c1):len(c1) + len(c2), 0], whole[len(c1):len(c1) + len(c2), 1], alpha=0.2, color=c2_color,
                s=scatter_size)
    plt.scatter(whole[len(c1) + len(c2): len(c1) + len(c2) + len(c3), 0],
                whole[len(c1) + len(c2):len(c1) + len(c2) + len(c3), 1], alpha=0.2, color=c3_color, s=scatter_size)
    plt.scatter(whole[len(c1) + len(c2) + len(c3):len(c1) + len(c2) + len(c3) + len(c4), 0],
                whole[len(c1) + len(c2) + len(c3):len(c1) + len(c2) + len(c3) + len(c4), 1], alpha=0.2, color=c4_color,
                s=scatter_size)

    # TODO: maybe i can remove that
    # VECTORS
    ax = plt.gca()
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)

        arrowprops = dict(arrowstyle='<-', linewidth=2, shrinkA=0, shrinkB=0)
        plt.annotate('', pca.mean_, pca.mean_ + v, arrowprops=arrowprops, color='black')

    plt.axis('off')
    plt.savefig(base_pca_vectors_fig)

    plt.clf()

    # TRANFORMED
    whole = np.array(transformed)
    plt.scatter(whole[0:len(c1), 0], whole[0:len(c1), 1], alpha=0.2, color=c1_color, s=scatter_size)
    plt.scatter(whole[len(c1):len(c1) + len(c2), 0], whole[len(c1):len(c1) + len(c2), 1], alpha=0.2, color=c2_color,
                s=scatter_size)
    plt.scatter(whole[len(c1) + len(c2): len(c1) + len(c2) + len(c3), 0],
                whole[len(c1) + len(c2):len(c1) + len(c2) + len(c3), 1], alpha=0.2, color=c3_color, s=scatter_size)
    plt.scatter(whole[len(c1) + len(c2) + len(c3):len(c1) + len(c2) + len(c3) + len(c4), 0],
                whole[len(c1) + len(c2) + len(c3):len(c1) + len(c2) + len(c3) + len(c4), 1], alpha=0.2, color=c4_color,
                s=scatter_size)
    plt.axis('off')
    plt.savefig(base_pca_transformed_fig)

    return copy_whole, transformed


def run_kernel(to_transform, fig_name, kernel, gamma=None, fCenter=False):
    c1, c2, c3, c4 = load_class_image(base_img)
    whole = to_transform
    print("#######")
    print("To transform: ")
    print(to_transform)

    pca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma)

    if fCenter:
        whole = StandardScaler().fit_transform(whole)
        print("After standarization: ")
        print(whole)

    transformed = pca.fit_transform(whole)
    print("Transformed: ")
    print(transformed)

    print("#######")

    whole = np.array(transformed)

    plt.clf()
    plt.scatter(whole[0:len(c1), 0], whole[0:len(c1), 1], alpha=0.2, color=c1_color, s=scatter_size)
    plt.scatter(whole[len(c1):len(c1) + len(c2), 0], whole[len(c1):len(c1) + len(c2), 1], alpha=0.2, color=c2_color,
                s=scatter_size)
    plt.scatter(whole[len(c1) + len(c2): len(c1) + len(c2) + len(c3), 0],
                whole[len(c1) + len(c2):len(c1) + len(c2) + len(c3), 1], alpha=0.2, color=c3_color, s=scatter_size)
    plt.scatter(whole[len(c1) + len(c2) + len(c3):len(c1) + len(c2) + len(c3) + len(c4), 0],
                whole[len(c1) + len(c2) + len(c3):len(c1) + len(c2) + len(c3) + len(c4), 1], alpha=0.2, color=c4_color,
                s=scatter_size)

    plt.axis('off')
    plt.savefig(fig_name, dpi=300)


# DONE
oryginal, transformed = run_basic()

run_kernel(oryginal, "cosineFromOriginal.png", "cosine")
run_kernel(transformed, "cosineFromTransformed.png", "cosine")

run_kernel(oryginal, "cosineFromOriginalCenter.png", "cosine", fCenter=True)
run_kernel(transformed, "cosineFromTransformedCenter.png", "cosine", fCenter=True)

for g in [0.000001, 0.0001, 0.01, 1, 10, 100, 10000]:
    run_kernel(oryginal, "rbfFromOriginal" + str(g) + ".png", "rbf", gamma=g)
    run_kernel(transformed, "rbfFromTransformed" + str(g) + ".png", "rbf", gamma=g)
