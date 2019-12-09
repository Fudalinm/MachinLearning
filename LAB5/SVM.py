import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.svm import SVC


def load_data():
    image = cv2.imread('data.png', cv2.IMREAD_GRAYSCALE)
    x, y = image.shape
    c1 = []  # 150
    c2 = []  # 138
    for i in range(x):
        for j in range(y):
            if image[i][j] == 150:
                c1.append([i, j])
            elif image[i][j] == 138:
                c2.append([i, j])
    return c1, c2, image.shape


def distance_point_plane(point_x, point_y, clf):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 50)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    # print(w, a, xx, yy)
    # plt.plot(xx, yy)
    # print(w.shape)
    # print(w.shape)
    w_norm = np.linalg.norm(clf.coef_)
    d = clf.decision_function([[point_x, point_y]]) / w_norm
    # print(d)
    return d


def map_distance_to_val(x, y, d):
    max_distance = x ** 2 + y ** 2
    return int((255 * d) / sqrt(max_distance))


def calculate_accuracy(c1, c2, clf):
    pred1 = clf.predict(c1)
    pred2 = clf.predict(c2)
    bad = 0
    for i in pred1:
        if i != 1:
            bad += 1
    for i in pred2:
        if i != 2:
            bad += 1

    accuracy = (len(c1) + len(c2) - bad) / (len(c1) + len(c2))

    print("BAD: ", bad)
    print("ACCURACY: ", accuracy)


if __name__ == "__main__":
    c1, c2, (x, y) = load_data()
    c1_arr = np.array(c1)
    c2_arr = np.array(c2)
    all_classes = []
    all_classes.extend(c1)
    all_classes.extend(c2)
    class_classyfication = []
    class_classyfication.extend([1 for element in c1])
    class_classyfication.extend([2 for element in c2])
    # print(all_classes)
    # print(class_classyfication)
    #
    # print(x, y)
    fid_dir = 'figs/'
    pred_dir = 'pred/'
    # print(c1, c2)
    for kernel in ['linear', 'rbf',
                   'poly']:  # Dokonajmy tych samych obliczeń dla zwykłego SVM, SVM z kernelem wielomianowym
        for c in [0.1, 0.5, 0.7, 1, 1.3, 2, 2.5, 3, 5, 7]:
            if kernel == 'linear':
                gamma_list = [1]
            else:
                gamma_list = [0.1, 1, 10, 100]
            for gamma in gamma_list:
                current = fid_dir + kernel + '_c' + str(c) + '_g' + str(gamma) + '.png'
                current_prediction = pred_dir + kernel + '_c' + str(c) + '_g' + str(gamma) + '.png'
                print(current)
                clf = SVC(kernel=kernel, C=c)
                clf.fit(all_classes, class_classyfication)
                map = np.zeros((x, y, 3))
                to_predict = [[i, j] for i in range(x) for j in range(y)]
                predicted = clf.predict(to_predict)
                for (point, class_prediction) in zip(to_predict, predicted):
                    i = point[0]
                    j = point[1]
                    if class_prediction == 1:
                        map[i, j, 0] = 128
                        map[i, j, 1] = 128
                    else:
                        map[i, j, 0] = 128
                        map[i, j, 2] = 128
                for class_point1 in c1:
                    map[class_point1[0], class_point1[1], 1] = 150
                for class_point1 in c2:
                    map[class_point1[0], class_point1[1], 2] = 138

                cv2.imwrite(current_prediction, map)
                calculate_accuracy(c1, c2, clf)

                map = np.zeros((x, y))
                max_distance = x ** 2 + y ** 2
                for (point, class_prediction) in zip(to_predict, predicted):
                    i = point[0]
                    j = point[1]
                    # find closest from different class
                    closest = max_distance
                    for (point2, class_prediction2) in zip(to_predict, predicted):
                        if class_prediction2 == class_prediction:
                            continue
                        i2 = point2[0]
                        j2 = point2[1]
                        distance = (i - i2) ** 2 + (j - j2) ** 2
                        if distance < closest:
                            closest = distance
                    map[i, j] = 255 - int(sqrt(closest) * 255 / sqrt(max_distance))
                plt.imshow(map, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.savefig(current)
                plt.clf()
                # cv2.imwrite(current, map)
