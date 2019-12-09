import random
import matplotlib.pyplot as plt
import cv2
from math import sqrt
import numpy as np
import heapq
from scipy.spatial import distance

width = 400
height = 400
distances = []
type = '.jpg'
separated_circles = 'separated_circles' + type
# separated_ellipses = 'separated_ellipses' + type
one_on_another = 'one_on_another' + type
one_in_another = 'one_in_another' + type
one_ring_circle = 'ring_circle' + type
with_irregular = 'irregular' + type
different_density = 'density_different' + type
images = [separated_circles, one_on_another, one_in_another, one_ring_circle, different_density]


def make_irregular_shape(base, test, test_percent):
    count = 100  # step counts
    step = int(width / count) + 1
    current = step
    for i in range(count):
        rand = random.randint(10, 200)
        make_circle(current, current, rand, 2, 'green', base, test, test_percent)
        current += step


def make_circle(cx, cy, radius, count, color, base, test, test_percent):
    for i in range(count):
        while True:
            x = random.randint(cx - radius, cx + radius)
            y = random.randint(cy - radius, cy + radius)
            if (cx - x) * (cx - x) + (cy - y) * (cy - y) < radius * radius:
                f_break = True
                for p in base:
                    if p[0] == x and p[1] == y:
                        f_break = False
                for p in test:
                    if p[0] == x and p[1] == y:
                        f_break = False
                if f_break:
                    break
        plt.scatter(x, y, color=color, s=1)
        if random.random() > test_percent:
            base.append((x, y, color))
        else:
            test.append((x, y, color))


def make_ring(cx, cy, radius, inner_radius, count, color, base, test, test_percent):
    for i in range(count):
        while True:
            x = random.randint(cx - radius, cx + radius)
            y = random.randint(cy - radius, cy + radius)
            if inner_radius * inner_radius < (cx - x) * (cx - x) + (cy - y) * (cy - y) < radius * radius:
                f_break = True
                for p in base:
                    if p[0] == x and p[1] == y:
                        f_break = False
                for p in test:
                    if p[0] == x and p[1] == y:
                        f_break = False
                if f_break:
                    break
        plt.scatter(x, y, color=color, s=1)
        if random.random() > test_percent:
            base.append((x, y, color))
        else:
            test.append((x, y, color))


def create_separated_circles(base, test, percent_test):
    radious1 = 80
    count1 = 200
    x1_center = random.randint(width / 2, width - radious1 - 10)
    y1_center = random.randint(0, height / 2 - radious1 - 10)
    make_circle(x1_center, y1_center, radious1, count1, 'red', base, test, percent_test)

    radious2 = 80
    count2 = 200
    x2_center = random.randint(0, width - radious2 - 10)
    y2_center = random.randint(height / 2 + radious2 + 10, height - radious2)
    make_circle(x2_center, y2_center, radious2, count2, 'blue', base, test, percent_test)

    radious3 = 80
    count3 = 200
    x3_center = random.randint(0, width / 2 - 2 * radious3)
    y3_center = random.randint(height / 2 + radious3 + 10, height - radious3)
    make_circle(x3_center, y3_center, radious3, count3, 'green', base, test, percent_test)

    # plt.axis('off')
    plt.savefig(separated_circles)
    return 'result_' + separated_circles


def create_one_on_another(base, test, percent_test):
    radious1 = 80
    count1 = 350
    x1_center = random.randint(width / 2 - radious1, width / 2 - 10)
    y1_center = random.randint(height / 2, height / 2 + radious1 - 10)
    make_circle(x1_center, y1_center, radious1, count1, 'red', base, test, percent_test)

    radious2 = 80
    count2 = 350
    x2_center = random.randint(width / 2 + 10, width / 2 + radious2)
    y2_center = random.randint(height / 2 - radious2 + 10, height)
    make_circle(x2_center, y2_center, radious2, count2, 'blue', base, test, percent_test)

    radious3 = 80
    count3 = 200
    x3_center = random.randint(0, width / 2 - radious1 - 10)
    y3_center = random.randint(height / 2 + radious2 - 10, height / 2 + radious2 + 10)
    make_circle(x3_center, y3_center, radious3, count3, 'green', base, test, percent_test)

    plt.axis('off')
    plt.savefig(one_on_another)
    return 'result_' + one_on_another


def create_different_density(base, test, percent_test):
    radious1 = 80
    x1_center = random.randint(width / 2 - radious1, width / 2 - 10)
    y1_center = random.randint(height / 2, height / 2 + radious1 - 10)
    make_circle(x1_center, y1_center, 20, 110, 'red', base, test, percent_test)
    make_ring(x1_center, y1_center, 40, 20, 120, 'red', base, test, percent_test)
    make_ring(x1_center, y1_center, 60, 40, 90, 'red', base, test, percent_test)
    make_ring(x1_center, y1_center, 80, 60, 80, 'red', base, test, percent_test)

    radious2 = 80
    count2 = 350
    x2_center = random.randint(width / 2 + 10, width / 2 + radious2)
    y2_center = random.randint(height / 2 - radious2 + 10, height)
    make_circle(x2_center, y2_center, radious2, count2, 'blue', base, test, percent_test)

    radious3 = 80
    count3 = 200
    x3_center = random.randint(0, width / 2 - radious1 - 10)
    y3_center = random.randint(height / 2 + radious2 - 10, height / 2 + radious2 + 10)
    make_circle(x3_center, y3_center, radious3, count3, 'green', base, test, percent_test)

    plt.axis('off')
    plt.savefig(different_density)
    return 'result_' + different_density


def create_one_in_another(base, test, percent_test):
    radious1 = 80
    radious2 = 20
    count1 = 350
    count2 = 100
    x1_center = random.randint(width / 2 - radious1, width / 2 - 10)
    y1_center = random.randint(height / 2, height / 2 + radious1 - 10)
    make_circle(x1_center, y1_center, radious1, count1, 'red', base, test, percent_test)
    make_circle(x1_center, y1_center, radious2, count2, 'green', base, test, percent_test)

    radious3 = 80
    count3 = 200
    x3_center = random.randint(0, width / 2 - radious1 - 10)
    y3_center = random.randint(height / 2 + radious2 - 10, height / 2 + radious2 + 10)
    make_circle(x3_center, y3_center, radious3, count3, 'blue', base, test, percent_test)

    plt.axis('off')
    plt.savefig(one_in_another)
    return 'result_' + one_in_another


def create_one_ring_circle(base, test, percent_test):
    radious1 = 80
    radious2 = 20
    count1 = 350
    count2 = 100
    x1_center = random.randint(width / 2 - radious1, width / 2 - 10)
    y1_center = random.randint(height / 2, height / 2 + radious1 - 10)
    make_ring(x1_center, y1_center, radious1, radious2, count1, 'red', base, test, percent_test)
    make_circle(x1_center, y1_center, radious2, count2, 'green', base, test, percent_test)

    radious3 = 80
    count3 = 200
    x3_center = random.randint(0, width / 2 - radious1 - 10)
    y3_center = random.randint(height / 2 + radious2 - 10, height / 2 + radious2 + 10)
    make_circle(x3_center, y3_center, radious3, count3, 'blue', base, test, percent_test)

    plt.axis('off')
    plt.savefig(one_ring_circle)
    return 'result_' + one_ring_circle


def create_irregular(base, test, percent_test):
    radious1 = 80
    count1 = 350
    x1_center = random.randint(width / 2 - radious1, width / 2 - 10)
    y1_center = random.randint(height / 2, height / 2 + radious1 - 10)
    make_circle(x1_center, y1_center, radious1, count1, 'red', base, test, percent_test)

    radious2 = 80
    count2 = 350
    x2_center = random.randint(width / 2 + 10, width / 2 + radious2)
    y2_center = random.randint(height / 2 - radious2 + 10, height)
    make_circle(x2_center, y2_center, radious2, count2, 'blue', base, test, percent_test)

    make_irregular_shape(base, test, percent_test)
    plt.axis('off')
    plt.savefig(with_irregular)
    return 'result_' + with_irregular


def make_blur():
    for i in images:
        im = cv2.imread(i)
        cv2.blur(im, (5, 5))
        cv2.imwrite(i, im)


############################# K-NN ###################################
def calculate_distance_euclidean(x1, y1, x2, y2):
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def find_nearest_N_euclidean(base, test_point, n):
    tmp_distances = []
    for tmp in base:
        distance = calculate_distance_euclidean(tmp[0], tmp[1], test_point[0], test_point[1])
        tmp_distances.append((distance, tmp))
    tmp_distances.sort(key=lambda x: x[0])
    # print(tmp_distances)
    return tmp_distances[:n]


# TODO: make it distance mahalanobis

def calculate_covariance_matrixes(base):
    red = []
    green = []
    blue = []
    for p in base:
        color = p[2]
        if color == 'red':
            red.append([p[0], p[1]])
        elif color == 'green':
            green.append([p[0], p[1]])
        else:
            blue.append([p[0], p[1]])

    red_arr = np.zeros((2, len(red)))
    for i, elem in enumerate(red):
        red_arr[0][i] = elem[0]
        red_arr[1][i] = elem[1]
    green_arr = np.zeros((2, len(green)))
    for i, elem in enumerate(green):
        green_arr[0][i] = elem[0]
        green_arr[1][i] = elem[1]
    blue_arr = np.zeros((2, len(blue)))
    for i, elem in enumerate(blue):
        blue_arr[0][i] = elem[0]
        blue_arr[1][i] = elem[1]

    cov_red = np.cov(red_arr)
    cov_green = np.cov(green_arr)
    cov_blue = np.cov(blue_arr)
    return [('red', np.linalg.inv(cov_red)), ('green', np.linalg.inv(cov_green)),
            ('blue', np.linalg.inv(cov_blue))]


def calculate_distance_mahalanobis(x1, y1, x2, y2, matrixes, color):
    for m in matrixes:
        if m[0] == color:
            current_matrix = m[1]
            d = distance.mahalanobis([x1, y1], [x2, y2], current_matrix)
            # print(d)
            return d
    return None


def find_nearest_N_mahalanobis(base, test_point, n, matrixes):
    tmp_distances = []
    for tmp in base:
        color = tmp[2]
        distance = calculate_distance_mahalanobis(tmp[0], tmp[1], test_point[0], test_point[1], matrixes, color)
        tmp_distances.append((distance, tmp))
    tmp_distances.sort(key=lambda x: x[0])
    return tmp_distances[:n]


def predict_class_most(nearest):
    to_ret = [['red', 0], ['green', 0], ['blue', 0]]

    for distance, point in nearest:
        x, y, color = point
        if color == 'red':
            to_ret[0][1] += 1
        elif color == 'green':
            to_ret[1][1] += 1
        elif color == 'blue':
            to_ret[2][1] += 1

        to_ret.sort(key=lambda m: m[1])

    return to_ret[-1][0]


# TODO: modify to wage
def predict_class_wage(nearest):
    to_ret = [['red', 0], ['green', 0], ['blue', 0]]
    nearest.sort(key=lambda m: m[0], reverse=True)
    # print(nearest)
    max_distance = nearest[0][0]
    # print(max_distance)
    for distance, point in nearest:
        x, y, color = point
        if distance == 0:
            return point[2]
        if color == 'red':
            to_ret[0][1] += max_distance / distance
        elif color == 'green':
            to_ret[1][1] += max_distance / distance
        elif color == 'blue':
            to_ret[2][1] += max_distance / distance

        to_ret.sort(key=lambda m: m[1])

    return to_ret[-1][0]


results = []
best_accuracy = []
lowest_accuracy = []
diff_accuracy = []


def proceed_all_results():
    # print needed in this alg

    plt.clf()

    ROC_x = []
    ROC_y = []
    ROC_txt = []
    PRC_x = []
    PRC_y = []
    PRC_txt = []

    two_largest = heapq.nlargest(2, range(len(diff_accuracy)), key=diff_accuracy.__getitem__)
    print("TWO_LARGEST ", two_largest)

    for i in two_largest:
        alg_worst = lowest_accuracy[i][0]
        alg_worst_whole = []
        alg_best = best_accuracy[i][0]
        alg_best_whole = []
        print("alg_worst: ", alg_worst)
        print("alg_best: ", alg_best)
        print(lowest_accuracy)
        print(best_accuracy)
        if i == 0:
            image = separated_circles
        elif i == 1:
            image = one_on_another
        elif i == 2:
            image = one_in_another
        elif i == 3:
            image = one_ring_circle
        elif i == 4:
            image = with_irregular
        elif i == 5:
            image = different_density

        for j, alg, accuracy, name, base, test, bad, well in results:

            if j == i and alg == alg_best:
                alg_best_whole.extend([j, alg, accuracy, name, base, test, bad, well])
            if j == i and alg == alg_worst:
                alg_worst_whole.extend([j, alg, accuracy, name, base, test, bad, well])
        # print(alg_worst_whole[4]) #base
        # print(alg_worst_whole[5]) #test
        # print()

        # TODO: policz confusion matrix, precision, recall, F1 score, G-score, narysuj wykresy ROC i PRC.
        best_confusion_matrix = calculate_confusion_matrix(alg_best_whole)
        worst_confusion_matrix = calculate_confusion_matrix(alg_worst_whole)
        best_precision_and_recall = calculate_precision_recall(best_confusion_matrix)
        worst_precision_and_recall = calculate_precision_recall(worst_confusion_matrix)

        for b in best_precision_and_recall:
            # ROC y - recall x - speciffity
            ROC_x.append(b[3])
            ROC_y.append(b[2])
            ROC_txt.append(image + b[0] + alg_best_whole[1])

            # PRC y - precission x - recall
            PRC_x.append(b[2])
            PRC_y.append(b[1])
            PRC_txt.append(image + b[0] + alg_best_whole[1])

        for w in worst_precision_and_recall:
            ROC_x.append(w[3])
            ROC_y.append(w[2])
            ROC_txt.append(image + w[0] + alg_worst_whole[1])

            PRC_x.append(w[2])
            PRC_y.append(w[1])
            PRC_txt.append(image + w[0] + alg_worst_whole[1])

        print("####################################################")
        print(i)
        print(image)
        print("WORST:", alg_worst_whole[1])
        print("\taccuracy", alg_worst_whole[2])
        print("\tname", alg_worst_whole[3])
        print("\tConfusion matrix")
        print(worst_confusion_matrix)
        print(
            "[['red', R_precision, R_recall,R_specifity , R_F1, R_G], ['green', G_precision, G_recall,G_specifity, G_F1, G_G], ['blue', B_precision, B_recall,B_specifity , B_F1, B_G]")
        print(worst_precision_and_recall)
        print("BEST:", alg_best_whole[1])
        print("\taccuracy", alg_best_whole[2])
        print("\tname", alg_best_whole[3])
        print("\tConfusion matrix")
        print(best_confusion_matrix)
        print(
            "[['red', R_precision, R_recall,R_specifity , R_F1, R_G], ['green', G_precision, G_recall,G_specifity, G_F1, G_G], ['blue', B_precision, B_recall,B_specifity , B_F1, B_G]")
        print(best_precision_and_recall)
        print("####################################################")

    # TODO: print roc and PRC
    print("##ROC##")
    print(ROC_x)
    print(ROC_y)
    print(ROC_txt)

    plt.clf()
    for i, txt in enumerate(ROC_txt):
        plt.scatter(ROC_x[i], ROC_y[i], label=txt)
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC_results.jpg")
    plt.savefig("ROC_results.jpg")

    plt.clf()
    for i, txt in enumerate(PRC_txt):
        plt.scatter(PRC_x[i], PRC_y[i], label=txt)
    plt.legend()
    plt.xlabel("recall")
    plt.ylabel("precission")
    plt.title("PRC_results.jpg")
    plt.savefig("PRC_results.jpg")

    # fig, axs = plt.subplots(2)
    #
    # for i, txt in enumerate(ROC_txt):
    #     axs[0].annotate(txt, (ROC_x[i], ROC_y[i]))
    # for i, txt in enumerate(PRC_txt):
    #     axs[0].annotate(txt, (PRC_x[i], PRC_y[i]))
    # plt.show()
    # plt.savefig("last_results.jpg")


# calculating confusion matrix
# predicted\actual   green  blue
# red
# green
# blue
def calculate_confusion_matrix(whole_data):
    confusion_matrix = np.zeros((3, 3))
    i, alg, accuracy, name, base, test, bad, well = whole_data
    for point, predicted in bad:
        x, y, color = point
        confusion_matrix[calculate_index(predicted)][calculate_index(color)] += 1
    for point, predicted in well:
        x, y, color = point
        confusion_matrix[calculate_index(predicted)][calculate_index(color)] += 1

    return confusion_matrix


def calculate_precision_recall(confusion_matrix):
    # TP index on diagonal
    # FP sum in row  but diagonal
    # FN sum in column but diagonal
    # TN rest not used
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # F1 = (2*precision*recall)/(precision + recall)
    # for RED
    R_TP = confusion_matrix[0][0]
    R_FP = confusion_matrix[0][1] + confusion_matrix[0][2]
    R_FN = confusion_matrix[1][0] + confusion_matrix[2][0]
    R_TN = confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][1] + confusion_matrix[2][2]
    R_precision = R_TP / (R_TP + R_FP)  #
    R_recall = R_TP / (R_TP + R_FN)  #
    R_specifity = R_TN / (R_TN + R_FP)
    R_F1 = (2 * R_precision * R_recall) / (R_precision + R_recall)
    R_G = sqrt(R_precision * R_recall)

    # for GREEN
    G_TP = confusion_matrix[1][1]
    G_FP = confusion_matrix[1][0] + confusion_matrix[1][2]
    G_FN = confusion_matrix[0][1] + confusion_matrix[2][1]
    G_TN = confusion_matrix[0][0] + confusion_matrix[2][2] + confusion_matrix[2][0] + confusion_matrix[0][2]
    G_precision = G_TP / (G_TP + G_FP)
    G_recall = G_TP / (G_TP + G_FN)
    G_specifity = G_TN / (G_TN + G_FP)
    G_F1 = (2 * G_precision * G_recall) / (G_precision + G_recall)
    G_G = sqrt(G_precision * G_recall)

    # for blue
    B_TP = confusion_matrix[2][2]
    B_FP = confusion_matrix[2][0] + confusion_matrix[2][1]
    B_FN = confusion_matrix[0][2] + confusion_matrix[1][2]
    B_TN = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[1][0] + confusion_matrix[0][1]
    B_precision = B_TP / (B_TP + B_FP)
    B_recall = B_TP / (B_TP + B_FN)
    B_specifity = B_TN / (B_TN + B_FP)
    B_F1 = (2 * B_precision * B_recall) / (B_precision + B_recall)
    B_G = sqrt(B_precision * B_recall)

    return [['red', R_precision, R_recall, R_specifity, R_F1, R_G],
            ['green', G_precision, G_recall, G_specifity, G_F1, G_G],
            ['blue', B_precision, B_recall, B_specifity, B_F1, B_G]]


def calculate_index(color):
    if color == 'red':
        return 0
    if color == 'green':
        return 1
    if color == 'blue':
        return 2

    pass


def calculate_result(name, base, test, well, bad, i, alg):
    accuracy = len(well) / len(test)
    results.append([i, alg, accuracy, name, base, test, bad, well])

    if accuracy > best_accuracy[i][1]:
        best_accuracy[i] = [alg, accuracy]
        diff_accuracy[i] = best_accuracy[i][1] - lowest_accuracy[i][1]
    if lowest_accuracy[i][1] > accuracy:
        lowest_accuracy[i] = [alg, accuracy]
        diff_accuracy[i] = best_accuracy[i][1] - lowest_accuracy[i][1]


def run():
    for i in range(6):
        best_accuracy.append([i, -1])
        lowest_accuracy.append([i, 101])
        diff_accuracy.append(0)

        base = []
        test = []

        test_percent = 0.2
        # figs are not cleaned
        if i == 0:
            name_to_save = create_separated_circles(base, test, test_percent)
        elif i == 1:
            name_to_save = create_one_on_another(base, test, test_percent)
        elif i == 2:
            name_to_save = create_one_in_another(base, test, test_percent)
        elif i == 3:
            name_to_save = create_one_ring_circle(base, test, test_percent)
        elif i == 4:
            name_to_save = create_irregular(base, test, test_percent)
        elif i == 5:
            name_to_save = create_different_density(base, test, test_percent)

        print("###############################")
        print(name_to_save)
        # TODO: euclidean most 1 , 7
        for N in [1, 7]:
            well_predicted = []
            bad_predicted = []
            for test_point in test:
                nearest = find_nearest_N_euclidean(base, test_point, N)
                predicted_class = predict_class_most(nearest)
                if predicted_class == test_point[2]:
                    well_predicted.append((test_point, predicted_class))
                else:
                    bad_predicted.append((test_point, predicted_class))
            # TODO: calculate everything he wanted
            calculate_result("outcome_most_euclidean_" + str(N) + "_" + name_to_save, base, test, well_predicted,
                             bad_predicted, i, "EM" + str(N))

            # TODO: draw prediction map
            plt.clf()
            for k in range(1, width, 4):
                print(k)
                for j in range(1, height, 4):
                    nearest = find_nearest_N_euclidean(base, (k, j), N)
                    predicted_class = predict_class_most(nearest)
                    plt.scatter(k, j, color=predicted_class, s=1)

            plt.savefig("most_euclidean_" + str(N) + "_" + name_to_save)
            # end of drawing map

        # TODO: 7 euclidean wage
        well_predicted = []
        bad_predicted = []
        for test_point in test:
            nearest = find_nearest_N_euclidean(base, test_point, 7)
            predicted_class = predict_class_wage(nearest)
            if predicted_class == test_point[2]:
                well_predicted.append((test_point, predicted_class))
            else:
                bad_predicted.append((test_point, predicted_class))
        calculate_result("outcome_wage_euclidean_7_" + name_to_save, base, test, well_predicted, bad_predicted, i,
                         "EW7")

        # TODO: draw prediction map
        plt.clf()
        for k in range(1, width, 4):
            for j in range(1, height, 4):
                nearest = find_nearest_N_euclidean(base, (k, j), 7)
                predicted_class = predict_class_wage(nearest)
                plt.scatter(k, j, color=predicted_class, s=1)

        plt.savefig("wage_euclidean_7_" + name_to_save)
        # end of drawing map

        # TODO: matrixes for mahalanobis
        matrixes = calculate_covariance_matrixes(base)

        # TODO: 1 mahalanobis most

        well_predicted = []
        bad_predicted = []
        for test_point in test:
            nearest = find_nearest_N_mahalanobis(base, test_point, 1, matrixes)
            predicted_class = predict_class_most(nearest)
            if predicted_class == test_point[2]:
                well_predicted.append((test_point, predicted_class))
            else:
                bad_predicted.append((test_point, predicted_class))
        # TODO: calculate everything he wanted
        calculate_result("outcome_most_mahalanobis_1_" + name_to_save, base, test, well_predicted, bad_predicted, i,
                         "MM1")

        # TODO: draw prediction map
        plt.clf
        for k in range(1, width, 4):
            for j in range(1, height, 4):
                nearest = find_nearest_N_mahalanobis(base, (k, j), 1, matrixes)
                predicted_class = predict_class_most(nearest)
                plt.scatter(k, j, color=predicted_class, s=1)
        plt.savefig("most_mahalanobis_1_" + name_to_save)
        # end of drawing map

        # TODO: 7 mahalanobis wage
        well_predicted = []
        bad_predicted = []
        for test_point in test:
            nearest = find_nearest_N_mahalanobis(base, test_point, 7, matrixes)
            predicted_class = predict_class_wage(nearest)
            if predicted_class == test_point[2]:
                well_predicted.append((test_point, predicted_class))
            else:
                bad_predicted.append((test_point, predicted_class))
        # TODO: calculate everything he wanted
        calculate_result("outcome_wage_mahalanobis_7_" + name_to_save, base, test, well_predicted, bad_predicted, i,
                         "MW7")

        # TODO: draw prediction map
        plt.clf()
        for k in range(1, width, 4):
            for j in range(1, height, 4):
                nearest = find_nearest_N_mahalanobis(base, (k, j), 7, matrixes)
                predicted_class = predict_class_wage(nearest)
                plt.scatter(k, j, color=predicted_class, s=1)

        plt.savefig("wage_mahalanobis_7_" + name_to_save)
        # end of drawing map

        plt.clf()

    proceed_all_results()


run()
