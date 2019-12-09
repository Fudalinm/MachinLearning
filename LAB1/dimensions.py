# https://pl.wikipedia.org/wiki/Hipersze%C5%9Bcian
# https://pl.wikipedia.org/wiki/Hiperkula


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, acos
import random


# losowy punkt w hiperszescianie

def A():
    # TODO: add this info to plot
    center = 0
    hypercube_edge_length = 20
    hypersphere_radius = hypercube_edge_length / 2
    points_number = 10000
    dimension_range = 15
    xes = []

    collectedResults = []
    for i in range(dimension_range):
        xes.append(i)
        collectedResults.append([])

    for h in range(10):
        result_list = []

        for dimensions in range(1, dimension_range + 1):

            # get points
            points = np.random.uniform(low=center - hypercube_edge_length / 2, high=center + hypercube_edge_length / 2,
                                       size=(points_number, dimensions))

            # print(points)

            inHypersphere = 0
            # check if in hypersphere
            for p in points:
                distance = 0
                for x in p:
                    distance += (x - center) * (x - center)
                # print(distance)
                if distance < hypersphere_radius * hypersphere_radius:
                    inHypersphere += 1

            inHyperspherePercent = inHypersphere * 100 / points_number
            result_list.append([dimensions, inHyperspherePercent])
            collectedResults[dimensions - 1].append(inHyperspherePercent)

            # print("Total points: {}".format(points_number))
            # print("Points in hypersphere: {}".format(inHypersphere))
            # print("Points in hypersphere/Total points * 100%: {}".format(inHyperspherePercent))

    # print(result_list)
    plt.title("DimensionCourse")
    plt.xlabel("Dimensions")
    plt.ylabel("PercentResult")
    deviations = []
    means = []

    for x in range(len(collectedResults)):
        dimensionResults = collectedResults[x]
        deviations.append(np.std(dimensionResults))
        means.append(np.mean(dimensionResults))

    p1 = plt.bar(xes, means, yerr=deviations)

    plt.show()
    # plt.savefig("A.jpg")


def B():
    center = 0
    hypercube_edge_length = 1
    points_number = 1000
    dimension_range = 14

    xes = []
    collectedResults = []
    for i in range(dimension_range + 1):
        xes.append(i)
        collectedResults.append([])

    for h in range(10):
        print(h)
        resultList = []

        for dimensions in range(1, dimension_range + 1):
            points = np.random.uniform(low=center - hypercube_edge_length / 2, high=center + hypercube_edge_length / 2,
                                       size=(points_number, dimensions))

            # wszystkie odleglosci miedzy punktami

            allDistances = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    distance = 0
                    p1 = points[i]
                    p2 = points[j]

                    for k in range(dimensions):
                        distance += (p1[k] - p2[k]) * (p1[k] - p2[k])
                    allDistances.append(sqrt(distance))

            standardDeviation = np.std(allDistances)
            mean = np.mean(allDistances)
            ratio = standardDeviation / mean
            resultList.append((dimensions, ratio))
            collectedResults[dimensions].append(ratio)

        plt.title("DimensionCourse")
        plt.xlabel("Dimensions")
        plt.ylabel("standardDev/mean")

    deviations = []
    means = []
    for x in range(len(collectedResults)):
        dimensionResults = collectedResults[x]
        deviations.append(np.std(dimensionResults))
        means.append(np.mean(dimensionResults))
    p1 = plt.bar(xes, means, yerr=deviations)
    plt.show()

    #
    # for d, r in resultList:
    #     plt.scatter(d, r, s=100)
    # # plt.show()
    # plt.savefig("B.jpg")


def C():
    center = 0
    hypercube_edge_length = 1
    points_number = 4000
    dimension_range = 14

    attempts = 15000

    resultList = []
    resultProceededList = []

    for x in range(dimension_range + 1):
        resultList.append([])

    for dimensions in range(1, dimension_range + 1):
        points = np.random.uniform(low=center - hypercube_edge_length / 2, high=center + hypercube_edge_length / 2,
                                   size=(points_number, dimensions))

        for a in range(attempts):
            p = random.sample(range(0, points_number), 4)
            pairA = (points[p[0]], points[p[1]])
            pairB = (points[p[2]], points[p[3]])
            vectorA = []
            vectorB = []

            # prepare vectors
            for d in range(dimensions):
                vectorA.append(pairA[1][d] - pairA[0][d])
                vectorB.append(pairB[1][d] - pairB[0][d])

            lengthA = 0
            lengthB = 0
            scalarAB = 0
            for d in range(dimensions):
                scalarAB += vectorA[d] * vectorB[d]
                lengthA += vectorA[d] * vectorA[d]
                lengthB += vectorB[d] * vectorB[d]

            lengthA = sqrt(lengthA)
            lengthB = sqrt(lengthB)

            angel = acos(scalarAB / (lengthA * lengthB))

            resultList[dimensions].append(angel)

        # po prostu policzmy srednia i rozklad
        standardDeviation = np.std(resultList[dimensions])
        mean = np.mean(resultList[dimensions])
        ratio = standardDeviation / mean

        resultProceededList.append((dimensions, standardDeviation, mean, ratio))
        #collectedResults[dimensions].append((standardDeviation, mean, ratio))

    fig, axes = plt.subplots(1, 3, subplot_kw=dict(polar=False))

    ratio = []
    dev = []
    mean = []
    x = []

    # TODO: create figure from this
    for i in range(dimension_range):
        ratio.append(resultProceededList[i][3])
        mean.append(resultProceededList[i][2])
        dev.append(resultProceededList[i][1])
        x.append(i)
    axes[0].set_title("Mean")
    axes[1].set_title("StandardDeviation")
    axes[2].set_title("standardDeviation/mean")

    axes[0].plot(x, mean)
    axes[1].plot(x, dev)
    axes[2].plot(x, ratio)

    axes[0].set_xlabel("dimensions")
    axes[1].set_xlabel("dimensions")
    axes[2].set_xlabel("dimensions")

    axes[0].set_ylabel("Radians")

    # plt.show()
    plt.savefig("C.jpg")


# A()
# B()
C()
