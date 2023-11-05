# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def getColor(c, N, idx):
    cmap = mpl.colormaps[c]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


def printWithErrorBars(base, datasets, metrics, header, column):
    plt_colors = [
        getColor(
            "Spectral",
            len(datasets),
            i) for i in range(
            len(datasets))]

    for metric in metrics:
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']

        for i, dataset in enumerate(datasets):
            print(dataset)
            print(metric)
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            mean = df[column[0]].tolist()
            std = df[column[1]].tolist()

            plt.errorbar(
                y,
                mean,
                std,
                linestyle='None',
                marker='^',
                label=dataset,
                color=plt_colors[i])
        plt.title("{:s} distance metric".format(metric))
        plt.ylabel(column[0])
        plt.xlabel('num of points')
        plt.legend(bbox_to_anchor=(1., 1), ncol=2)
        plt.show()


def printWinners(base, datasets, metrics, header, column):
    plt_colors = [
        getColor(
            "Spectral",
            len(datasets),
            i) for i in range(
            len(datasets))]

    for metric in metrics:
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        winners = dict()
        for yy in y:
            winners[yy] = 0
        for i, dataset in enumerate(datasets):
            print(dataset)
            print(metric)
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            mean = df[column[0]].tolist()[:1]
            mean.index(max(mean))

            std = df[column[1]].tolist()

            plt.errorbar(
                y,
                mean,
                std,
                linestyle='None',
                marker='^',
                label=dataset,
                color=plt_colors[i])
        plt.title("{:s} distance metric".format(metric))
        plt.ylabel(column[0])
        plt.xlabel('num of points')
        plt.legend(bbox_to_anchor=(1., 1), ncol=2)
        plt.show()


def printOneWithComparison(base, dataset, metrics, header, column):
    plt_colors = [getColor("Spectral", 4, i) for i in range(4)]
    plt.title("{:s}".format(dataset))
    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']

        filename = base + dataset + '_' + metric + '.csv'
        df = pd.read_csv(filename)
        mean = df[column[0]].tolist()
        std = df[column[1]].tolist()

        filename = base + dataset + '_' + metric + '_preprocessed.csv'
        df = pd.read_csv(filename)
        mean_preprocessed = df[column[0]].tolist()
        std_preprocessed = df[column[1]].tolist()

        # plt.axhline(y = mean[0], color = plt_colors[0], linestyle = '-')
        # plt.axhline(y = mean_preprocessed[0], color = plt_colors[1], linestyle = '-')

        plt.plot(y[1:], np.full(5, mean[0], dtype=np.float16),
                 color=plt_colors[2 * j])
        plt.plot(y[1:], np.full(5, mean_preprocessed[0],
                 dtype=np.float16), color=plt_colors[2 * j + 1])
        plt.fill_between(y[1:],
                         [a - b for a,
                          b in zip(np.full(5,
                                           mean[0],
                                           dtype=np.float16),
                                   np.full(5,
                                           std[0],
                                           dtype=np.float16))],
                         [a + b for a,
                          b in zip(np.full(5,
                                           mean[0],
                                           dtype=np.float16),
                                   np.full(5,
                                           std[0],
                                           dtype=np.float16))],
                         alpha=0.2, color=plt_colors[2 * j])
        plt.fill_between(y[1:],
                         [a - b for a,
                          b in zip(np.full(5,
                                           mean_preprocessed[0],
                                           dtype=np.float16),
                                   np.full(5,
                                           std_preprocessed[0],
                                           dtype=np.float16))],
                         [a + b for a,
                          b in zip(np.full(5,
                                           mean_preprocessed[0],
                                           dtype=np.float16),
                                   np.full(5,
                                           std_preprocessed[0],
                                           dtype=np.float16))],
                         alpha=0.2, color=plt_colors[2 * j + 1])

        plt.plot(y[1:],
                 mean[1:],
                 marker='o',
                 label='without processing ' + metric,
                 color=plt_colors[2 * j])
        plt.plot(y[1:],
                 mean_preprocessed[1:],
                 marker='o',
                 label='with processing ' + metric,
                 color=plt_colors[2 * j + 1])

        plt.fill_between(y[1:],
                         [a - b for a,
                          b in zip(mean[1:],
                                   std[1:])],
                         [a + b for a,
                          b in zip(mean[1:],
                                   std[1:])],
                         alpha=0.3,
                         color=plt_colors[2 * j])
        plt.fill_between(y[1:],
                         [a - b for a,
                          b in zip(mean_preprocessed[1:],
                                   std_preprocessed[1:])],
                         [a + b for a,
                          b in zip(mean_preprocessed[1:],
                                   std_preprocessed[1:])],
                         alpha=0.3,
                         color=plt_colors[2 * j + 1])

    plt.ylabel(column[0])
    plt.xlabel('num of points')
    plt.legend()
    # plt.legend(bbox_to_anchor =(1., 1), ncol=2)
    plt.show()


def printMeanComparisonLoss(base, datasets, metrics, header, column):

    plt_colors = [getColor("Spectral", 4, i) for i in range(4)]
    plt.title("Mean Loss of ARI")
    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        without = []
        preprocessed = []
        for i, dataset in enumerate(datasets):
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            mean = df[column[0]].tolist()
            std = df[column[1]].tolist()

            filename = base + dataset + '_' + metric + '_preprocessed.csv'
            df = pd.read_csv(filename)
            mean_preprocessed = df[column[0]].tolist()
            std_preprocessed = df[column[1]].tolist()

            without.append(mean[0] - np.average(mean[1:]))
            preprocessed.append(
                mean_preprocessed[0] - np.average(mean_preprocessed[1:]))

        print('For metric ' +
              metric +
              ' without preprocessing there is a loss of ' +
              str(np.average(without)) +
              '\u0080' +
              str(np.std(without)))
        print('For metric ' +
              metric +
              ' with preprocessing there is a loss of ' +
              str(np.average(preprocessed)) +
              '\u0080' +
              str(np.std(preprocessed)))

        plt.plot(datasets,
                 without,
                 marker='o',
                 label='without processing ' + metric,
                 color=plt_colors[2 * j + 0])
        plt.plot(datasets,
                 preprocessed,
                 marker='o',
                 label='with processing ' + metric,
                 color=plt_colors[2 * j + 1])

    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('dataset')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1., 1))
    plt.show()


def printMinComparisonLoss(base, datasets, metrics, header, column):

    plt_colors = [getColor("Spectral", 4, i) for i in range(4)]
    plt.title("Min Loss of ARI")
    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        without = []
        preprocessed = []
        for i, dataset in enumerate(datasets):
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            mean = df[column[0]].tolist()
            std = df[column[1]].tolist()

            filename = base + dataset + '_' + metric + '_preprocessed.csv'
            df = pd.read_csv(filename)
            mean_preprocessed = df[column[0]].tolist()
            std_preprocessed = df[column[1]].tolist()

            without.append(mean[0] - np.max(mean[1:]))
            preprocessed.append(
                mean_preprocessed[0] - np.max(mean_preprocessed[1:]))

        print('For metric ' +
              metric +
              ' without preprocessing there is a loss of ' +
              str(np.average(without)) +
              '\u0080' +
              str(np.std(without)))
        print('For metric ' +
              metric +
              ' with preprocessing there is a loss of ' +
              str(np.average(preprocessed)) +
              '\u0080' +
              str(np.std(preprocessed)))

        plt.plot(datasets,
                 without,
                 marker='o',
                 label='without processing ' + metric,
                 color=plt_colors[2 * j + 0])
        plt.plot(datasets,
                 preprocessed,
                 marker='o',
                 label='with processing ' + metric,
                 color=plt_colors[2 * j + 1])

    plt.ylabel('loss')
    plt.xlabel('dataset')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1., 1))
    plt.show()


datasets = ['ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF',
            'Car', 'Coffee', 'DiatomSizeReduction', 'DistalPhalanxTW', 'ECG200',
            'ECGFiveDays', 'FaceFour', 'Fish', 'LargeKitchenAppliances',
            'Lightning2', 'Lightning7', 'Meat', 'OSULeaf', 'OliveOil', 'Plane',
            'ProximalPhalanxTW', 'ShapeletSim', 'SonyAIBORobotSurface1',
            'SonyAIBORobotSurface2', 'Trace']

# datasets = ['Meat']
metrics = ['euclidean', 'dtw']  # , 'softdtw']
header = [
    'points',
    'adj_rand_mean',
    'adj_rand_std',
    'adj_mut_mean',
    'adj_mut_std',
    'run_time']
images = [('adj_rand_mean', 'adj_rand_std')]
# ('adj_mut_mean', 'adj_mut_std')]
base = 'results/'

for column in images:
    printWithErrorBars(base, datasets, metrics, header, column)
    printMeanComparisonLoss(base, datasets, metrics, header, column)
    printMinComparisonLoss(base, datasets, metrics, header, column)
    # printWinners(base, datasets, metrics, header, column)
    for dataset in datasets:
        printOneWithComparison(base, dataset, metrics, header, column)
# for column in images:
#     for metric in metrics:
#         y = ['original', '1log2or','2log2or','3log2or','4log2or','5log2or']

#         for dataset in datasets:
#             print(dataset)
#             print(metric)
#             filename ='results/' + dataset+'_'+metric+'.csv'
#             df = pd.read_csv(filename)
#             mean = df[column[0]].tolist()
#             std = df[column[1]].tolist()
#             filename ='results/' + dataset+'_'+metric+'_preprocessed.csv'
#             df = pd.read_csv(filename)
#             mean_preprocessed = df[column[0]].tolist()
#             std_preprocessed = df[column[1]].tolist()

#             plt.axhline(y = mean[0], color = 'r', linestyle = '-')
#             plt.axhline(y = mean_preprocessed[0], color = 'b', linestyle = '-')

#             plt.errorbar(y[1:], mean[1:], std[1:], linestyle='None', marker='^', label=dataset, color = 'r')
#             plt.errorbar(y[1:], mean_preprocessed[1:], std_preprocessed[1:], linestyle='None', marker='^', label=dataset, color = 'b')


#         plt.title("{:s} - {:s} - Adjusted Rand Index ".format(dataset, metric), fontsize=15)
#         plt.ylabel('ARI', fontsize=14)
#         plt.xlabel('Weights', fontsize=14)
#         plt.show()
