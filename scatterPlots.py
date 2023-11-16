import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import load


def getColor(c, N, idx):
    cmap = mpl.colormaps[c]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


def printMeanComparisonLoss(column):
    header = load.header
    base = load.baseResults
    metrics = load.metrics
    datasets = load.datasets
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


def printScatterComparison(column, choose='mean', a_i=None, normalized=False, metric='dtw'):
    header = load.header
    baseResults = load.baseResults
    baseAnalytics = load.baseAnalytics
    datasets = load.datasets
    if normalized:
        preprocessing = '_preprocessed'
        normalization = 'with normalization'
    else:
        preprocessing = ''
        normalization = 'without normalization'
    if a_i == None:
        aa = ''
    else:
        aa = f"for $\\alpha$={load.a[a_i-1]} "

    # y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
    framework = []
    base = []
    if column == load.ARI:
        type = load.ADJ_RAND
        acc = load.ARI_LABEL
    elif column == load.AMI:
        type = load.ADJ_MUT
        acc = load.AMI_LABEL

    for dataset in datasets:
        filename = f"{baseResults}{dataset}_{metric}{preprocessing}.csv"
        df = pd.read_csv(filename)
        mean = df[header[column[0]]].tolist()
        base.append(mean[0])
        if (a_i == None):
            if choose == 'mean':
                framework.append(np.average(mean[1:]))
            elif choose == 'best':
                filename = f"{baseAnalytics}{dataset}_{metric}_{type}{preprocessing}.csv"
                df = pd.read_csv(filename, header=None)
                mean = []
                for x_i in load.x:
                    mean.append(df[x_i].tolist())
                framework.append(np.max(mean))
        else:

            filename = f"{baseAnalytics}{dataset}_{metric}_{type}{preprocessing}.csv"
            df = pd.read_csv(filename, header=None)
            mean = df[a_i].tolist()
            if choose == 'mean':
                framework.append(np.average(mean))
            elif choose == 'best':
                framework.append(np.max(mean))

    d = [(i-j)**2 for i, j in zip(base, framework)]
    loss = [(i-j) for i, j in zip(base, framework)]

    f, ax = plt.subplots()
    ax.set(xlim=(-0.1, 1), ylim=(-0.1, 1))
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    if (np.mean(loss) > 0):
        ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nLoss = {np.mean(loss):.3f}',
                verticalalignment='top',
                transform=ax.transAxes, bbox=props)
    else:
        ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nGain = {np.abs(np.mean(loss)):.3f}',
                verticalalignment='top',
                transform=ax.transAxes, bbox=props)
    scatter = ax.scatter(framework,
                         base,
                         marker='o', c=d, cmap='Spectral',  vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label=f'{acc} difference')
    ax.set_xlabel(rf'framework {choose} {acc} {aa}{normalization}')
    ax.set_ylabel(f'base {acc} {normalization}')
    plt.savefig(
        f'images/{metric}_{acc}_{choose}_{a_i}{preprocessing}.pdf', bbox_inches='tight')


images = [('adj_rand_mean', 'adj_rand_std')]

# for column in images:
#     # printStatistics(datasets, 'euclidean', True)
#     printMeanComparisonLoss(column)

columns = ('run_time', 'run_time_base')

for metric in load.metrics:
    for choose in ['mean', 'best']:
        for preprocess in [True, False]:
            for col in [None, 1]:
                for acc in [load.AMI, load.ARI]:
                    print(acc, choose, col, preprocess, metric)
                    printScatterComparison(
                        acc, choose, col, preprocess, metric)
