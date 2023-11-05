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

def printScatterComparisonMean(column):
    header = load.header
    base = load.baseResults
    metrics = load.metrics
    datasets = load.datasets

    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        without = []
        base_without = []
        base_preprocessed = []
        preprocessed = []
        for i, dataset in enumerate(datasets):
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            mean = df[column[0]].tolist()
            base_without.append(mean[0])
            without.append(np.average(mean[1:]))

            filename = base + dataset + '_' + metric + '_preprocessed.csv'
            df = pd.read_csv(filename)
            mean_preprocessed = df[column[0]].tolist()
            base_preprocessed.append(mean_preprocessed[0])
            preprocessed.append(np.average(mean_preprocessed[1:]))

        d = [(i-j)**2 for i, j in zip(without, base_without)]
        loss = [(i-j) for i, j in zip(without, base_without)]

        f, ax = plt.subplots()
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        if (np.mean(loss) > 0):
            ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nGain = {np.mean(loss):.3f}',
                    verticalalignment='top',
                    transform=ax.transAxes, bbox=props)
        else:
            ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nLoss = {np.abs(np.mean(loss)):.3f}',
                    verticalalignment='top',
                    transform=ax.transAxes, bbox=props)
        scatter = ax.scatter(without,
                             base_without,
                             marker='o', c=d, cmap='Spectral',  vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='ARI difference')

        ax.set_xlabel('framework mean ARI without preprocessing')
        ax.set_ylabel('base ARI without preprocessing')

        plt.savefig(f'images/{metric}_without_mean.pdf', bbox_inches='tight')

        d = [(i-j)**2 for i, j in zip(preprocessed, base_preprocessed)]
        loss = [(i-j) for i, j in zip(preprocessed, base_preprocessed)]

        f, ax = plt.subplots()
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

        if (np.mean(loss) > 0):
            ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nGain = {np.mean(loss):.3f}',
                    verticalalignment='top',
                    transform=ax.transAxes, bbox=props)
        else:
            ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nLoss = {np.abs(np.mean(loss)):.3f}',
                    verticalalignment='top',
                    transform=ax.transAxes, bbox=props)

        scatter = ax.scatter(preprocessed,
                             base_preprocessed,
                             marker='o', c=d, cmap='Spectral',  vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='ARI difference')

        ax.set_xlabel('framework mean ARI with preprocessing')
        ax.set_ylabel('base ARI with preprocessing')

        plt.savefig(
            f'images/{metric}_preprocessed_mean.pdf', bbox_inches='tight')

def printScatterComparisonBest(column):
    header = load.header
    base = load.baseResults
    metrics = load.metrics
    datasets = load.datasets

    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        without = []
        base_without = []
        base_preprocessed = []
        preprocessed = []
        for i, dataset in enumerate(datasets):
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            mean = df[column[0]].tolist()
            base_without.append(mean[0])
            without.append(np.max(mean[1:]))

            filename = base + dataset + '_' + metric + '_preprocessed.csv'
            df = pd.read_csv(filename)
            mean_preprocessed = df[column[0]].tolist()
            base_preprocessed.append(mean_preprocessed[0])
            preprocessed.append(np.max(mean_preprocessed[1:]))

        d = [(i-j)**2 for i, j in zip(without, base_without)]
        loss = [(i-j) for i, j in zip(without, base_without)]

        f, ax = plt.subplots()
        ax.set(xlim=(0, 1), ylim=(0, 1))

        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        if (np.mean(loss) > 0):
            ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nGain = {np.mean(loss):.3f}',
                    verticalalignment='top',
                    transform=ax.transAxes, bbox=props)
        else:
            ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nLoss = {np.abs(np.mean(loss)):.3f}',
                    verticalalignment='top',
                    transform=ax.transAxes, bbox=props)

        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        scatter = ax.scatter(without,
                             base_without,
                             marker='o', c=d, cmap='Spectral',  vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='ARI difference')

        ax.set_xlabel('framework best ARI without preprocessing')
        ax.set_ylabel('base ARI without preprocessing')

        plt.savefig(
            f'images/{metric}_without_best.pdf', bbox_inches='tight')

        d = [(i-j)**2 for i, j in zip(preprocessed, base_preprocessed)]
        loss = [(i-j) for i, j in zip(preprocessed, base_preprocessed)]

        f, ax = plt.subplots()
        ax.set(xlim=(0, 1), ylim=(0, 1))

        if (np.mean(loss) > 0):
            ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nGain = {np.mean(loss):.3f}',
                    verticalalignment='top',
                    transform=ax.transAxes, bbox=props)
        else:
            ax.text(.05, .95, f'Spread = {np.mean(d):.3f}\nLoss = {np.abs(np.mean(loss)):.3f}',
                    verticalalignment='top',
                    transform=ax.transAxes, bbox=props)

        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        scatter = ax.scatter(preprocessed,
                             base_preprocessed,
                             marker='o', c=d, cmap='Spectral',  vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='ARI difference')

        ax.set_xlabel('framework best ARI with preprocessing')
        ax.set_ylabel('base ARI with preprocessing')

        plt.savefig(
            f'images/{metric}_preprocessed_best.pdf', bbox_inches='tight')

def printRuntimes(columns):
    header = load.header
    base = load.baseResults
    metrics = load.metrics
    datasets = load.datasets
    
             
    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        run_time_without = []
        run_time_preprocessed = []
        total_without = []
        total_preprocessed = []
        for i, dataset in enumerate(datasets):
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            # base_without.append(mean[0])
            # without.append(np.average(mean[1:]))
            run_time_without.append(run_time_cluster[0]/np.average(run_time_cluster[1:]))
            total_without.append(run_time_cluster[0]/(np.average(run_time_cluster[1:]) + np.average(run_time_train[1:])))          
            
            filename = base + dataset + '_' + metric + '_preprocessed.csv'
            df = pd.read_csv(filename)
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            run_time_preprocessed.append(run_time_cluster[0]/np.average(run_time_cluster[1:]))
            total_preprocessed.append(run_time_cluster[0]/(np.average(run_time_cluster[1:]) + np.average(run_time_train[1:])))
            
        fig, ax = plt.subplots(2,1)
        ax[0].hist(run_time_preprocessed, bins=15, color='#86bf91', zorder=2, rwidth=0.9)
        ax[0].set_ylabel('Number of datasets')
        for bar in ax[0].containers[0]:
            x = bar.get_x() #+0.5 * bar.get_width()
            if x < 1:
                    bar.set_color('orange')
        ax[0].set_title(r"$\frac{T_k}{T_b}$",x=0.1,y=0.95, pad=-20,
            bbox=dict(facecolor='lightgrey', boxstyle='round,pad=0.5',alpha=0.5))
        
        # props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        # plt.text(0,0, f'T_k/T_b',
        #             # verticalalignment='top',
        #             # horizontalalignment = 'right'
        #             transform = ax.transAxes,
        #             bbox=props)
        ax[1].hist(total_preprocessed, bins=15, color='#86bf91', zorder=2, rwidth=0.9)
        ax[1].set_xlabel('Speed up times')
        ax[1].set_ylabel('Number of datasets')
        
        for bar in ax[1].containers[0]:
            x = bar.get_x() #+0.5 * bar.get_width()
            if x < 1:
                    bar.set_color('orange')

        ax[1].set_title(r"$\frac{T_k}{T_a+T_b}$",x=0.1,y=0.95, pad=-20,     
            bbox=dict(facecolor='lightgrey', boxstyle='round,pad=0.5',alpha=0.5))

        plt.tight_layout()        
        fig.savefig(
            f'images/{metric}_without_runtime.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(2,1)

        ax[0].hist(run_time_without, bins=15, color='#86bf91', zorder=2, rwidth=0.9)
        ax[0].set_ylabel('Number of datasets')
        ax[0].set_title(r"$\frac{T_k}{T_b}$",x=0.1,y=0.95, pad=-20,
            bbox=dict(facecolor='lightgrey', boxstyle='round,pad=0.5',alpha=0.5))
        for bar in ax[0].containers[0]:
            x = bar.get_x() +0.5 * bar.get_width()
            if x < 1:
                    bar.set_color('orange')


        ax[1].hist(total_without, bins=15, color='#86bf91', zorder=2, rwidth=0.9)
        ax[1].set_xlabel('Speed up times')
        ax[1].set_ylabel('Number of datasets')
        ax[1].set_title(r"$\frac{T_k}{T_a+T_b}$",x=0.1,y=0.95, pad=-20,
            bbox=dict(facecolor='lightgrey', boxstyle='round,pad=0.5',alpha=0.5))
# props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
       
        for bar in ax[1].containers[0]:
            x = bar.get_x() +0.5 * bar.get_width()
            if x < 1:
                    bar.set_color('orange')

        plt.tight_layout()        
        fig.savefig(
            f'images/{metric}_preprocessed_runtime.pdf', bbox_inches='tight')


        # f, ax = plt.subplots()
        # ax.hist(run_time_without, bins=25, color='#86bf91', zorder=2, rwidth=0.9)
        # ax.set_xlabel('Speed up times')
        # ax.set_ylabel('Number of datasets')
        # plt.savefig(
        #     f'images/{metric}_preprocessed_runtime.pdf', bbox_inches='tight')

def printScatterRuntimes(columns):
    header = load.header
    base = load.baseResults
    metrics = load.metrics
    datasets = load.datasets
    
             
    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        run_time_without = []
        run_time_preprocessed = []
        total_without = []
        total_preprocessed = []
        for i, dataset in enumerate(datasets):
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            run_time_without.append(run_time_cluster[0]/np.average(run_time_cluster[1:]))
            total_without.append(run_time_cluster[0]/(np.average(run_time_cluster[1:]) + np.average(run_time_train[1:])))          
            
            filename = base + dataset + '_' + metric + '_preprocessed.csv'
            df = pd.read_csv(filename)
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            run_time_preprocessed.append(run_time_cluster[0]/np.average(run_time_cluster[1:]))
            total_preprocessed.append(run_time_cluster[0]/(np.average(run_time_cluster[1:]) + np.average(run_time_train[1:])))


        f, ax = plt.subplots()
        ax.scatter(run_time_without,total_without)
        # fig, ax = plt.subplots(2,1)

        # ax[0].hist(run_time_preprocessed, bins=15, color='#86bf91', zorder=2, rwidth=0.9)
        # ax[0].set_xlabel('Speed up times')
        # ax[0].set_ylabel('Number of datasets')

        # for bar in ax[0].containers[0]:
        #     x = bar.get_x() #+0.5 * bar.get_width()
        #     if x < 1:
        #             bar.set_color('orange')

        # ax[1].hist(total_preprocessed, bins=15, color='#86bf91', zorder=2, rwidth=0.9)
        # ax[1].set_xlabel('Speed up times')
        # ax[1].set_ylabel('Number of datasets')
        
        # for bar in ax[1].containers[0]:
        #     x = bar.get_x() #+0.5 * bar.get_width()
        #     if x < 1:
        #             bar.set_color('orange')


        plt.savefig(
            f'images/{metric}_without_runtime.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(2,1)

        ax[0].hist(run_time_without, bins=15, color='#86bf91', zorder=2, rwidth=0.9)
        ax[0].set_xlabel('Speed up times')
        ax[0].set_ylabel('Number of datasets')
        for bar in ax[0].containers[0]:
            x = bar.get_x() +0.5 * bar.get_width()
            if x < 1:
                    bar.set_color('orange')

        ax[1].hist(total_without, bins=15, color='#86bf91', zorder=2, rwidth=0.9)
        ax[1].set_xlabel('Speed up times')
        ax[1].set_ylabel('Number of datasets')

        for bar in ax[1].containers[0]:
            x = bar.get_x() +0.5 * bar.get_width()
            if x < 1:
                    bar.set_color('orange')

        plt.tight_layout()        
        fig.savefig(
            f'images/{metric}_preprocessed_runtime.pdf', bbox_inches='tight')


        # f, ax = plt.subplots()
        # ax.hist(run_time_without, bins=25, color='#86bf91', zorder=2, rwidth=0.9)
        # ax.set_xlabel('Speed up times')
        # ax.set_ylabel('Number of datasets')
        # plt.savefig(
        #     f'images/{metric}_preprocessed_runtime.pdf', bbox_inches='tight')



def printRuntimes(columns):
    header = load.header
    base = load.baseResults
    metrics = load.metrics
    datasets = load.datasets

    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        run_time_without = []
        run_time_preprocessed = []
        total_without = []
        total_preprocessed = []
        for i, dataset in enumerate(datasets):
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            # base_without.append(mean[0])
            # without.append(np.average(mean[1:]))
            run_time_without.append(
                run_time_cluster[0]/np.average(run_time_cluster[1:]))
            total_without.append(
                run_time_cluster[0]/(np.average(run_time_cluster[1:]) + np.average(run_time_train[1:])))

            filename = base + dataset + '_' + metric + '_preprocessed.csv'
            df = pd.read_csv(filename)
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            run_time_preprocessed.append(
                run_time_cluster[0]/np.average(run_time_cluster[1:]))
            total_preprocessed.append(
                run_time_cluster[0]/(np.average(run_time_cluster[1:]) + np.average(run_time_train[1:])))

        fig, ax = plt.subplots(2, 1)
        ax[0].hist(run_time_preprocessed, bins=15,
                   color='#86bf91', zorder=2, rwidth=0.9)
        ax[0].set_ylabel('Number of datasets')
        for bar in ax[0].containers[0]:
            x = bar.get_x()  # +0.5 * bar.get_width()
            if x < 1:
                bar.set_color('orange')
        ax[0].set_title(r"$\frac{T_k}{T_b}$", x=0.1, y=0.95, pad=-20,
                        bbox=dict(facecolor='lightgrey', boxstyle='round,pad=0.5', alpha=0.5))

        # props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        # plt.text(0,0, f'T_k/T_b',
        #             # verticalalignment='top',
        #             # horizontalalignment = 'right'
        #             transform = ax.transAxes,
        #             bbox=props)
        ax[1].hist(total_preprocessed, bins=15,
                   color='#86bf91', zorder=2, rwidth=0.9)
        ax[1].set_xlabel('Speed up times')
        ax[1].set_ylabel('Number of datasets')

        for bar in ax[1].containers[0]:
            x = bar.get_x()  # +0.5 * bar.get_width()
            if x < 1:
                bar.set_color('orange')

        ax[1].set_title(r"$\frac{T_k}{T_a+T_b}$", x=0.1, y=0.95, pad=-20,
                        bbox=dict(facecolor='lightgrey', boxstyle='round,pad=0.5', alpha=0.5))

        plt.tight_layout()
        fig.savefig(
            f'images/{metric}_without_runtime.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(2, 1)

        ax[0].hist(run_time_without, bins=15,
                   color='#86bf91', zorder=2, rwidth=0.9)
        ax[0].set_ylabel('Number of datasets')
        ax[0].set_title(r"$\frac{T_k}{T_b}$", x=0.1, y=0.95, pad=-20,
                        bbox=dict(facecolor='lightgrey', boxstyle='round,pad=0.5', alpha=0.5))
        for bar in ax[0].containers[0]:
            x = bar.get_x() + 0.5 * bar.get_width()
            if x < 1:
                bar.set_color('orange')

        ax[1].hist(total_without, bins=15,
                   color='#86bf91', zorder=2, rwidth=0.9)
        ax[1].set_xlabel('Speed up times')
        ax[1].set_ylabel('Number of datasets')
        ax[1].set_title(r"$\frac{T_k}{T_a+T_b}$", x=0.1, y=0.95, pad=-20,
                        bbox=dict(facecolor='lightgrey', boxstyle='round,pad=0.5', alpha=0.5))
# props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)

        for bar in ax[1].containers[0]:
            x = bar.get_x() + 0.5 * bar.get_width()
            if x < 1:
                bar.set_color('orange')

        plt.tight_layout()
        fig.savefig(
            f'images/{metric}_preprocessed_runtime.pdf', bbox_inches='tight')

def printRuntimesBar(columns, full):
    header = load.header
    base = load.baseResults
    metrics = load.metrics
    datasets = load.datasets

    for j, metric in enumerate(metrics):
        y = ['original', '1log2or', '2log2or', '3log2or', '4log2or', '5log2or']
        run_time_without = []
        base_run_time_without = []
        base_run_time_preprocessed = []
        run_time_preprocessed = []
        train_time_without = []
        train_time_preprocessed = []
        for i, dataset in enumerate(datasets):
            filename = base + dataset + '_' + metric + '.csv'
            df = pd.read_csv(filename)
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            base_run_time_without.append(run_time_cluster[0])
            run_time_without.append(np.average(run_time_cluster[1:]))
            train_time_without.append(np.average(run_time_train[1:]))

            filename = base + dataset + '_' + metric + '_preprocessed.csv'
            df = pd.read_csv(filename)
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            run_time_cluster = df[columns[0]].tolist()
            run_time_train = df[columns[1]].tolist()
            base_run_time_preprocessed.append(run_time_cluster[0])
            run_time_preprocessed.append(np.average(run_time_cluster[1:]))
            train_time_preprocessed.append(np.average(run_time_train[1:]))

        width = 0.33

        fig, ax = plt.subplots()
        bottom = np.zeros(len(datasets))

        x = np.arange(len(datasets))
        offset = width * 0
        ax.bar(x + offset, base_run_time_without, width, label="base run time")

        offset = width*1
        p = ax.bar(x + offset, run_time_without, width,
                label="run time", bottom=bottom)
        if full:
            bottom = bottom + run_time_without
            p = ax.bar(x + offset, train_time_without, width,
                    label="train time", bottom=bottom)

        ax.set_xticks(x + width, datasets, rotation=90, fontsize=5)

        # ax.set_title("Number of penguins with above average body mass")
        ax.legend(loc="upper right")
        if full:
            fig.savefig(
                f'images/{metric}_without_runtime_bar_full.pdf', bbox_inches='tight')
        else:
            fig.savefig(
                f'images/{metric}_without_runtime_bar.pdf', bbox_inches='tight')

        fig, ax = plt.subplots()
        bottom = np.zeros(len(datasets))

        x = np.arange(len(datasets))
        offset = width * 0
        ax.bar(x + offset, base_run_time_preprocessed,
            width, label="base run time")

        offset = width*1
        p = ax.bar(x + offset, run_time_preprocessed,
                width, label="run time", bottom=bottom)
        if full:
            bottom = bottom + run_time_preprocessed
            p = ax.bar(x + offset, train_time_preprocessed,
                    width, label="train time", bottom=bottom)

        ax.set_xticks(x + width, datasets, rotation=90, fontsize=5)
        ax.legend(loc="upper right")

        if full:
            fig.savefig(
                f'images/{metric}_preprocessed_runtime_bar_full.pdf', bbox_inches='tight')
        else:
            fig.savefig(
                f'images/{metric}_preprocessed_runtime_bar.pdf', bbox_inches='tight')


images = [('adj_rand_mean', 'adj_rand_std')]

# for column in images:
#     # printStatistics(datasets, 'euclidean', True)
#     printMeanComparisonLoss(column)

columns = ('run_time', 'run_time_base')
printRuntimesBar(columns, False)
# printScatterRuntimes(columns)
