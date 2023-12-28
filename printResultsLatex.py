import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import io
import load
import os
import sys


x = load.x
a = load.a
ARI = load.ARI
AMI = load.AMI
RUNTIME = load.RUNTIME
ADJ_MUT = load.ADJ_MUT
ADJ_RAND = load.ADJ_RAND


def get_filename(dataset, metric, preprocessed):
    if preprocessed:
        ret = f'{dataset}_{metric}_preprocessed.csv'
    else:
        ret = f'{dataset}_{metric}.csv'
    return ret


def get_specific_filename(dataset, metric, preprocessed, adjs):
    if preprocessed:
        ret = f'{dataset}_{metric}_{adjs}_preprocessed.csv'
    else:
        ret = f'{dataset}_{metric}_{adjs}.csv'
    return ret


def write_specific_table_header(stream, metric, preprocessed, adjs, col):
    if adjs == ADJ_MUT:
        write_table_header(stream, metric, preprocessed, AMI, col)
    elif adjs == ADJ_RAND:
        write_table_header(stream, metric, preprocessed, ARI, col)


def write_table_header(stream, metric, preprocessed, cols, col=None):
    ass_metric = ''
    if cols == ARI:
        ass_metric = 'ARI'
    elif cols == AMI:
        ass_metric = 'AMI'
    elif cols == RUNTIME:
        ass_metric = 'RUNTIME'
    stream.write(f'\\begin{{table}}[ht]')
    stream.write('\n')
    stream.write(
        f'\\caption{{{ass_metric} {metric} preprocessed {preprocessed} col {col}}} ')
    stream.write
    stream.write('\n')
    if ass_metric == 'RUNTIME':
        stream.write(f'\\begin{{adjustbox}}{{width=0.7\\textwidth,center}}')
        stream.write('\n')
        stream.write(f'\\begin{{tabular}}{{lccccc}}')
    else:
        stream.write(f'\\begin{{adjustbox}}{{width=\\textwidth}}')
        stream.write('\n')
        stream.write(f'\\begin{{tabular}}{{lcccccc}}')
    stream.write('\n')
    stream.write(f'\\hline')
    stream.write('\n')
    if ass_metric == 'RUNTIME':
        string = 'dataset & $T_k$ & $T_b$ & $T_a$ & $\\frac{T_k}{T_b}$ & $\\frac{T_a + T_b}{T_k}$'
    else:
        string = 'dataset & baseline'
        for i in x:
            string = f"{string} & \\textbf{{${i}\\cdot \log{{T}}$}}"
    string = f'{string} \\\\ \\hline'
    stream.write(string)
    stream.write('\n')


def write_table_close(stream):
    stream.write('\\hline')
    stream.write('\n')
    stream.write(f'\\end{{tabular}}')
    stream.write('\n')
    stream.write(f'\\end{{adjustbox}}')
    stream.write('\n')
    stream.write(f'\\end{{table}}')
    stream.write('\n')


def create_df(col, adjs, df1, df2):
    df = pd.DataFrame(columns=['metric'], index=range(6))
    if adjs == ADJ_MUT:
        run = AMI[0]
    elif adjs == ADJ_RAND:
        run = ARI[0]
    df.iat[0, 0] = df1.iat[0, run]
    for i in range(5):
        df.iat[i+1, 0] = df2.iat[i, col]
    return df

def create_df_bm(cols, adjs, df1, df2, option='mean'):
    df = pd.DataFrame(columns=['metric'], index=range(6))
    df_std = pd.DataFrame(columns=['metric'], index=range(6))
    if adjs == ADJ_MUT:
        run = AMI[0]
        run_std  = AMI[1]
    elif adjs == ADJ_RAND:
        run = ARI[0]
        run_std = ARI[1]
    df.iat[0, 0] = df1.iat[0, run]
    df_std.iat[0, 0] = df1.iat[0, run_std]
    for i in range(5):
        values = []
        for col in cols:
            values.append(df2.iat[i, col])
        if option ==  'mean':
            df.iat[i+1, 0] = np.mean(values)
            df_std.iat[i+1, 0] = np.std(values)
        elif option == 'best':
            df.iat[i+1, 0] = np.max(values)
    if option == 'mean':
        # print(df_std)
        return df, df_std
    else:
        return df


def write_specific_table_contents(stream, dataset, df, accumulate):
    lookup = dict()
    cols = 0
    sorted_indices = df.iloc[:, cols].argsort()
    for argsort_index, original_index in sorted_indices.items():
        lookup[original_index] = argsort_index

    string = f'{dataset}'

    for index, row in df.iterrows():
        ord = 5 - lookup[index] + 1
        if ord == 1:
            string = f'{string} & \\textbf{{{row.iloc[cols]:.3f}}} ({ord})'
        else:
            string = f'{string} & {row.iloc[cols]:.3f} ({ord})'
        accumulate[0][index] = accumulate[0][index] + row.iloc[cols]
        accumulate[1][index] = accumulate[1][index] + ord
    string = f'{string} \\\\'
    stream.write(string)
    stream.write('\n')
    return accumulate

def write_specific_table_contents_bm(stream, dataset, df, df_std, accumulate):
    lookup = dict()
    cols = 0
    sorted_indices = df.iloc[:, cols].argsort()
    for argsort_index, original_index in sorted_indices.items():
        lookup[original_index] = argsort_index

    string = f'\\textbf{{{dataset}}}'

    for (index, row), (_, row_std) in zip(df.iterrows(), df_std.iterrows()):
        ord = 5 - lookup[index] + 1
        if ord == 1:
            string = f'{string} & \\textbf{{{row.iloc[cols]:.3f} \u00B1 {row_std.iloc[cols]:.3f}}} ({ord})'
        else:
            string = f'{string} & {row.iloc[cols]:.3f} \u00B1 {row_std.iloc[cols]:.3f}({ord})'
        accumulate[0][index] = accumulate[0][index] + row.iloc[cols]
        accumulate[1][index] = accumulate[1][index] + ord
    string = f'{string} \\\\'
    stream.write(string)
    stream.write('\n')
    return accumulate


def write_table_contents(stream, dataset, cols, df, accumulate):
    lookup = dict()
    sorted_indices = df.iloc[:, cols[0]].argsort()
    for argsort_index, original_index in sorted_indices.items():
        lookup[original_index] = argsort_index

    string = f'{dataset}'

    if cols == RUNTIME:
        Ta = 0
        Tb = 0
        Tk = 0
        for index, row in df.iterrows():
            if index == 0:
                Tk = row.iloc[cols[0]]
                string = f'{string} & {Tk:.3f}'
            else:
                Tb += row.iloc[cols[0]]
                Ta += row.iloc[cols[1]]
        Ta = Ta/5
        Tb = Tb/5
        string = f'{string} & {Tb:.3f} & {Ta:.3f} & {Tk/Tb:.3f} & {(Ta+Tb)/Tk:.3f}'
        accumulate[0][0] = accumulate[0][0] + Tk
        accumulate[0][1] = accumulate[0][1] + Tb
        accumulate[0][2] = accumulate[0][2] + Ta
        accumulate[0][3] = accumulate[0][3] + (Tk/Tb)
        accumulate[0][4] = accumulate[0][4] + ((Ta+Tb)/Tk)
    else:
        for index, row in df.iterrows():
            ord = 5 - lookup[index] + 1
            if ord == 1:
                string = f'{string} & \\textbf{{{row.iloc[cols[0]]} \u00B1 {row.iloc[cols[1]]}}} ({ord})'
            else:
                string = f'{string} & {row.iloc[cols[0]]} \u00B1 {row.iloc[cols[1]]} ({ord})'
            accumulate[0][index] = accumulate[0][index] + row.iloc[cols[0]]
            accumulate[1][index] = accumulate[1][index] + ord
    string = f'{string} \\\\'
    stream.write(string)
    stream.write('\n')
    return accumulate


def print_table(datasets, metric, preprocessed, base, cols, stream=sys.stdout):
    header_buffered_string = io.StringIO()
    write_table_header(header_buffered_string, metric, preprocessed, cols)
    stream.write(header_buffered_string.getvalue())
    accumulate = np.zeros((2, 6))
    for d in datasets:
        filename = os.path.join(base, get_filename(d, metric, preprocessed))
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            accumulate = write_table_contents(stream, d, cols, df, accumulate)
    avg = accumulate / len(datasets)
    stream.write('\hline \n')
    string = f'mean'
    if cols == RUNTIME:
        for i in range(3):
            string = f'{string} & {avg[0][i]:.3f}'
    else:
        for i in range(avg.shape[1]):
            string = f'{string} & {avg[0][i]:.3f} ({avg[1][i]:.3f})'
    string = f'{string} \\\\ '
    last_line = string
    stream.write(string)
    stream.write('\n')
    footer_buffered_string = io.StringIO()
    write_table_close(footer_buffered_string)
    stream.write(footer_buffered_string.getvalue())
    return header_buffered_string.getvalue(), footer_buffered_string.getvalue(), last_line


def print_specific_table(datasets, metric, preprocessed, base1, base2, adjs, col, stream):
    write_specific_table_header(stream, metric, preprocessed, adjs, col)
    accumulate = np.zeros((2, 6))
    for d in datasets:
        filename1 = os.path.join(base1, get_filename(d, metric, preprocessed))
        filename2 = os.path.join(
            base2, get_specific_filename(d, metric, preprocessed, adjs))

        if os.path.exists(filename1) and os.path.exists(filename2):
            df1 = pd.read_csv(filename1)
            df2 = pd.read_csv(filename2, header=None)
            df = create_df(col, adjs, df1, df2)
            accumulate = write_specific_table_contents(
                stream, d, df, accumulate)
    avg = accumulate / len(datasets)
    stream.write('\hline \n')
    string = f'mean'
    for i in range(avg.shape[1]):
        string = f'{string} & {avg[0][i]:.3f} ({avg[1][i]:.3f})'
    string = f'{string} \\\\ '
    last_line = string
    stream.write(string)
    stream.write('\n')
    write_table_close(stream)
    return last_line

def print_specific_table_bm(datasets, metric, preprocessed, base1, base2, adjs, cols, bm, stream ):
    write_specific_table_header(stream, metric, preprocessed, adjs, cols)
    accumulate = np.zeros((2, 6))
    for d in datasets:
        filename1 = os.path.join(base1, get_filename(d, metric, preprocessed))
        filename2 = os.path.join(
            base2, get_specific_filename(d, metric, preprocessed, adjs))

        if os.path.exists(filename1) and os.path.exists(filename2):
            df1 = pd.read_csv(filename1)
            df2 = pd.read_csv(filename2, header=None)
            if bm == 'mean':
                df, df_std = create_df_bm(cols, adjs, df1, df2, bm)
                accumulate = write_specific_table_contents_bm(
                    stream, d, df, df_std, accumulate)
            else:
                df = create_df_bm(cols, adjs, df1, df2, bm)
                accumulate = write_specific_table_contents(
                    stream, d, df, accumulate)

    avg = accumulate / len(datasets)
    stream.write('\hline \n')
    string = f'mean'
    for i in range(avg.shape[1]):
        string = f'{string} & {avg[0][i]:.3f} ({avg[1][i]:.3f})'
    string = f'{string} \\\\ '
    last_line = string
    stream.write(string)
    stream.write('\n')
    write_table_close(stream)
    return last_line



def print_run_time_table(datasets, metric, preprocessed, base, stream=sys.stdout):
    return print_table(datasets, metric, preprocessed, base, RUNTIME, stream)


def print_ami_table(datasets, metric, preprocessed, base, stream=sys.stdout):
    return print_table(datasets, metric, preprocessed, base, AMI, stream)


def print_ari_table(datasets, metric, preprocessed, base, stream=sys.stdout):
    return print_table(datasets, metric, preprocessed, base, ARI, stream)


def print_ari_specific_table_bm(datasets, metric, preprocessed, base1, base2, cols, bm, stream=sys.stdout):
    return print_specific_table_bm(datasets, metric, preprocessed,
                         base1, base2, ADJ_RAND, cols, bm, stream)


def print_ari_specific_table(datasets, metric, preprocessed, base1, base2, col, stream=sys.stdout):
    return print_specific_table(datasets, metric, preprocessed,
                         base1, base2, ADJ_RAND, col, stream)


def print_ami_specific_table(datasets, metric, preprocessed, base1, base2, col, stream=sys.stdout):
    return print_specific_table(datasets, metric, preprocessed,
                         base1, base2, ADJ_MUT, col, stream)


def printStatistics(datasets, metric, preprocessed):

    stats = dict()
    for d in datasets:
        dataset_dict = dict()
        filename = d + '_' + metric + '_adj_rand_ind'
        if preprocessed:
            filename = filename + '_preprocessed'
        filename = filename + '.csv'
        print(filename)
        df = pd.read_csv(filename, header=None)

        to_sort = []
        for i in range(0, df.shape[0]):
            for j in range(1, df.shape[1]):
                to_sort.append((i, j, df.iat[i, j]))  # i = a , j = w[j]
        sorted_list = sorted(to_sort, key=lambda item: item[2], reverse=True)
        # print(sorted_list)
        for i, item in enumerate(sorted_list):
            dataset_dict[(item[0], item[1])] = (item[2], i + 1)
        # print(dataset_dict)
        stats[d] = dataset_dict

    string = 'dataset'
    cum = dict()

    for i in range(0, df.shape[0]):
        for j in range(1, df.shape[1]):
            cum[(i, j)] = 0
            string = string + '&' + '(' + str(i) + ',' + str(j) + ')'
    string = string + '\\\\ \\hline'
    print(string)
    for d in datasets:
        string = d
        dataset_dict = stats[d]
        for i in range(0, df.shape[0]):
            for j in range(1, df.shape[1]):
                s1 = dataset_dict[(i, j)][0]
                s2 = dataset_dict[(i, j)][1]
                string = string + '&' + \
                    str(format(s1, '.3f')) + ' (' + str(s2) + ')'
                cum[(i, j)] = cum[(i, j)] + s2 / len(datasets)
                # string = string+'('+str(dataset_dict[(item[i], item[j])][1])+')'
        string = string + '\\\\'
        print(string)

    string = ''
    for i in range(0, df.shape[0]):
        for j in range(1, df.shape[1]):
            string = string + '&' + str(format(cum[(i, j)], '.3f'))
    print(string)


def printFullResultsPaper(datasets, metric, preprocessed, base):
    print("\\begin{table}[ht]")
    print(f"\\caption{{{metric} preprocessed {preprocessed}}}")
    print("\\begin{adjustbox}{width=\\textwidth}%,angle=90}")
    print(" \\begin{tabular}{lcccccccccccccccccccccccccccccc}")
    print("\\hline")
    stats = dict()
    for d in datasets:
        dataset_dict = dict()
        filename = f"{base}{d}_{metric}_adj_rand_ind"
        if preprocessed:
            filename = f"{filename}_preprocessed"
        filename = f"{filename}.csv"
        # print(filename)
        df = pd.read_csv(filename, header=None)

        to_sort = []
        for i in range(0, df.shape[0]):
            for j in range(1, df.shape[1]):
                to_sort.append((i, j, df.iat[i, j]))  # i = a , j = w[j]
        sorted_list = sorted(to_sort, key=lambda item: item[2], reverse=True)
        # print(sorted_list)
        for i, item in enumerate(sorted_list):
            dataset_dict[(item[0], item[1])] = (item[2], i + 1)
        # print(dataset_dict)
        stats[d] = dataset_dict
    string = " "
    for i in range(0, df.shape[0]):
        string = f"{string} & \multicolumn{{6}}{{c}}{{\\textbf{{{x[i]}$\\cdot logT$}}}}"
    string = f"{string} \\\\"
    print(string)
    string = '\\textbf{{Dataset name}}'
    cum = dict()
    avg = dict()
    for i in range(0, df.shape[0]):
        for j in range(1, df.shape[1]):
            cum[(i, j)] = 0
            avg[(i, j)] = 0
            string = f"{string} & \\textbf{{{a[j-1]}}}"
    string = f"{string} \\\\ \hline"
    print(string)
    for d in datasets:
        string = d
        dataset_dict = stats[d]
        for i in range(0, df.shape[0]):
            for j in range(1, df.shape[1]):
                s1 = dataset_dict[(i, j)][0]
                s2 = dataset_dict[(i, j)][1]
                string = f"{string} & {s1:.3f} ({s2})"
                cum[(i, j)] = cum[(i, j)] + s2 / len(datasets)
                avg[(i, j)] = avg[(i, j)] + s1 / len(datasets)
                # string = string+'('+str(dataset_dict[(item[i], item[j])][1])+')'
        string = f"{string} \\\\"
        print(string)

    print("\\hline")

    string = ''
    for i in range(0, df.shape[0]):
        for j in range(1, df.shape[1]):
            string = f"{string} & {avg[(i, j)]:.3f} ({cum[(i, j)]:.1f})"
    print(f"{string} \\\\ \\hline")
    # print("\\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{adjustbox}")
    print("\\end{table}")


class Total:
    def __init__(self):
        self.header = str()
        self.footer = str()
        self.top = str()
        self.rows = []

    def insert_mean(self, header, footer, mean):
        self.header = header
        self.footer = footer
        self.top = mean

    def insert_rows(self, row, replacement):
        modified = row.replace('mean', replacement)
        self.rows.append(modified)

    def write(self, stream):
        stream.write(self.header)
        for row in self.rows:
            stream.write(row)
            stream.write('\n')
        stream.write(self.top)
        stream.write('\n')
        stream.write(self.footer)


datasets = load.datasets
header = load.header
base = load.baseResults
baseAnalytics = load.baseAnalytics
metrics = load.metrics

# for metric in ["dtw"]:#metrics:
#     # ami = Total()
#     # ami_preprocessed = Total()
#     ari = Total()
#     # ari_preprocessed = Total()

#     # ami_preprocessed.insert_mean(*print_ami_table(datasets, metric, True, base))
#     # sys.stdout.write('\n\n\n')
#     # ari_preprocessed.insert_mean(*print_ari_table(datasets, metric, True, base))
#     # sys.stdout.write('\n\n\n')
#     # ami.insert_mean(*print_ami_table(datasets, metric, False, base))
#     # sys.stdout.write('\n\n\n')
       
#     ari.insert_mean(*print_ari_table(datasets, metric, False, base))
#     sys.stdout.write('\n\n\n')
#     # print_run_time_table(datasets, metric, True, base)
#     # sys.stdout.write('\n\n\n')
#     print_run_time_table(datasets, metric, False, base)
#     # print_ami_specific_table(datasets, metric, False, base, baseAnalytics, 1)
#     # print_ari_specific_table(datasets, metric, False, base, baseAnalytics, 1)
#     # print_ami_specific_table(datasets, metric, False, base, baseAnalytics, 2)
#     # print_ari_specific_table(
#     #     datasets, metric, False, base, baseAnalytics, 1)
#     # sys.stdout.write('\n\n\n')
#     # # print_ari_specific_table(
#     #     datasets, metric, True, base, baseAnalytics, 1)
#     # sys.stdout.write('\n\n\n')
#     # print_ami_specific_table(
#     #     datasets, metric, False, base, baseAnalytics, 1)
#     # sys.stdout.write('\n\n\n')
#     # print_ami_specific_table(
#     #     datasets, metric, True, base, baseAnalytics, 1)
#     # sys.stdout.write('\n\n\n')
#     # print_run_time_table(datasets, metric, True, base)
#     # sys.stdout.write('\n\n\n')
#     # print_run_time_table(datasets, metric, False, base)
#     # sys.stdout.write('\n\n\n')
#     for i in range(1, 5):
#         # ami.insert_rows(print_ami_specific_table(
#         #     datasets, metric, False, base, baseAnalytics, i), a[i-1])
#         # ami_preprocessed.insert_rows(print_ami_specific_table(
#         #     datasets, metric, True, base, baseAnalytics, i), a[i-1])
#         ari.insert_rows(print_ari_specific_table(
#             datasets, metric, False, base, baseAnalytics, i), a[i-1])
#         # ari_preprocessed.insert_rows(print_ari_specific_table(
#         #     datasets, metric, True, base, baseAnalytics, i), a[i-1])
#     ari.insert_rows(print_ari_specific_table_bm(
#             datasets, metric, False, base, baseAnalytics, [1,2,3,4], 'mean'), 'avg')
#     ari.insert_rows(print_ari_specific_table_bm(
#         datasets, metric, False, base, baseAnalytics, [1,2,3,4], 'best'), 'best')
#     # sys.stdout.write('\n\n\n')
#     # ami.write(sys.stdout)
#     # sys.stdout.write('\n\n\n')
#     # ami_preprocessed.write(sys.stdout)
#     sys.stdout.write('\n\n\n')
#     ari.write(sys.stdout)
#     sys.stdout.write('\n\n\n')
#     # ari_preprocessed.write(sys.stdout)
#     # sys.stdout.write('\n\n\n')
#     # ami.write(sys.stdout)
#     # sys.stdout.write('\n\n\n')
#     # ami_preprocessed.write(sys.stdout)
#     # sys.stdout.write('\n\n\n')



# for s in ami:
#     sys.stdout.write(s)
# for column in images:
    # printFullResultsPaper(datasets, 'dtw', True, base)
    # printFullResultsPaper(datasets, 'dtw', False, base)
for metric in ["dtw"]:#metrics:
 
    print_ari_specific_table_bm(
            datasets, metric, False, base, baseAnalytics, [1,2,3,4], 'mean')
    
