import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from test_parameters import *

color_list = ['r', 'b', 'g', 'orange', 'indigo']

def boxplot(csv_file, n_methods, method_id):
    with open(csv_file)as f:
        f_csv = csv.reader(f)
        f_csv = list(zip(*f_csv))
        for i, row in enumerate(f_csv):
            print(row)
            row = list(map(float, row))
            bp = plt.boxplot(row, positions=[i*(n_methods+2)+method_id], widths=0.8, patch_artist=True, sym='.')
            set_box_color(bp, color_list[method_id])
    return bp

def set_box_color(bp, color):
    plt.setp(bp['boxes'], facecolor=color, linewidth=0.5)
    plt.setp(bp['whiskers'], color=color, linewidth=0.5)
    plt.setp(bp['caps'], color=color, linewidth=0.5)
    plt.setp(bp['medians'], color='k', linewidth=0.5)
    plt.setp(bp['fliers'], markeredgecolor=color, markersize=0.5)


def plot_comparison_result():
    perf_filetype = 'results'
    time_filetype = 'planning_time'
    perf_file_list = []
    time_file_list = []
    method_name_list = ['ts.(4)', 'g.(800)', 'CMA-ES', 'RIG-tree', 'RAOr']
    n_method = len(method_name_list)

    method_param_list = ['ts_15', 'greedy', 'CMAES', 'RIG', 'RAOr']
    budget_list = [10, 12, 6, 8]
    legend_list = []
    for _, _, files in os.walk(f'result/comparison_analysis'):
        for f in files:
            if perf_filetype in f:
                perf_file_list.append(f)
            if time_filetype in f:
                time_file_list.append(f)
    perf_file_list.sort()
    time_file_list.sort()
    print(perf_file_list)

    fig = plt.figure(figsize=(3.7, 1.68))
    for file_list in [perf_file_list, time_file_list]:
        for j, param in enumerate(method_param_list):
            X = []
            Y = []
            var = []
            i = 0
            for f in file_list:
                if param in f:
                    print(f, param)
                    csv_file = f'result/comparison_analysis/'+f
                    with open(csv_file)as csv_f:
                        f_csv = csv.reader(csv_f)
                        f_csv = list(f_csv)
                        data = []
                        for item in f_csv:
                            item = list(map(float, item))
                            data.append(item)
                    # perf = np.mean(np.array(data).reshape(-1,1))
                    # std = np.std(np.array(data).reshape(-1,1))
                    perf = np.array(data).reshape(-1,1)
                    Y.append(perf)
                    X.append(budget_list[i])
                    # var.append(std)
                    i+=1
            idx = np.array(X).argsort()
            X = np.array(X)[idx]
            Y = np.array(Y)[idx]
            # var = np.array(var)[idx]
            if file_list == perf_file_list:
                plt.subplot(1,2,1)
                plt.ylim(-2,75)
                plt.yticks([10,30,50,70])
                plt.ylabel('Tr(P)', fontdict={'family':'Times New Roman', 'size':8})
            else:
                plt.subplot(1,2,2)
                plt.ylabel('Total Planning Time/s', fontdict={'family':'Times New Roman', 'size':8})
            plt.xlabel('Budget', fontdict={'family':'Times New Roman', 'size':8})
            # line = plt.plot(X, Y, marker='x', markersize=3)
            # legend_list.append(line[0])
            # plt.fill_between(X, Y+var, Y-var, alpha=0.15)
            bp = generalization_boxplot(Y, n_method, j)
            legend_list.append(bp['boxes'][0])

        # plt.xticks([6,8,10,12], fontproperties= 'Times New Roman', size=8)
        ticks = [6,8,10,12]
        plt.xticks(
            np.arange(0 + (n_method - 1) / 2, (n_method + 2) * len(ticks)+ (n_method- 1) / 2, step=n_method + 2),
            ticks, fontproperties= 'Times New Roman', size=8)
        plt.yticks(fontproperties='Times New Roman', size=8)
        plt.tick_params(bottom=False, left=False, axis='both', pad=0.1)
        if file_list == perf_file_list:
                plt.legend(legend_list, method_name_list, borderaxespad=0.1, handlelength=0.5, prop={'family': 'Times New Roman', 'size': 8},
                           frameon=False, labelspacing=0.1)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'result/comparison_analysis.pdf')


def plot_trajectory_history_result():
    filetype = '.csv'
    file_list = []
    method_list = ['ts.(4)', 'g.(800)', 'CMA-ES', 'RIG-tree', 'RAOr']
    method_param_list = ['ts_', 'greedy', 'CMAES', 'RIG', 'RAOr']
    legend_list = []
    for _, _, files in os.walk(f'result/trajectory_history'):
        for f in files:
            if filetype in f:
                file_list.append(f)
    file_list.sort(reverse=True)

    fig = plt.figure(figsize=(3.8, 1.68))

    for i, param in enumerate(method_param_list):
        for csv_file in file_list:
            if param in csv_file:
                print(csv_file)
                csv_file = f'result/trajectory_history/'+csv_file
                trajectory_history = pd.read_csv(csv_file)
                trajectory_history = trajectory_history.sort_values('budget')
                if 'greedy' in csv_file:
                    plt.subplot(1,2,1)
                    trajectory_mean = trajectory_history.rolling(200, on='budget', min_periods=20, center=True).mean()
                    trajectory_std = trajectory_history.rolling(200, on='budget', min_periods=20, center=True).std()
                    line = plt.plot(trajectory_mean.budget, trajectory_mean.obj, color=color_list[i], linewidth=0.8, zorder=10-i)
                    # legend_list.append(line[0])
                    plt.fill_between(trajectory_mean.budget, trajectory_mean.obj - trajectory_std.obj,
                                     trajectory_mean.obj + trajectory_std.obj, color=color_list[i], alpha=0.15)
                    plt.subplot(1, 2, 2)
                    line = plt.plot(trajectory_mean.budget, trajectory_mean.obj2, color=color_list[i], linewidth=0.8)
                    legend_list.append(line[0])
                    plt.fill_between(trajectory_mean.budget, trajectory_mean.obj2 - trajectory_std.obj2,
                                     trajectory_mean.obj2 + trajectory_std.obj2, color=color_list[i], alpha=0.15)
                else:
                    plt.subplot(1,2,1)
                    trajectory_mean = trajectory_history.rolling(500, on='budget', center=True, min_periods=300).mean()
                    trajectory_std = trajectory_history.rolling(500, on='budget', center=True, min_periods=300).std()
                    # trajectory_min = trajectory_history.rolling(20, on='budget', min_periods=5 ).min()
                    line = plt.plot(trajectory_mean.budget, trajectory_mean.obj, color=color_list[i], linewidth=0.8, zorder=10-i)
                    # legend_list.append(line[0])
                    plt.fill_between(trajectory_mean.budget, trajectory_mean.obj-trajectory_std.obj, trajectory_mean.obj+trajectory_std.obj, color=color_list[i], alpha=0.15)
                    plt.subplot(1,2,2)
                    line = plt.plot(trajectory_mean.budget, trajectory_mean.obj2, color=color_list[i], linewidth=0.8, zorder=10-i)
                    legend_list.append(line[0])
                    plt.fill_between(trajectory_mean.budget, trajectory_mean.obj2 - trajectory_std.obj2,
                                     trajectory_mean.obj2 + trajectory_std.obj2, color=color_list[i], alpha=0.15)
                plt.subplot(1,2,1)
                plt.xlabel('Used budget', fontdict={'family':'Times New Roman', 'size':8})
                plt.yscale('log')
                plt.tick_params(bottom=False, left=False, axis='both', pad=0.1)
                plt.ylabel('Tr(P)', fontdict={'family':'Times New Roman', 'size':8})
                plt.yticks(size=8)
                plt.xticks([0,2,4,6,8,10], fontproperties='Times New Roman', size=8)
                plt.subplot(1,2,2)
                plt.xlabel('Used budget', fontdict={'family':'Times New Roman', 'size':8})
                plt.ylabel('RMSE', fontdict={'family':'Times New Roman', 'size':8})
                plt.xticks([0,2,4,6,8,10], fontproperties='Times New Roman', size=8)
                plt.yticks(fontproperties='Times New Roman', size=8)
                plt.tick_params(bottom=False, left=False, axis='both', pad=0.1)
                # plt.tick_params(direction='in')
                plt.subplot(1,2,1)
                plt.legend(legend_list, method_list, labelspacing=0.1, borderaxespad=0.1, handlelength=0.5, prop={'family':'Times New Roman', 'size':8}, frameon=False)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'result/trajectory_history_comparison.pdf')

def plot_generalization_analysis():
    perf_filetype = 'results'
    time_filetype = 'planning_time'
    perf_file_list = []
    time_file_list = []
    method_name_list1 = ['g.(200)', 'g.(400)', 'g.(600)', 'g.(800)']
    method_name_list2 = ['g.(400)', 'ts.(4)','ts.(8)','ts.(16)']

    method_param_list1 = ['greedy_200','greedy_400', 'greedy_600', 'greedy_800']
    method_param_list2 = ['greedy_400', 'ts_15_4', 'ts_15_8', 'ts_15_16']
    budget_list = [10, 12, 6, 8]
    legend_list1 = []
    legend_list2 = []
    for _, _, files in os.walk(f'result/generalization_analysis'):
        for f in files:
            if perf_filetype in f:
                perf_file_list.append(f)
            if time_filetype in f:
                time_file_list.append(f)
    perf_file_list.sort()
    time_file_list.sort()
    print(perf_file_list)

    fig = plt.figure(figsize=(3.7, 3.2))
    for file_list in [perf_file_list, time_file_list]:
        for method_param_list in [method_param_list1,method_param_list2]:
            for j, param in enumerate(method_param_list):
                X = []
                Y = []
                var = []
                i = 0
                for f in file_list:
                    if param in f:
                        print(f)
                        csv_file = f'result/generalization_analysis/'+f
                        with open(csv_file)as csv_f:
                            f_csv = csv.reader(csv_f)
                            f_csv = list(f_csv)
                            data = []
                            for item in f_csv:
                                item = list(map(float, item))
                                data.append(item)
                        # perf = np.mean(np.array(data).reshape(-1,1))
                        # std = np.std(np.array(data).reshape(-1,1))
                        perf = np.array(data).reshape(-1,1)
                        Y.append(perf)
                        X.append(budget_list[i])
                        # var.append(std)
                        i+=1
                idx = np.array(X).argsort()
                X = np.array(X)[idx]
                Y = np.array(Y)[idx]
                # var = np.array(var)[idx]
                if file_list == perf_file_list:
                    if method_param_list == method_param_list1:
                        plt.subplot(2,2,1)
                        legend_list = legend_list1
                    if method_param_list == method_param_list2:
                        plt.subplot(2,2,3)
                        legend_list = legend_list2
                    plt.ylabel('Tr(P)', fontdict={'family':'Times New Roman', 'size':8})
                else:
                    if method_param_list == method_param_list1:
                        plt.subplot(2,2,2)
                    if method_param_list == method_param_list2:
                        plt.subplot(2,2,4)
                    plt.ylabel('Total Planning Time/s', fontdict={'family':'Times New Roman', 'size':8})
                plt.xlabel('Budget', fontdict={'family':'Times New Roman', 'size':8})
                # line = plt.plot(X, Y, marker='x', markersize=3)
                # legend_list.append(line[0])
                # plt.fill_between(X, Y+var, Y-var, alpha=0.15)
                bp = generalization_boxplot(Y, len(method_param_list), j)
                legend_list.append(bp['boxes'][0])

            # plt.xticks([6,8,10,12], fontproperties= 'Times New Roman', size=8)
            ticks = [6,8,10,12]
            plt.xticks(
                np.arange(0 + (4 - 1) / 2, (4 + 2) * len(ticks)+ (4- 1) / 2, step=4 + 2),
                ticks, fontproperties= 'Times New Roman', size=8)
            plt.yticks(fontproperties='Times New Roman', size=8)
            plt.tick_params(bottom=False, left=False, axis='both', pad=0.1)
            if file_list == perf_file_list:
                if method_param_list == method_param_list1:
                    plt.legend(legend_list1, method_name_list1, borderaxespad=0.1, handlelength=0.5, prop={'family': 'Times New Roman', 'size': 8},
                               frameon=False, labelspacing=0.1)
                else:
                    plt.legend(legend_list1,method_name_list2, borderaxespad=0.1, handlelength=0.5, prop={'family':'Times New Roman','size':8}, frameon=False, labelspacing=0.1)
        plt.tight_layout()
    # plt.show()
    plt.savefig(f'result/generalization_analysis.pdf')

def generalization_boxplot(data, n_methods, method_id):
    for i, row in enumerate(data):
        bp = plt.boxplot(row, positions=[i * (n_methods + 2) + method_id], widths=0.8, patch_artist=True,
                         sym='.')
        set_box_color(bp, color_list[method_id])
    return bp

if __name__ == '__main__':
   plot_comparison_result()
   plot_trajectory_history_result()
   plot_generalization_analysis()
