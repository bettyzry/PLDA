import getpass
import random

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from scipy.spatial.distance import pdist
from deepod.metrics import ts_metrics, point_adjustment, point_adjustment_min
import seaborn as sns
from deepod.utils.utility import get_sub_seqs_label, get_sub_seqs_label2
from testbed.utils import import_ts_data_unsupervised
from sklearn.preprocessing import normalize
dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'


def zscore(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))  # 最值归一化


def get_label(score, threshold):
    label = np.zeros(len(score))
    index = np.where(score > threshold)[0]
    label[index] = 1
    return label


def plot_case_study():
    fontsize = 13

    new_df = pd.read_csv('./plotsource/case_study.csv')

    true = new_df[new_df.label == 0]
    false = new_df[new_df.label == 1]
    hard = new_df[new_df.label == 2]

    dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'
    dataset_root_DC = f'/home/{getpass.getuser()}/dataset/5-TSdata/_DCDetector/'
    dataset = 'PUMP'
    data_pkg = import_ts_data_unsupervised(dataset_root, dataset_root_DC,
                                           dataset, entities='FULL',
                                           combine=1)
    mycolor = ["#1E90FF", "#FF7256", '#85b243']
    train_lst, test_lst, label_lst, name_lst = data_pkg

    for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
        seq_len = 30
        index='hard'
        l = {'true':true, 'false':false, 'hard':hard}
        df2 = l[index]
        real_loc = df2['loc'].values
        loss = df2['loss'].values
        param = df2['param'].values
        cnum = [3, 4, 5, 6]
        # cnum = [i for i in range(test_data.shape[1])]
        if index=='false':
            id = 33
            # real_loc = real_loc[id:]
            real_loc = [real_loc[id]]
            color = mycolor[1]
        elif index=='hard':
            id = 106
            # id = 300
            real_loc = [real_loc[id]]
            # real_loc = real_loc[id:]
            color = mycolor[2]
        else:
            id = 3512
            # real_loc = real_loc[id:]
            real_loc = [real_loc[id]]
            color = mycolor[0]
        for ii, loc in enumerate(real_loc):
            # if param[id+ii] <0.5:
            #     if loss[id+ii] > 0.6:
                    plt.figure(figsize=(3, 2))
                    # plt.figure(figsize=(3, 16))
                    loc = int(loc)
                    x = test_data[max(0, loc-50): loc + seq_len + 50]
                    for jj, c in enumerate(cnum):
                        # plt.rc('axes', linewidth=1.5)
                        ax0 = plt.subplot(len(cnum), 1, jj+1)
                        plt.plot(x[:, c], color='black')
                        ax0.text(0, 0.05, "Feature %d" % (jj + 1), fontdict={'fontsize': fontsize * 0.8})  # bbox={'facecolor': 'g', 'alpha': 0.5}
                        # plt.ylabel('Feature %d' % (jj+1), fontsize=fontsize, rotation= 90)
                        # xtick = [str(i) for i in range(loc-80, loc + seq_len + 80, 40)]
                        # if jj == 3:
                            # plt.xticks([0, 40, 80, 120], labels=xtick,fontsize=fontsize*0.8)
                        plt.xticks([])
                        plt.yticks([0, 1], fontsize=fontsize*0.8)
                        plt.xlim(-2, None)
                        plt.ylim(-0.2, 1.2)
                        plt.yticks([])
                        ax0.add_patch(plt.Rectangle((50, -0.5), seq_len, 2, fill=True, color=color, alpha=0.5))
                    # plt.tight_layout()
                    plt.gcf().subplots_adjust(left=0.04, top=0.85, bottom=0.05, right=0.96)
                    # plt.suptitle('Loss: %.4f, Param: %.4f, %s' % (loss[id+ii], param[id+ii], ii), fontsize=fontsize)
                    plt.suptitle('Loss: %.4f, Param: %.4f' % (loss[id+ii], param[id+ii]), fontsize=fontsize)
                    plt.savefig('./plotsource/case_%s.png'%index, dpi=600)
                    plt.show()



def plot_distribution():
    fontsize = 13
    data_root = '/home/xuhz/zry/DeepOD-new/plotsource/PUMP_myfunc20.0_10_0.50.csv'
    df = pd.read_csv(data_root)
    df = df.dropna()
    step = "1"
    loss = df['loss' + step].values
    df['loss' + step] = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    adjloss = point_adjustment(df['yseq0'].values, df['loss' + step].values)
    dis = df['dis' + step].values
    df['dis' + step] = (dis - np.min(dis)) / (np.max(dis) - np.min(dis))
    adjdis = point_adjustment(df['yseq0'].values, df['dis' + step].values)
    ytrue = df['yseq'+str(int(step)-1)].values
    ytrue2 = np.copy(ytrue)

    threshold= np.percentile(adjloss, 73)
    ypred  = get_label(adjloss, threshold)
    loc1 = np.where(ypred==1)[0]
    loc2 = np.where(ytrue==0)[0]
    loc = np.array(list(set(loc1) & set(loc2)))
    ytrue2[loc] = 2

    new_df = pd.DataFrame()
    new_df['label'] = ytrue
    new_df['rlabel'] = ytrue2
    new_df['loss'] = adjloss
    new_df['dis'] = adjdis

    true = new_df[new_df.label == 0]
    false = new_df[new_df.label == 1]

    mycolor = ["#1E90FF", "#FF7256"]
    current_palette = sns.color_palette(mycolor)

    plt.figure(figsize=(10, 2.5), dpi=600)
    grid = plt.GridSpec(6, 17, wspace=0.05, hspace=0.05)
    plt.gcf().subplots_adjust(left=0.08, top=0.95, bottom=0.1, right=0.95)

    ax0 = plt.subplot(grid[1:5, 0:4])
    sns.kdeplot(true['loss'], shade=True, color=mycolor[0], label='Normal')
    sns.kdeplot(false['loss'], shade=True, color=mycolor[1], label='Abnormal')
    ax0.set_ylabel('Density', fontsize=fontsize)
    ax0.set_xlabel('Loss Value', fontsize=fontsize)
    ax0.tick_params(labelsize=fontsize * 0.8)
    ax0.add_patch(plt.Rectangle((0.765, 0), 0.15, 3.4, color='black', fill=False, linewidth=2))
    ax0.text(-0.05, 4.5, "Hard Samples", fontdict={'fontsize': fontsize * 0.8})  # bbox={'facecolor': 'g', 'alpha': 0.5}

    plt.quiver(0.6, 4.5, 0.15, -1.05, angles='xy', scale_units='xy', scale=1,  width=0.01)
    ax0.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax0.set_yticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
    ax0.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.48, 1.3), fontsize=fontsize * 0.8, handletextpad=0.4, columnspacing=1)

    # 4.1 绘制长度的边缘分布图
    ax1 = plt.subplot(grid[0, 6:10])
    # ax1.spines[:].set_linewidth(0.4)  # 设置坐标轴线宽
    ax1.tick_params(width=0.6, length=2.5, labelsize=8)  # 设置坐标轴刻度的宽度与长度、刻度标注的字体大小

    # sns.kdeplot(true['loss'], shade=True, color="b", label='Normal')
    # sns.kdeplot(false['loss'], shade=True, color="r", label='Abnormal')

    sns.kdeplot(data=new_df, x="loss", hue="label",
                fill=True, common_norm=False, legend=False,
                alpha=.5, linewidth=0.5, ax=ax1, palette=current_palette)  # 边缘分布图
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_xticks([])
    ax1.set_xlabel("")
    ax1.set_yticks([])
    ax1.set_ylabel("")

    # # 4.2 绘制宽度的边缘分布图
    ax2 = plt.subplot(grid[1:5, 10])
    # ax2.spines[:].set_linewidth(0.4)
    ax2.tick_params(width=0.6, length=2.5, labelsize=8)
    sns.kdeplot(data=new_df, y="dis", hue="label",
                fill=True, common_norm=False, legend=False,
                alpha=.5, linewidth=0.5, ax=ax2, palette=current_palette)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xticks([])
    ax2.set_xlabel("")
    ax2.set_yticks([])
    ax2.set_ylabel("")

    # 4.3 绘制二元分布图（散点图）
    true = new_df[new_df.rlabel == 0]
    false = new_df[new_df.rlabel == 1]
    hard = new_df[new_df.rlabel == 2]

    ax3 = plt.subplot(grid[1:5, 6:10])
    # ax3.spines[:].set_linewidth(0.4)
    ax3.tick_params(width=0.6, length=2.5, labelsize=8)
    ax3.grid(linewidth=0.6, ls='-.', alpha=0.4)
    ax3.scatter(x=true['loss'], y=true['dis'], s=100, alpha=1, marker='*',
                edgecolors='w', linewidths=0.5, label='Simple Normal Sample', color=mycolor[0])
    ax3.scatter(x=hard['loss'], y=hard['dis'], s=100, alpha=1, marker='*',
                edgecolors='w', linewidths=0.5, label='Hard Normal Sample', color='#85b243')
    ax3.scatter(x=false['loss'], y=false['dis'], s=60, alpha=1, marker='^',
                edgecolors='w', linewidths=0.5, label='Abnormal Sample', color=mycolor[1])

    ax3.set_xlabel("Loss Behavior", fontsize=fontsize, x=0.55)
    ax3.set_ylabel("Parameter Behavior", fontsize=fontsize, y=0.55)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax3.set_ylim(-0.1, 1.1)
    plt.tick_params(labelsize=fontsize*0.8)


    # 4.4 画线
    plt.plot([0.765, 0.765], [-0.1, 1.1], linewidth=1, color='#000000', linestyle='dashed', label='Decision Boundary of Loss Behavior')
    plt.plot([-0.1, 1.1], [0.25, 0.25], linewidth=1, color='#000000', linestyle='dotted', label='Decision Boundary of Parameter Behavior')
    # plt.text(0.8, 0.075, "Hard\nSamples", fontdict={'fontsize': fontsize * 0.6})  # bbox={'facecolor': 'g', 'alpha': 0.5}
    # plt.text(0.78, 0.8, "Anomalies", fontdict={'fontsize': fontsize * 0.6})  # bbox={'facecolor': 'g', 'alpha': 0.5}

    # ax3.legend(fontsize=fontsize*0.8, labelspacing=0.35, handleheight=1.2, handletextpad=0, loc=(0.98, 1.01), frameon=False)
    ax3.legend(ncol=1, loc='upper center', bbox_to_anchor=(2.13, 0.95), fontsize=fontsize*0.8, handletextpad=0.0, columnspacing=0.1)

    plt.savefig('./plotsource/loss.png', dpi=600)
    plt.show()


def plot_pollute():
    from collections import OrderedDict
    palette = sns.color_palette('deep', 7)
    temp = palette[0]
    palette[0] = palette[3]
    palette[3] = temp
    palette[4] = palette[1]
    palette[5] = palette[0]
    sns.set_theme(palette=palette, style='ticks')

    dataset = '/home/xuhz/zry/PLDA/plotsource/polluted.csv'
    alldf = pd.read_csv(dataset)       # .values[:, 1:]
    ts = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

    # data = ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT',
    #          'PUMP', 'DASADS', 'Fault', 'Gait', 'Heart Sbeat']
    TSAD = ['TcnED', 'TranAD', 'NeuTral', 'NCAD']
    # TSAD = ['NCAD', 'NeuTral', 'TranAD', 'TcnED']
    DA = ['RODA', 'ORIG', 'PI', 'LOSS', 'ORIG', 'RODA']
    DA2 = ['PLDA', 'ORIG', 'PI', 'LOSS', 'ORIG', 'PLDA']
    # DA = ['LOSS', 'PI', 'ORIG', 'PLDA']

    marker_lst = ['o', 'x', '^', 's', 'x', 'o', 'p']
    fontsize=13
    fig = plt.figure(figsize=(10, 2.5))
    for ii, tsad in enumerate(TSAD):
        plt.subplot(1, 4, ii+1)
        plt.grid(ls='--')
        tsaddf = alldf[alldf['TSAD'] == tsad]
        for jj, da in enumerate(DA):
            df = tsaddf[tsaddf['DA'] == da]
            df = df.reset_index(drop='True')
            avg = df['F1'].values
            std = df['std'].values
            r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  # 上方差
            r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  # 下方差
            plt.plot(ts, avg, linewidth=1.5, marker=marker_lst[jj])
            plt.fill_between(ts, r1, r2, alpha=0.1)
        if ii == 0:
            min = 0.65
            max = 0.9
            ticks = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            ticks2 = ['0.65', '0.70', '0.75', '0.80', '0.85', '0.90']
        elif ii == 1:
            min = 0.5
            max = 0.9
            ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
            ticks2 = ['0.50', '0.60', '0.70', '0.80', '0.90']
        elif ii == 2:
            min = 0.55
            max = 0.75
            ticks = [0.55, 0.6, 0.65, 0.7, 0.75]
            ticks2 = ['0.55', '0.60', '0.65', '0.70', '0.75']
        else:
            min = 0.65
            max = 0.85
            ticks = [0.65, 0.7, 0.75, 0.8, 0.85]
            ticks2 = ['0.65', '0.70', '0.75', '0.80', '0.85']
        plt.ylim(min, max)
        plt.yticks(ticks, ticks2, fontsize=fontsize*0.8)
        plt.title(TSAD[ii], fontsize=fontsize)
        plt.xticks([0, 5, 10, 15, 20], ['0%', '5%', '10%', '15%', '20%'], fontsize=fontsize*0.8)
        plt.xlabel('Contamination (%)', fontsize=fontsize)
        if ii == 0:
            plt.ylabel('F1-Score', fontsize=fontsize)

    plt.subplots_adjust(left=0.08, bottom=0.35, right=0.98, top=0.85, wspace=0.35, hspace=None)
    # fig.subplots_adjust(left=0.18, bottom=0.2, right=0.98, top=0.95)
    fig.legend(DA2[: 4], ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.01))

    # plt.tight_layout()
    plt.savefig('./plotsource/diff_Orate.png', dpi=600)
    plt.show()


def plot_hyparam():
    from matplotlib import rcParams
    # palette = sns.color_palette('deep', 15)
    # # temp = palette[4]
    # palette[0] = palette[11]
    # palette[4] = palette[10]
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#E377C2', '#1C4D7A', '#845454', '#DCC3B3', '#17BECF', '#707B90']
    sns.set_theme(palette=colors, style='ticks')
    pathe = '/home/xuhz/zry/PLDA/plotsource/param-e.csv'
    patha = '/home/xuhz/zry/PLDA/plotsource/param-a.csv'
    pathk = '/home/xuhz/zry/PLDA/plotsource/param-k.csv'
    dfe = pd.read_csv(pathe).values
    dfa = pd.read_csv(patha).values
    dfk = pd.read_csv(pathk).values
    l = [dfe, dfa, dfk]
    # avgs = df['f1'].values
    # stds = df['std'].values
    dataset = ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT',
             'PUMP', 'DASADS', 'Fault', 'Gait', 'Heart Sbeat']
    name = ['e', r'$\alpha$', 'k']
    e = [1, 2, 4, 8, 16, 32]
    e2 = [1, 2, 3, 4, 5, 6]
    a = [0.1, 0.3, 0.5, 0.7, 0.9]
    k = [200, '', 800, '', '3,200', '', '12,800']
    k2 = [1, 2, 3, 4, 5, 6, 7]
    xl = {'0': e2, '1': a, '2': k2}
    xtick = {'0': e, '1': a, '2': k}
    indexs = [6, 5, 7]

    marker_lst = ['o', 'x', '^', 's', 'v', 'd', '<', 'p', '>', 'P']
    fontsize = 13
    fig = plt.figure(figsize=(8.5, 2.7))
    # avgs = []
    for jj in range(3):
        plt.subplot(1, 3, jj+1)
        df = l[jj]
        for ii in range(10):
            index = np.array([indexs[jj]*ii+j for j in range(indexs[jj])])
            d = df[index, :]
            avg = d[:, 1]
            plt.plot(xl[str(jj)], avg, linewidth=1.5, label=dataset[ii], marker=marker_lst[ii])
        if jj == 0:
            plt.ylabel('F1-Score', fontsize=fontsize)
        plt.xlabel(name[jj], fontsize=fontsize)
        plt.xticks(fontsize=fontsize*0.8)
        plt.yticks(fontsize=fontsize*0.8)
        plt.ylim(0.6, 1)
        plt.xticks(xl[str(jj)], xtick[str(jj)])
        plt.grid(ls='--')

    fig.legend(dataset, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0), fontsize=fontsize*0.8)
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.08, top=0.92, bottom=0.4, right=0.97)
    plt.savefig('./plotsource/parameter.png', dpi=600)
    plt.show()


def plot_scalability():
    fontsize = 13
    name = 'NeuTraLTS'
    funcs = [' myfunc', ' norm', ' ICLM21', ' Arxiv22', ' myfunc']
    funname = ['PLDA', 'ORIG', 'PI', 'LOSS']
    d = [8, 16, 32, 64, 128, 256, 512]
    l = [8000, 16000, 32000, 64000, 128000, 256000, 512000]
    path = '/home/xuhz/zry/PLDA/plotsource/Scalability.csv'
    marker_lst = ['o', 'x', '^', 's', 'v', 'd', '<', 'p', '>', 'P']
    df = pd.read_csv(path)
    df = df[df.Model == name]

    colors = ['#C44E52', '#DD8452', '#55A868', '#4C72B0', '#C44E52', '#1C4D7A', '#845454', '#DCC3B3', '#17BECF',
              '#707B90']
    sns.set_theme(palette=colors, style='ticks')

    fig = plt.figure(figsize=(6, 2.2))
    plt.subplot(1, 2, 1)
    for ii, f in enumerate(funcs):
        df1 = df[df['funcs'] == f]
        df1 = df1.reset_index(drop='True')
        plt.plot([0, 1, 2, 3, 4, 5, 6], df1['times'][len(d)+len(l):2*len(d)+len(l)].values, linewidth=1.5, marker=marker_lst[ii%4])
    plt.xticks([0, 1, 2, 3, 4, 5, 6], [8, '', 32, '', 128, '', 512], fontsize=fontsize*0.8)
    plt.yticks(fontsize=fontsize*0.8)
    plt.xlabel('Dimensionality', fontsize=fontsize)
    plt.ylabel('Execution Time/s', fontsize=fontsize)
    plt.ylim(30, 10000)
    plt.yscale('log')
    plt.grid(ls='--')

    plt.subplot(1, 2, 2)
    for ii, f in enumerate(funcs):
        df1 = df[df['funcs'] == f]
        df1 = df1.reset_index(drop='True')
        plt.plot([0, 1, 2, 3, 4, 5, 6], df1['times'][2*len(d)+len(l):2*len(d)+2*len(l)].values, linewidth=1.5, marker=marker_lst[ii%4])
    plt.xlabel('Data Size', fontsize=fontsize)
    # plt.ylabel('Execution Time/s', fontsize=fontsize)
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['8,000', '', '32,000', '', '128,000', '', '512,000'], rotation=20, fontsize=fontsize*0.8)
    plt.yticks(fontsize=fontsize*0.8)
    plt.ylim(30, 10000)
    plt.yscale('log')
    # plt.yticks([])
    plt.grid(ls='--')

    fig.legend(funname[: 4], ncol=1, loc='right', bbox_to_anchor=(1, 0.62), fontsize=fontsize*0.8)
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.12, top=0.88, bottom=0.35, right=0.82)
    plt.savefig('./plotsource/scalability.png', dpi=600)
    plt.show()



def plot_fill_samples():
    fontsize = 13
    plt.figure(figsize=(4, 2.2))
    sns.set_style("whitegrid")
    sns.despine(top=True, right=True, left=True, bottom=True)

    path = '/home/xuhz/zry/PLDA/plotsource/fill_samples.csv'
    df = pd.read_csv(path)
    palette_colors = {"HS": "#85b243", "AC": "#FF7256"}

    sns.boxplot(x='epoch', y='value', hue='kind', data=df, showfliers=False, palette=palette_colors)  # x轴和y轴的数据，data是数据框，y参数中使用f-string格式化i的值来指定列名
    plt.yticks([0, 2, 4, 6, 8, 10, 12, 14], fontsize=fontsize * 0.8)
    plt.xticks(fontsize=fontsize * 0.8)
    plt.xlabel('#Epochs', fontsize=fontsize)
    plt.ylabel('Proportion (%)', fontsize=fontsize)
    plt.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.27, 0.77), fontsize=fontsize * 0.8)

    plt.grid(ls='--')
    plt.gcf().subplots_adjust(left=0.16, top=0.95, bottom=0.23, right=0.95)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')

    plt.savefig('./plotsource/propotion.png', dpi=600)
    plt.show()  # 显示图形
    return


if __name__ == '__main__':
    plot_distribution()             # loss和dis的二维分布图
    # plot_case_study()               # 绘制case study
    # plot_scalability()              # 可扩展性
    # plot_pollute()                  # 不同方法在不同污染率下的性能
    # plot_hyparam()                  # 参数敏感性实验
    # plot_fill_samples()             # 难例和异常的过滤情况