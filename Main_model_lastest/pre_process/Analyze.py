# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from collections import Counter
'''
analyze the turns of dial
'''
from Main_model_lastest.configs_manage import preProcess
import pandas as pd

def draw_fig(data):
    # 柱状图
    x = []
    y = []
    for i in data:
        x.append(i[0])
        y.append(i[1])

    plt.bar(range(len(y)), y, fc='r', tick_label=x)

    # 设置横坐标轴的标签说明
    plt.xlabel('Turn')
    # 设置纵坐标轴的标签说明
    plt.ylabel('Number')
    # 设置标题
    plt.title('Turn Number of Conversation')
    # 绘图
    plt.show()

    # 饼图
    fig = plt.figure()
    plt.pie(y, labels=x)
    # plt.pie(X,labels=labels,autopct='%1.2f%%') # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.title("Pie chart")
    plt.show()


def split_test(base_path):
    # personaChat、dailyDialog、hollE
    names = ['personaChat','dailyDialog','hollE','WOW']
    for i in names:
        data = i
        df = pd.read_csv(base_path+data+'/data_dialog_knowledge.csv')
        sum_cnt = len(df)
        split_rate = 0.3
        split_num = int(sum_cnt*split_rate)
        test_df = df[0:split_num]
        train_df = df[split_num:]
        test_df.to_csv(base_path+data+'/data_dialog_knowledge_test.csv')
        train_df.to_csv(base_path+data+'/data_dialog_knowledge_train.csv')


if __name__ == '__main__':
    pre_con = preProcess('dailyDialog')
    data_path = pre_con.data_path
    dataChoice = pre_con.dataChoice
    draw_flag = False

    if draw_flag:
        with open(data_path+'turn_num.txt','r') as f:
            turns = f.read()
        # print(turns)
        if dataChoice == 'dailyDialog':
            ts = eval(turns.split('\n')[0])
        elif dataChoice == 'personaChat':
            ts = eval(turns.split('\n')[0])+eval(turns.split('\n')[1])

        # 用counter进行统计
        cv = Counter(ts)
        # 对字典的键值对元组列表排序，按元组的第1个元素排序，也就是 key
        re = sorted(cv.items(), key=lambda obj: obj[0])
        print(re)
        draw_fig(data=re)
    # 是否事先划分测试集和训练集
    split_flag = True
    if split_flag:
        split_test(base_path=pre_con.data_base_path)