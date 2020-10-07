import random
import GlobalVariable as gl
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import math
import numpy as np


def shuffle_data(data):
    feature_num = len(data[0])-1
    data_num = len(data)
    feature_num_average = int(feature_num / gl.VIEW_NUM)
    for i in range(0, gl.VIEW_NUM - 1):
        for _ in range(0, gl.RANDOM_CHANGE_NUM):
            row1 = int(random.randrange(0, data_num))
            row2 = int(random.randrange(0, data_num))
            data[row1][feature_num_average * i:feature_num_average * (i + 1)], data[row2][feature_num_average * i:feature_num_average * (i + 1)] = data[row2][feature_num_average * i:feature_num_average * (i + 1)], data[row1][feature_num_average * i:feature_num_average * (i + 1)]
            data[row1][feature_num] = 1
            data[row2][feature_num] = 1
    for _ in range(0, gl.RANDOM_CHANGE_NUM):
        row1 = int(random.randrange(0, data_num))
        row2 = int(random.randrange(0, data_num))
        data[row1][feature_num_average*(gl.VIEW_NUM-1):feature_num], data[row2][feature_num_average*(gl.VIEW_NUM-1):feature_num] = data[row2][feature_num_average*(gl.VIEW_NUM-1):feature_num], data[row1][feature_num_average*(gl.VIEW_NUM-1):feature_num]
        data[row1][feature_num] = 1
        data[row2][feature_num] = 1
    return data


def read_file_new(file_name):
    f = open(file_name, 'r')
    data_str = f.readlines()
    f.close()
    data = []
    for line in data_str:
        line = list(map(float, line.strip().split(',')))
        data.append(line)
    return data


def load_data_from_ready_file(data_name, iterator):
    data_each_view = []
    feature_num_each_view = []
    file_dir = 'data/' + data_name + '/' + data_name + '_' + str(gl.VIEW_NUM) + '/data' + str(iterator+1) + '/'
    for i in range(gl.VIEW_NUM):
        v_file = file_dir + 'v' + str(i+1) + '.txt'
        data_each_view.append(read_file_new(v_file))
        feature_num_each_view.append(len(data_each_view[i][0]))
    l_file = file_dir + 'label.txt'
    label = np.array(read_file_new(l_file))[:, 0].tolist()
    return data_each_view, feature_num_each_view, label


def load_data_from_several_file(file_name_list, label_file_name):
    data_each_view = []
    feature_num_each_view = []
    file_num = len(file_name_list)
    for i in range(0, file_num):
        data_each_view.append([])
        with open(file_name_list[i], 'r') as f:
            data_str = f.readlines()
            feature_num_each_view.append(len(data_str[0].strip().split('	')))
            for line in data_str:
                line = list(map(float, line.strip().split('	')))
                data_each_view[i].append(line)
    if label_file_name is None:
        return data_each_view, feature_num_each_view, None
    label = []
    with open(label_file_name, 'r') as f:
        label_str = f.readlines()
        for line in label_str:
            label.append(int(line))
    return data_each_view, feature_num_each_view, label


def data_join(data_each_view):
    data_all_view = []
    view_num = len(data_each_view)
    data_num = len(data_each_view[0])
    for j in range(0, data_num):
        data_all_view.append([])
        for i in range(0, view_num):
            data_all_view[j].extend(data_each_view[i][j])
    return data_all_view


def data_perturbation(data_each_view, feature_num_each_view, label):
    view_num = len(data_each_view)
    data_num = len(data_each_view[0])
    outlier_num_each_type = int(data_num*gl.outlier_percent)
    attribute_outlier = []

    new_label = []
    for i in range(0, data_num):
        new_label.append(label[i])
    for i in range(0, view_num):
        attribute_outlier.append([])
        for j in range(0, outlier_num_each_type):
            attribute_outlier[i].append([])
            for k in range(0, feature_num_each_view[i]):
                attribute_outlier[i][j].append(random.uniform(0, 1))
    swap_view_num = int(view_num/2)
    swap_cnt = 0
    while swap_cnt < int(outlier_num_each_type/2):
        random_index1 = random.randint(0, data_num-1)
        random_index2 = random.randint(0, data_num-1)
        if new_label[random_index1] == new_label[random_index2] or new_label[random_index1] == -1 or new_label[random_index2] == -1:
            continue
        for i in range(0, swap_view_num):
            data_each_view[i][random_index1], data_each_view[i][random_index2] = \
                data_each_view[i][random_index2], data_each_view[i][random_index1]
            new_label[random_index1] = -1
            new_label[random_index2] = -1
            swap_cnt += 1
    swap_cnt = 0
    while swap_cnt < int(outlier_num_each_type/2):
        random_index1 = random.randint(0, data_num - 1)
        random_index2 = random.randint(0, data_num - 1)
        if new_label[random_index1] == new_label[random_index2] or new_label[random_index1] == -1 or new_label[random_index2] == -1:
            continue
        for i in range(0, swap_view_num):
            data_each_view[i][random_index1], data_each_view[i][random_index2] = \
                data_each_view[i][random_index2], data_each_view[i][random_index1]
            new_label[random_index1] = -1
            new_label[random_index2] = -1
            swap_cnt += 1
        for i in range(swap_view_num, view_num):
            for j in range(0, len(data_each_view[i][0])):
                data_each_view[i][random_index1][j] = random.uniform(0, 1)
                data_each_view[i][random_index2][j] = random.uniform(0, 1)
    for i in range(0, data_num):
        if new_label[i] == -1:
            new_label[i] = 1
        else:
            new_label[i] = 0
    for j in range(0, outlier_num_each_type):
        new_label.append(1)
        label.append(7)
        for i in range(0, view_num):
            data_each_view[i].append(attribute_outlier[i][j])
    return data_each_view, label, new_label


def data_normalization(data_each_view, feature_num_each_view):
    view_num = len(data_each_view)
    data_num = len(data_each_view[0])
    for i in range(0, view_num):
        for k in range(0, feature_num_each_view[i]):
            max_temp = -1
            min_temp = 1e10
            for j in range(0, data_num):
                if data_each_view[i][j][k] > max_temp:
                    max_temp = data_each_view[i][j][k]
                if data_each_view[i][j][k] < min_temp:
                    min_temp = data_each_view[i][j][k]
            interval_length = max_temp-min_temp
            if interval_length > 0:
                for j in range(0, data_num):
                    data_each_view[i][j][k] = (data_each_view[i][j][k]-min_temp)/interval_length
            else:
                for j in range(0, data_num):
                    data_each_view[i][j][k] = 0.5
    return data_each_view


def load_data(file_name):
    f = open(file_name, 'r')
    title = f.readline()
    data_str = f.readlines()
    f.close()
    data = []
    for line in data_str:
        line = list(map(float, line.strip().split(',')))
        data.append(line)
    return data, len(data), len(data[0])-1


def cut_data(data, data_num, feature_num):
    feature_num_average = int(feature_num/gl.VIEW_NUM)
    data_in_each_view = []
    feature_num_in_each_view = []
    label = []
    data_new = []
    for x in data:
        data_new.append(x[0:feature_num])
        label.append([int(x[feature_num])])
    for i in range(0, gl.VIEW_NUM-1):
        data_in_each_view.append([])
        for j in range(0, data_num):
            data_in_each_view[i].append(data[j][feature_num_average * i:feature_num_average * (i + 1)])
        feature_num_in_each_view.append(feature_num_average)
    data_in_each_view.append([])
    for j in range(0, data_num):
        data_in_each_view[gl.VIEW_NUM-1].append(data[j][feature_num_average*(gl.VIEW_NUM-1):feature_num])
    feature_num_in_each_view.append(feature_num-feature_num_average*(gl.VIEW_NUM-1))
    return data_new, data_in_each_view, label, feature_num_in_each_view


def computer_outlier_factor2(data_embedding, k):
    distances_all_view = []
    indices_all_view = []
    for i in range(0, gl.VIEW_NUM):
        neighbour_model = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(data_embedding[i])
        distances, indices = neighbour_model.kneighbors(data_embedding[i])
        distances_all_view.append(distances)
        indices_all_view.append(indices)
    outlier_factor = []
    data_num = len(data_embedding[0])
    for j in range(0, data_num):
        outlier_factor_temp = 0
        data_embedding_ave = data_embedding[0][j]
        for i in range(1, gl.VIEW_NUM):
            data_embedding_ave += data_embedding[i][j]
        data_embedding_ave /= gl.VIEW_NUM
        for i in range(0, gl.VIEW_NUM):
            outlier_factor_temp += math.sqrt(sum(pow(data_embedding[i][j]-data_embedding_ave, 2)))*gl.VIEW_NUM
            outlier_factor_temp += distances_all_view[i][j][k]*gl.alpha
        outlier_factor.append(outlier_factor_temp)
    return outlier_factor


def evaluate_auc(outlier_factor, label):
    return roc_auc_score(label, outlier_factor)


def write_evaluation_into_file(filename, auc, run_time):
    fp = open(filename, "a", encoding="utf-8")
    fp.write(str(auc) + ", run_time = " + str(run_time) + ", view_num = " + str(gl.VIEW_NUM) + ", REGULARIZATION_RATE = " + str(gl.REGULARIZATION_RATE) +", beta = "+str(gl.beta)+", k_nearest = "+str(gl.k_nearest)+", LAYER2_NODE = "+str(gl.LAYER2_NODE)+", outlier_percent = "+str(gl.outlier_percent*3)+", TRAINING_STEPS = "+str(gl.TRAINING_STEPS)+", LEARNING_RATE_BASE = "+str(gl.LEARNING_RATE_BASE)+", LEARNING_RATE_DECAY = "+str(gl.LEARNING_RATE_DECAY)+", LAYER1_NODE = "+str(gl.LAYER1_NODE)+", LAYER3_NODE = "+str(gl.LAYER3_NODE)+", alpha = "+str(gl.alpha))
    fp.write('\n')
    fp.close()
