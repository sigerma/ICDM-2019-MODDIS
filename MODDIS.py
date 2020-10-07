import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import random
import math
import Methods
import GlobalVariable as gl
import os
import time

np.seterr(divide='ignore', invalid='ignore')


def inference(input_tensor_all, input_tensor_each_view, input_tensor_each_view_couple1, input_tensor_each_view_couple2, w1_common, b1_common, w2_common, b2_common, w1_each_view, b1_each_view, w2_each_view, b2_each_view, w3_each_view, b3_each_view, w4_each_view, b4_each_view):
    layer1_common = tf.nn.sigmoid(tf.matmul(input_tensor_all, w1_common) + b1_common)
    layer2_common = tf.nn.sigmoid(tf.matmul(layer1_common, w2_common) + b2_common)
    layer1_each_view = []
    layer1_each_view_couple1 = []
    layer1_each_view_couple2 = []

    for i in range(0, gl.VIEW_NUM):
        layer1_each_view.append(tf.nn.sigmoid(tf.matmul(input_tensor_each_view[i], w1_each_view[i]) + b1_each_view[i]))
        layer1_each_view_couple1.append(tf.nn.sigmoid(tf.matmul(input_tensor_each_view_couple1[i], w1_each_view[i]) + b1_each_view[i]))
        layer1_each_view_couple2.append(tf.nn.sigmoid(tf.matmul(input_tensor_each_view_couple2[i], w1_each_view[i]) + b1_each_view[i]))

    layer2_each_view = []
    layer2_each_view_couple1 = []
    layer2_each_view_couple2 = []

    for i in range(0, gl.VIEW_NUM):
        layer2_each_view.append(tf.nn.sigmoid(tf.matmul(layer1_each_view[i], w2_each_view[i]) + b2_each_view[i]))
        layer2_each_view_couple1.append(tf.nn.sigmoid(tf.matmul(layer1_each_view_couple1[i], w2_each_view[i]) + b2_each_view[i]))
        layer2_each_view_couple2.append(tf.nn.sigmoid(tf.matmul(layer1_each_view_couple2[i], w2_each_view[i]) + b2_each_view[i]))

    offset_each_view = []

    middle_layer_each_view = []
    middle_layer_each_view_couple1 = []
    middle_layer_each_view_couple2 = []

    for i in range(0, gl.VIEW_NUM):
        offset_each_view.append(layer2_each_view[i])
        middle_layer_each_view.append(layer2_common + layer2_each_view[i])
        middle_layer_each_view_couple1.append(layer2_common + layer2_each_view_couple1[i])
        middle_layer_each_view_couple2.append(layer2_common + layer2_each_view_couple2[i])

    return layer2_common, middle_layer_each_view, offset_each_view, middle_layer_each_view_couple1, middle_layer_each_view_couple2


def train(data, data_each_view, feature_num_each_view, label):
    data_num = len(data)
    feature_num_all_view = len(data[0])

    x_common = tf.placeholder(tf.float32, shape=[None, feature_num_all_view], name='x-input-common')

    x_each_view = []
    for i in range(0, gl.VIEW_NUM):
        x_each_view.append(
            tf.placeholder(tf.float32, shape=[None, feature_num_each_view[i]], name='x-input-' + str(i)))

    x_each_view_couple1 = []
    for i in range(0, gl.VIEW_NUM):
        x_each_view_couple1.append(
            tf.placeholder(tf.float32, shape=[None, feature_num_each_view[i]], name='x-input2-' + str(i)))
    x_each_view_couple2 = []
    for i in range(0, gl.VIEW_NUM):
        x_each_view_couple2.append(
            tf.placeholder(tf.float32, shape=[None, feature_num_each_view[i]], name='x-input3-' + str(i)))

    label_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

    w1_common = tf.Variable(tf.truncated_normal(shape=[feature_num_all_view, gl.LAYER1_NODE], stddev=0.1))
    b1_common = tf.Variable(tf.constant(0.1, shape=[gl.LAYER1_NODE]))

    w2_common = tf.Variable(tf.truncated_normal(shape=[gl.LAYER1_NODE, gl.LAYER2_NODE], stddev=0.1))
    b2_common = tf.Variable(tf.constant(0.1, shape=[gl.LAYER2_NODE]))

    w1_each_view = []
    b1_each_view = []
    for i in range(0, gl.VIEW_NUM):
        w1_each_view.append(
            tf.Variable(tf.truncated_normal(shape=[feature_num_each_view[i], gl.LAYER1_NODE], stddev=0.1)))
        b1_each_view.append(tf.Variable(tf.constant(0.1, shape=[gl.LAYER1_NODE])))

    w2_each_view = []
    b2_each_view = []
    for i in range(0, gl.VIEW_NUM):
        w2_each_view.append(tf.Variable(tf.truncated_normal(shape=[gl.LAYER1_NODE, gl.LAYER2_NODE], stddev=0.1)))
        b2_each_view.append(tf.Variable(tf.constant(0.1, shape=[gl.LAYER2_NODE])))

    middle_y_common, middle_y_each_view, offset_each_view, middle_y_each_view_couple1, middle_y_each_view_couple2 = inference(
        x_common, x_each_view, x_each_view_couple1, x_each_view_couple2, w1_common, b1_common, w2_common,
        b2_common, w1_each_view, b1_each_view, w2_each_view, b2_each_view)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(gl.MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # loss
    loss1 = 0
    for i in range(0, gl.VIEW_NUM):
        loss1 += tf.reduce_mean(tf.square(offset_each_view[i]))
    loss2 = 0
    for i in range(0, gl.VIEW_NUM):
        loss2 += tf.abs(tf.reduce_mean(
            tf.square(middle_y_each_view_couple2[i] - middle_y_each_view_couple1[i])) - tf.reduce_mean(
            tf.square(x_each_view_couple2[i] - x_each_view_couple1[i])))
    loss = loss1 * gl.beta + loss2

    learning_rate = tf.train.exponential_decay(gl.LEARNING_RATE_BASE, global_step, 900, gl.LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_averages_op)

    embedding_data = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        label_input = [[xxx] for xxx in label]
        validate_feed = {x_common: data, label_: label_input}
        for i in range(0, gl.VIEW_NUM):
            validate_feed[x_each_view[i]] = data_each_view[i]
        input_couple1_each_view = []
        input_couple2_each_view = []
        for i in range(0, gl.VIEW_NUM):
            input_couple1_each_view.append([])
            input_couple2_each_view.append([])
            for j in range(0, 1):
                for k in range(0, 1):
                    input_couple1_each_view[i].append(data_each_view[i][j])
                    input_couple2_each_view[i].append(data_each_view[i][k])
        for i in range(0, gl.VIEW_NUM):
            validate_feed[x_each_view_couple1[i]] = input_couple1_each_view[i]
            validate_feed[x_each_view_couple2[i]] = input_couple2_each_view[i]

        for i in range(gl.TRAINING_STEPS):
            x_common_batch = []
            x_each_view_batch = []
            label_batch = []
            x_each_view_couple1_batch = []
            x_each_view_couple2_batch = []
            for k in range(0, gl.VIEW_NUM):
                x_each_view_couple1_batch.append([])
                x_each_view_couple2_batch.append([])
                x_each_view_batch.append([])
            for _ in range(0, gl.batch_size):
                random_index1 = random.randint(0, data_num - 1)
                random_index2 = random.randint(0, data_num - 1)
                x_common_batch.append(data[random_index1])
                x_common_batch.append(data[random_index2])
                label_batch.append([label[random_index1]])
                label_batch.append([label[random_index2]])
                for k in range(0, gl.VIEW_NUM):
                    x_each_view_batch[k].append(data_each_view[k][random_index1])
                    x_each_view_batch[k].append(data_each_view[k][random_index2])
                    x_each_view_couple1_batch[k].append(data_each_view[k][random_index1])
                    x_each_view_couple1_batch[k].append(data_each_view[k][random_index1])
                    x_each_view_couple2_batch[k].append(data_each_view[k][random_index2])
                    x_each_view_couple2_batch[k].append(data_each_view[k][random_index2])
            test_feed = {x_common: x_common_batch,
                         label_: label_batch}
            for k in range(0, gl.VIEW_NUM):
                test_feed[x_each_view[k]] = x_each_view_batch[k]
                test_feed[x_each_view_couple1[k]] = x_each_view_couple1_batch[k]
                test_feed[x_each_view_couple2[k]] = x_each_view_couple2_batch[k]
            sess.run(train_op, feed_dict=test_feed)

        middle_data = sess.run(middle_y_each_view, feed_dict=validate_feed)
        for i in range(0, gl.VIEW_NUM):
            embedding_data.append([])
            for j in range(0, data_num):
                embedding_data[i].append(middle_data[i][j])
    return embedding_data


def run(data_all_view, data_each_view, feature_num_each_view, label):
    embedding_res = []
    cnt = 0
    while True:
        embedding_res = train(data_all_view, data_each_view, feature_num_each_view, label)
        cnt += 1
        if embedding_res is not None:
            break
    return embedding_res


auc_list = []
for _ in range(0, 50):
    start_time = time.clock()
    data_each_view, feature_num_each_view, new_label = Methods.load_data_from_ready_file(gl.DATA_NAME, _)
    data_each_view = Methods.data_normalization(data_each_view, feature_num_each_view)
    data_all_view = Methods.data_join(data_each_view)
    data_embedding = run(data_all_view, data_each_view, feature_num_each_view, new_label)
    outlier_factor = Methods.computer_outlier_factor2(data_embedding, gl.k_nearest)
    auc = Methods.evaluate_auc(outlier_factor, new_label)
    end_time = time.clock()
    print("auc: ", auc, "run time: ", end_time - start_time)
    auc_list.append(auc)
    Methods.write_evaluation_into_file(gl.EVALUATION_FILE_NAME, auc, end_time-start_time)
print("auc average: ", np.average(auc_list), ", auc std: ", np.std(auc_list))
