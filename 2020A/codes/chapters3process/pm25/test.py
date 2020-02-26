# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np
import csv
decay_rate = 0.96  # 衰减率
global_steps = 1000  # 总的迭代次数
decay_steps = 100  # 衰减次数
global_ = tf.Variable(tf.constant(0))
rnn_unit = 50  # the amount of hidden lstm units
batch_size = 72  # the amount of data trained every time
#input_size = 76
input_size = 6# size of input
output_size = 1  # size of output
lr = 0.99  # learn rate
train_x, train_y = [], []  #
test_x, test_y = [], []
pre_x = []
prediction = []
f = open('data\\pm25_train.csv', encoding='UTF-8')
df = pd.read_csv(f)  # read the csv file
y_true = df['pm2.5'].values
df_test = pd.read_csv("data\\pm25_test.csv")
del df['date']
del df_test['date']
del df['hour']
del df_test['hour']
del df['Is']
del df['Ir']
del df['PRES']
del df['cbwd_NE']
del df_test['Is']
del df_test['Ir']
del df_test['PRES']
del df_test['cbwd_NE']
df_train = preprocessing.normalize(df_train, norm='l1')
df_train = (df_train - np.mean(df_train, axis=0)) / np.std(df_train, axis=0)  # 标准化
df_train = preprocessing.scale(df_train)
df_test = preprocessing.normalize(df_test, norm='l1')
df_test = (df_test - np.mean(df_test, axis=0)) / np.std(df_test, axis=0)  # 标准化
df_test = preprocessing.scale(df_test)
#min_max_scaler = preprocessing.MinMaxScaler()
#df = min_max_scaler.fit_transform(df.iloc[0:, 2:13])
#df = min_max_scaler.fit_transform(df)
#df_test = min_max_scaler.fit_transform(df_test)
pm25data = df[0 :28000 , 0:1]
pm25test = df[28000:, 0:1]
# quadratic_featurizer = preprocessing.PolynomialFeatures(degree=2)
# df = quadratic_featurizer.fit_transform(df[0:,1:])
# df_test = quadratic_featurizer.fit_transform(df_test[0:,0:])

weatherdata = df[0:28000, 1:7]  # weather with 7 items, not including PM2.5, for trai
  # pm2.5 data, for train
weathertest = df[28000:, 1:7]  # weatherdata with 7 items, not including PM2.5, for exam
pm25test = df[28000:, 0:1]  # pm2.5 data, for exam
weatherpre = df_test[0:,0:6]
# train_x is a tensor which [?,batch_size,input_size]
# train_y is a tensor which [?,batch_size,output_size]
i = 0
while i < len(weatherdata):
    x = weatherdata[i:i + batch_size]  # conver weatherdata to a tensor
    y = pm25data[i:i + batch_size]  # the same with the pm25data
    train_x.append(x.tolist())  # push them into train_x ans train_y
    train_y.append(y.tolist())
    i += batch_size
i = 0
while i < len(weathertest):
    xt = weathertest[i:i + batch_size]  # conver weatherdata to a tensor
    yt = pm25test[i:i + batch_size]  # the same with the pm25data
    test_x.append(xt.tolist())  # push them into train_x ans train_y
    test_y.append(yt.tolist())
    i += batch_size
i=0
while i < len(weatherpre):
    xt = weatherpre[i:i + batch_size]  # conver weatherdata to a tensor
    pre_x.append(xt.tolist())  # push them into train_x ans train_y
    i += batch_size
print(len(pre_x))
# placeholder
X = tf.placeholder(tf.float32, [None, batch_size, input_size])  # a placeholder as the input tensor
Y = tf.placeholder(tf.float32, [None, batch_size, output_size])  # the lable
# initialize the weights and biases [7,50] [50,1]
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, output_size]))
}

biases = {
    'in': tf.Variable(tf.random_normal([batch_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([batch_size, output_size]))
}


def lstm(batch):
    w_in = weights['in']
    b_in = biases['in']
    w_out = weights['out']
    b_out = biases['out']
    # convert the tensor(X Accepts value from outside function) to a 2-dimensional tensor, [?*7]
    input_ = tf.reshape(X, [-1, input_size])
    # make matrix multiplication between input_ and w_in, then add b_in
    input_rnn = tf.matmul(input_, w_in) + b_in
    # convert input_rnn to a 3-dimensional tensor as the input of BaicSTMCell
    input_rnn = tf.reshape(input_rnn, [-1, batch, rnn_unit])
    # BasicLSTMCell cell，the amount of rnn_unit
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # initial cell，batch_size is equal with the input parameter BATCH
    init_state = cell.zero_state(batch, dtype=tf.float32)
    # outputs is a tensor of shape [batch_size, max_time, cell_state_size]
    # final_state is a tensor of shape [batch_size, cell_state_size]
    # Create a Cell
    # time_major = True ==> Tensorshape [max_time, batch_size, ...] something goes wrong when FALSE
    output_rnn, final_state = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32,
                                                time_major=True)
    # convert the output tensor to a 2-dimensional tensor, then calculate the output
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    # make matrix multiplication between output and w_out, then add b_out
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_state


def train_lstm():
    print('start train lstm')
    print(len(train_x))
    global batch_size
    pred, _ = lstm(batch_size)
    # calculate the loss, use the sum of variance between PRED and Y
    789
    ##loss need to be improved
    ##
    loss = tf.reduce_sum(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    lr = tf.train.exponential_decay(0.1,global_, decay_steps, decay_rate, staircase=True)
    # use lr as the learn rate, to make the loss minimize
    train_op = tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_)
    # save the model
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            step = 0
            start = 0
            end = start + 1
            while (end < len(train_x) - 1):
                loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start + 1:end + 1]})
                start += 1
                end += 1
                # if step % 1000 == 0:
                #     print('round: ', i, '  step: ', step, '  loss : ', loss_)
                # if step % 1000 == 0:
                #     saver.save(sess, "data\\model.ckpt")
                #     print('save model')
                step += 1
        start= 0
        end = start+72
        pre_y = sess.run(pred,feed_dict={X:[test_x[1]] })
        print(test_x[1])
        print(pre_y)
        mse = tf.reduce_mean(tf.square(pre_y-test_y[1]))
        print("MSE:%.4f"%sess.run(mse))
        i=0
        for i in range(83):
            prediction[72*i:72*(i+1)] = sess.run(pred,feed_dict={X:[pre_x[i]]})
        i=0
        print(prediction)
        pre = np.array(prediction)
        pre = pre * np.std(y_true) + np.mean(y_true)
        print(pre.shape)
        print(pre)
        l= []
        for i in range(0,5976):
             for m in pre[i]:
                 l.append(m)
        result = pd.DataFrame({'pm2.5':l})
        result.to_csv("data\\result1.csv",index=False)

train_lstm()








