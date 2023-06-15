  # -*- coding: UTF-8 -*-
import multiprocessing
import os
import sys

import scipy
import time
from glob import glob
import shutil
import threading
import numpy as np
import tensorflow as tf
import random
import copy
#import matplotlib.pyplot as plt
#from toytest_EasyMKL import use_EasyMKL
#from imbalance_SVM  import use_imbalance_SVM
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve,auc
from demo import parse


os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
os.environ['CUDA_VISIBLE_DEVICES']='1'
class CNN_classifier(object):
    def __init__(self,run_config,conf):
        self.sess =''
        self.run_config=run_config
        self.train_flag=conf.train_flag  #是否是训练阶段
        # self.data_dir = ".\Data\mnist"  # 训练图像所在的文件夹
        # self.data_dir = ".\Data\ADNI\\test\\aug"  # 训练图像所在的文件夹
        # self.data_dir = ".\Data\\Number\\aug"  # 训练图像所在的文件夹
        self.data_dir = conf.data_dir         # 训练图像所在的文件夹

        self.train_data_dir = conf.train_data_dir  # 训练图像所在的文件夹
        self.test_data_dir = conf.test_data_dir         # 测试图像所在的文件夹
        self.SVM_tr_data_dir=conf.SVM_tr_data_dir       #训练图像文件（不带augment）
        self.SVM_te_data_dir=conf.SVM_te_data_dir        #测试图像文件，这里与test_data_dir设置的路径是相同的
        self.thread_name=conf.thread_name               #多进程训练时，传入进程名称，在打印进程时区分进程
        self.use_saved_model=conf.use_saved_model       #训练时是否使用已保存的模型
        self.model_idx=conf.model_idx                   #用来在多进程时，设置每个模型的variable_scope
        self.pretrain_data_dir=conf.pretrain_data_dir

        #是否每个切片只用最好的一个模型做测试
        self.use_best_one_model = conf.use_best_one_model
        #当use_best_one_model为FALSE时，要传入model_name用于test阶段restore模型
        self.model_name=''
        #传入验证集路径，这里验证集用509数据集之外的数据充当
        self.valid_data_dir=conf.valid_data_dir
        #传入训练集（不带augment）的样本数，用于创建placeholder
        self.train_data_num=conf.train_data_num
        # 传入测试集的样本数，用于创建placeholder
        self.test_data_num = conf.test_data_num



        #self.train_dir="."

        self.checkpoint_dir=conf.checkpoint_dir
        self.pretrain_checkpoint_dir = conf.pretrain_checkpoint_dir

        # self.checkpoint_dir = ".\checkpoint\cnn_layer_6_dropout0.5_smooth_BN_L1_acc0.819"
        self.train_size=np.inf          #拿多少图像做训练集


        self.max_epoch=conf.max_epoch
        self.batch_size=conf.batchsize
        self.learning_rate=conf.learning_rate
        self.momentum=0.9

        self.input_height = 145     #218
        self.input_width = 145      #218
        self.c_dim=1                #color_dim 灰度图像设为1
        self.y_dim=2                #图像有多少个类别
        self.is_grayscale = True    #是否是灰度图像

        #第一二位代表卷积核尺寸大小，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征。
        self.fliter_size1 = [3, 3, 1, 32]
        self.fliter_size2 = [3, 3, 32, 64]
        self.fliter_size3 = [3, 3, 64, 128]
        self.fliter_size4 = [1, 1, 128, 256]
        self.fliter_size5 = [1, 1, 256, 512]
        self.fliter_size6 = [1, 1, 512, 1024]

        # self.fliter_size1 = [3, 3, 1, 32]
        # self.fliter_size2 = [3, 3, 32, 64]
        # self.fliter_size3 = [3, 3, 64, 128]
        # self.fliter_size4 = [3, 3, 128, 256]
        # self.fliter_size5 = [3, 3, 256, 512]
        # self.fliter_size6 = [3, 3, 512, 1024]


        # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长
        self.stride1 = [1, 3, 3, 1]
        self.stride2 = [1, 3, 3, 1]
        self.stride3 = [1, 3, 3, 1]
        self.stride4 = [1, 1, 1, 1]
        self.stride5 = [1, 1, 1, 1]
        self.stride6 = [1, 1, 1, 1]
        # self.stride1 = [1, 1, 1, 1]
        # self.stride2 = [1, 1, 1, 1]
        # self.stride3 = [1, 1, 1, 1]
        # self.stride4 = [1, 1, 1, 1]
        # self.stride5 = [1, 1, 1, 1]
        # self.stride6 = [1, 1, 1, 1]

        #定义池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]
        self.pool_size1 = [1, 3, 3, 1]
        self.pool_size2 = [1, 3, 3, 1]
        self.pool_size3 = [1, 3, 3, 1]
        self.pool_size4 = [1, 3, 3, 1]
        self.pool_size5 = [1, 3, 3, 1]
        self.pool_size6 = [1, 3, 3, 1]
        #定义池化层strides,和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
        self.pool_stride2 = [1, 1, 1, 1]
        self.pool_stride3 = [1, 1, 1, 1]
        self.pool_stride4 = [1, 1, 1, 1]
        self.pool_stride5 = [1, 1, 1, 1]
        self.pool_stride6 = [1, 3, 3, 1]

        # self.pool_stride1 = [1, 3, 3, 1]
        # self.pool_stride2 = [1, 3, 3, 1]
        # self.pool_stride3 = [1, 3, 3, 1]
        # self.pool_stride4 = [1, 1, 1, 1]
        # self.pool_stride5 = [1, 1, 1, 1]
        # self.pool_stride6 = [1, 3, 3, 1]

        #定义全连接层结点数
        self.FC1_nodes=100
        self.FC2_nodes=self.y_dim

        # # 开始创建模型
        # self.build_model()



    #创建一个卷积层
    def conv2d(self,input_,fliter_size,stride, name="conv2d",stddev=0.01):
        output_dim=fliter_size[3]
        with tf.variable_scope(name):
            w = tf.get_variable('w',fliter_size,initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, stride, padding='SAME')
            biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.001))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            return conv
    #创建一个池化层
    def max_pool(self,x,pool_size,pool_stride):
        return tf.nn.max_pool(x, pool_size, pool_stride, padding='SAME')

    #创建全连接层
    def linear(self,input_, output_size, name="Linear", stddev=0.02, bias_start=0.0, with_w=False):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(name):
            matrix = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("b", [output_size],
                                   initializer=tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias

    def relu(self,x):
        return tf.nn.leaky_relu(x)
        # return tf.nn.relu(x)

    def CNN2d_BN(self, inputs,keep_prob,reuse=False):
        # 创建一个8层的卷积神经网络
        # with tf.variable_scope("CNN2d"+str(self.model_idx),reuse=tf.AUTO_REUSE) as scope:
        with tf.variable_scope("CNN2d", reuse=tf.AUTO_REUSE) as scope:
            c1 = self.relu(self.conv2d(inputs,self.fliter_size1,self.stride1, name='conv1'))  # 卷积层1，卷积层1后面没有池化层
            bn1 = tf.layers.batch_normalization(c1, training=True, name='bn1')  # batch_normalization层

            c2 = self.relu(self.conv2d(bn1, self.fliter_size2, self.stride2, name='conv2'))  # 卷积层2
            bn2 = tf.layers.batch_normalization(c2, training=True, name='bn2')              #batch_normalization层
            p2=self.max_pool(bn2,self.pool_size2,self.pool_stride2)                          #池化层2

            c3 = self.relu(self.conv2d(p2, self.fliter_size3, self.stride3, name='conv3'))  # 卷积层3
            bn3 = tf.layers.batch_normalization(c3, training=True, name='bn3')  # batch_normalization层
            p3=self.max_pool(bn3,self.pool_size3,self.pool_stride3)                          #池化层3

            c4 = self.relu(self.conv2d(p3, self.fliter_size4, self.stride4, name='conv4'))  # 卷积层4
            bn4 = tf.layers.batch_normalization(c4, training=True, name='bn4')  # batch_normalization层
            p4=self.max_pool(bn4,self.pool_size4,self.pool_stride4)                          #池化层4

            c5 = self.relu(self.conv2d(p4, self.fliter_size5, self.stride5, name='conv5'))  # 卷积层5
            bn5 = tf.layers.batch_normalization(c5, training=True, name='bn5')  # batch_normalization层
            p5=self.max_pool(bn5,self.pool_size5,self.pool_stride5)                          #池化层5

            c6 = self.relu(self.conv2d(p5, self.fliter_size6, self.stride6, name='conv6'))  # 卷积层6
            bn6 = tf.layers.batch_normalization(c6, training=True, name='bn6')  # batch_normalization层
            p6=self.max_pool(bn6,self.pool_size6,self.pool_stride6)                          #池化层6

            p6_size=p6.get_shape().as_list()
            p6_flat_len=p6_size[1]*p6_size[2]*p6_size[3]
            pool6_flat = tf.reshape(p6, [-1, p6_flat_len])  # 展开，第一个参数为样本数量，-1代表未知

            # p4_size=p4.get_shape().as_list()
            # p4_flat_len=p4_size[1]*p4_size[2]*p4_size[3]
            # pool4_flat = tf.reshape(p4, [-1, p4_flat_len])  # 展开，第一个参数为样本数量，-1代表未知

            # p3_size=p3.get_shape().as_list()
            # p3_flat_len=p3_size[1]*p3_size[2]*p3_size[3]
            # pool3_flat = tf.reshape(p3, [-1, p3_flat_len])  # 展开，第一个参数为样本数量，-1代表未知

            fc1 = self.relu(self.linear(pool6_flat,self.FC1_nodes, 'fc1'))  #创建第1个全连接层

            # Dropout层(原论文没有)
            fc1_drop = tf.nn.dropout(fc1, keep_prob)

            fc2 = self.relu(self.linear(fc1_drop,self.FC2_nodes, 'fc2'))            #创建第2个全连接层

            #收集摘要
            self.sum_c1 = self.summary_image("c1",c1)
            self.sum_c2 = self.summary_image("c2", c2)
            self.sum_c3 = self.summary_image("c3", c3)
            self.sum_c4 = self.summary_image("c4", c4)
            self.sum_c5 = self.summary_image("c5", c5)
            self.sum_c6 = self.summary_image("c6", c6)

            return tf.nn.softmax(fc2), fc2, fc1  # 返回经过softmax后和没经过softmax的值,#返回fc1这里是为了获得SVM的输入特征

    def CNN2d(self, inputs, keep_prob, reuse=False):
        # 创建一个8层的卷积神经网络
        with tf.variable_scope("CNN2d"+str(self.model_idx),reuse=tf.AUTO_REUSE) as scope:
        # with tf.variable_scope("CNN2d", reuse=tf.AUTO_REUSE) as scope:
            c1 = self.relu(self.conv2d(inputs, self.fliter_size1, self.stride1, name='conv1'))  # 卷积层1，卷积层1后面没有池化层

            c2 = self.relu(self.conv2d(c1, self.fliter_size2, self.stride2, name='conv2'))  # 卷积层2
            p2 = self.max_pool(c2, self.pool_size2, self.pool_stride2)  # 池化层2

            c3 = self.relu(self.conv2d(p2, self.fliter_size3, self.stride3, name='conv3'))  # 卷积层3
            p3 = self.max_pool(c3, self.pool_size3, self.pool_stride3)  # 池化层3

            c4 = self.relu(self.conv2d(p3, self.fliter_size4, self.stride4, name='conv4'))  # 卷积层4
            p4 = self.max_pool(c4, self.pool_size4, self.pool_stride4)  # 池化层4

            c5 = self.relu(self.conv2d(p4, self.fliter_size5, self.stride5, name='conv5'))  # 卷积层5
            p5 = self.max_pool(c5, self.pool_size5, self.pool_stride5)  # 池化层5

            c6 = self.relu(self.conv2d(p5, self.fliter_size6, self.stride6, name='conv6'))  # 卷积层6
            p6 = self.max_pool(c6, self.pool_size6, self.pool_stride6)  # 池化层6

            p6_size = p6.get_shape().as_list()
            p6_flat_len = p6_size[1] * p6_size[2] * p6_size[3]
            pool6_flat = tf.reshape(p6, [-1, p6_flat_len])  # 展开，第一个参数为样本数量，-1代表未知

            fc1 = self.relu(self.linear(pool6_flat, self.FC1_nodes, 'fc1'))  # 创建第1个全连接层

            # Dropout层(原论文没有)
            fc1_drop = tf.nn.dropout(fc1, keep_prob)

            fc2 = self.relu(self.linear(fc1, self.FC2_nodes, 'fc2'))  # 创建第2个全连接层

            # 收集摘要
            self.sum_c1 = self.summary_image("c1", c1)
            self.sum_c2 = self.summary_image("c2", c2)
            self.sum_c3 = self.summary_image("c3", c3)
            self.sum_c4 = self.summary_image("c4", c4)
            self.sum_c5 = self.summary_image("c5", c5)
            self.sum_c6 = self.summary_image("c6", c6)

            return tf.nn.softmax(fc2), fc2, fc1  # 返回经过softmax后和没经过softmax的值,#返回fc1这里是为了获得SVM的输入特征


    def CAE(self, inputs,reuse=False):
        # 创建一个8层的卷积神经网络
        # with tf.variable_scope("CNN2d"+str(self.model_idx),reuse=tf.AUTO_REUSE) as scope:
        with tf.variable_scope("CNN2d", reuse=tf.AUTO_REUSE) as scope:
            #前6层是Encoder
            #input(200,145,145,1)
            # c1(200,49,49,32)
            stride=[1,1,1,1]
            pool_size = [1, 3, 3, 1]
            pool_stride = [1, 3, 3, 1]

            c1 = self.relu(self.conv2d(inputs, self.fliter_size1, self.stride1, name='conv1'))  # 卷积层1，卷积层1后面没有池化层

            c2 = self.relu(self.conv2d(c1, self.fliter_size2, self.stride2, name='conv2'))  # 卷积层2
            p2 = self.max_pool(c2, self.pool_size2, self.pool_stride2)  # 池化层2

            c3 = self.relu(self.conv2d(p2, self.fliter_size3, self.stride3, name='conv3'))  # 卷积层3
            p3 = self.max_pool(c3, self.pool_size3, self.pool_stride3)  # 池化层3

            c4 = self.relu(self.conv2d(p3, self.fliter_size4, self.stride4, name='conv4'))  # 卷积层4
            p4 = self.max_pool(c4, self.pool_size4, self.pool_stride4)  # 池化层4

            c5 = self.relu(self.conv2d(p4, self.fliter_size5, self.stride5, name='conv5'))  # 卷积层5
            p5 = self.max_pool(c5, self.pool_size5, self.pool_stride5)  # 池化层5

            c6 = self.relu(self.conv2d(p5, self.fliter_size6, self.stride6, name='conv6'))  # 卷积层6
            encoder = self.max_pool(c6, self.pool_size6, self.pool_stride6)  # 池化层6


            #后6层是Decoder
            upsample1 = tf.image.resize_images(encoder, size=(6, 6), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            de_conv1 = self.relu(self.conv2d(upsample1, fliter_size=[3,3,1024,512], stride=[1, 1, 1, 1], name='de_conv1'))
            #
            # upsample2 = tf.image.resize_images(encoder, size=(6, 6), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            de_conv2 = self.relu(self.conv2d(de_conv1, fliter_size=[3, 3, 512, 256], stride=[1, 1, 1, 1], name='de_conv2'))

            de_conv3 = self.relu(self.conv2d(de_conv2, fliter_size=[3, 3, 256, 128], stride=[1, 1, 1, 1], name='de_conv3'))

            upsample4 = tf.image.resize_images(de_conv3, size=(17, 17), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            de_conv4 = self.relu(self.conv2d(upsample4, fliter_size=[3, 3, 128, 64], stride=[1, 1, 1, 1], name='de_conv4'))

            upsample5 = tf.image.resize_images(de_conv4, size=(49, 49), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            de_conv5= self.relu(self.conv2d(upsample5, fliter_size=[3, 3, 64, 32], stride=[1, 1, 1, 1], name='de_conv5'))

            upsample6 = tf.image.resize_images(de_conv5, size=(145, 145), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            de_conv6= self.relu(self.conv2d(upsample6, fliter_size=[3, 3, 32, 1], stride=[1, 1, 1, 1], name='de_conv6'))

            # de_conv6= self.relu(self.conv2d(de_conv5, fliter_size=[3, 3, 32, 1], stride=[1, 1, 1, 1], name='de_conv6'))

            decoded_logits = de_conv6
            decoded = tf.nn.sigmoid(decoded_logits)

            return decoded,decoded_logits,encoder # 返回经过softmax后和没经过softmax的值,#返回fc1这里是为了获得SVM的输入特征




    def CNN2d_test(self, inputs, keep_prob, reuse=False):
        # 创建一个8层的卷积神经网络
        with tf.variable_scope("CNN2d") as scope:
            ##########以下为不用BN的部分
            c1 = self.relu(self.conv2d(inputs, self.fliter_size1, self.stride1, name='conv1'))  # 卷积层1，卷积层1后面没有池化层
            bn1 = tf.layers.batch_normalization(c1, training=False, name='bn1')  # batch_normalization层

            c2 = self.relu(self.conv2d(c1, self.fliter_size2, self.stride2, name='conv2'))  # 卷积层2
            bn2 = tf.layers.batch_normalization(c2, training=False, name='bn2')  # batch_normalization层
            p2 = self.max_pool(c2, self.pool_size2, self.pool_stride2)  # 池化层2

            c3 = self.relu(self.conv2d(p2, self.fliter_size3, self.stride3, name='conv3'))  # 卷积层3
            bn3 = tf.layers.batch_normalization(c3, training=False, name='bn3')  # batch_normalization层
            p3 = self.max_pool(c3, self.pool_size3, self.pool_stride3)  # 池化层3

            c4 = self.relu(self.conv2d(p3, self.fliter_size4, self.stride4, name='conv4'))  # 卷积层4
            bn4 = tf.layers.batch_normalization(c4, training=False, name='bn4')  # batch_normalization层
            p4 = self.max_pool(c4, self.pool_size4, self.pool_stride4)  # 池化层4

            c5 = self.relu(self.conv2d(p4, self.fliter_size5, self.stride5, name='conv5'))  # 卷积层5
            bn5 = tf.layers.batch_normalization(c5, training=False, name='bn5')  # batch_normalization层
            p5 = self.max_pool(c5, self.pool_size5, self.pool_stride5)  # 池化层5

            c6 = self.relu(self.conv2d(p5, self.fliter_size6, self.stride6, name='conv6'))  # 卷积层6
            bn6 = tf.layers.batch_normalization(c6, training=False, name='bn6')  # batch_normalization层
            p6 = self.max_pool(c6, self.pool_size6, self.pool_stride6)  # 池化层6

            p6_size = p6.get_shape().as_list()
            p6_flat_len = p6_size[1] * p6_size[2] * p6_size[3]
            pool6_flat = tf.reshape(p6, [-1, p6_flat_len])  # 展开，第一个参数为样本数量，-1代表未知


            fc1 = self.relu(self.linear(pool6_flat, self.FC1_nodes, 'fc1'))  # 创建第1个全连接层

            # Dropout层(原论文没有)
            fc1_drop = tf.nn.dropout(fc1, keep_prob)

            fc2 = self.relu(self.linear(fc1_drop, self.FC2_nodes, 'fc2'))  # 创建第2个全连接层

            # 收集摘要
            self.sum_c1 = self.summary_image("c1", c1)
            self.sum_c2 = self.summary_image("c2", c2)
            self.sum_c3 = self.summary_image("c3", c3)
            self.sum_c4 = self.summary_image("c4", c4)
            self.sum_c5 = self.summary_image("c5", c5)
            self.sum_c6 = self.summary_image("c6", c6)

            self.fc1=fc1

            return tf.nn.softmax(fc2), fc2, fc1  # 返回经过softmax后和没经过softmax的值,#返回fc1这里是为了获得SVM的输入特征

    def build_model(self):
        self.g1 = tf.Graph()
        with self.g1.as_default():
            image_dims = [self.input_height, self.input_height, self.c_dim]

            #定义输入及标签
            self.inputs= tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='inputs')
            self.labels = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='labels')
            #定义drop的保持概率
            self.keep_prob = tf.placeholder(tf.float32)
            #获得网络模型
            self.CNN, self.CNN_logits,self.fc1 = self.CNN2d(self.inputs, self.keep_prob)


            #计算损失
            self.loss=tf.nn.softmax_cross_entropy_with_logits(logits=self.CNN_logits, labels=self.labels)



            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


            #计算准确率
            self.correct_prediction = tf.equal(tf.argmax(self.CNN, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
            #收集摘要
            self.cnn_sum = tf.summary.histogram("cnn", self.CNN)
            mean_loss = tf.reduce_mean(self.loss)
            self.loss_sum = tf.summary.scalar("mean_loss", mean_loss)
            self.accu_sum = tf.summary.scalar("accu", self.accuracy)


            #获取所有需要优化的变量
            self.t_vars = tf.trainable_variables()  # 获得所有需要训练的变量，tf创建variables时，默认是trainable=true的，如果有些变量不需要训练，则创建变量时设trainable=false
            self.fc_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN2d/fc*')
            self.cnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN2d/conv*')
            #保存batchnorm参数
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            self.t_vars += bn_moving_vars
            self.fc_vars += bn_moving_vars


            self.saver = tf.train.Saver(self.t_vars,max_to_keep=1)
            #创建一个只包含卷积层的saver，用来从预训练模型中加载初始化参数
            # self.saver_cnn=tf.train.Saver(self.cnn_vars,max_to_keep=2)


    def build_model_test(self):
        with self.g1.as_default():
            image_dims = [self.input_height, self.input_height, self.c_dim]

            # 定义输入及标签
            self.inputs_test = tf.placeholder(tf.float32, [self.test_data_num] + image_dims, name='inputs')
            self.labels_test = tf.placeholder(tf.float32, [self.test_data_num, self.y_dim], name='labels')

            # 获得网络模型
            CNN_test, CNN_logits_test, fc1 = self.CNN2d(self.inputs_test, self.keep_prob)

            # 计算准确率
            correct_prediction = tf.equal(tf.argmax(CNN_test, 1), tf.argmax(self.labels_test, 1))
            accuracy_test = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            return CNN_test,accuracy_test

    def build_model_train(self):
        with self.g1.as_default():
            image_dims = [self.input_height, self.input_height, self.c_dim]

            # 定义输入及标签
            self.inputs_train = tf.placeholder(tf.float32, [self.train_data_num] + image_dims, name='inputs')
            self.labels_train = tf.placeholder(tf.float32, [self.train_data_num, self.y_dim], name='labels')

            # 获得网络模型
            CNN_train, CNN_logits_train, fc1 = self.CNN2d(self.inputs_train, self.keep_prob)

            # 计算准确率
            correct_prediction = tf.equal(tf.argmax(CNN_train, 1), tf.argmax(self.labels_train, 1))
            accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            return CNN_train,accuracy_train

    def train_CAE(self):
        self.g2 = tf.Graph()
        with self.g2.as_default():
            # 是否接着上次保存的模型进行训练
            image_dims = [self.input_height, self.input_height, self.c_dim]

            # 定义输入及标签
            self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='inputs')
            # self.labels = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='labels')
            # 定义drop的保持概率
            self.keep_prob = tf.placeholder(tf.float32)
            # 获得网络模型
            self.decoded, self.decoded_logits,self.encoder = self.CAE(self.inputs)
            #计算自编码器损失
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputs, logits=self.decoded_logits)
            # Get cost and define the optimizer
            cost = tf.reduce_mean(loss)

            # learning_rate
            lr = self.learning_rate
            lrs = []
            global_ = tf.Variable(tf.constant(0))
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=global_,
                                                       decay_steps=self.max_epoch / 5, decay_rate=0.7, staircase=True)
            CAE_optim = tf.train.AdamOptimizer(learning_rate).minimize(cost)

            # CAE_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

            # 获取所有需要优化的变量
            CAE_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN2d/conv*')
            self.t_vars = tf.trainable_variables()  # 获得所有需要训练的变量，tf创建variables时，默认是trainable=true的，如果有些变量不需要训练，则创建变量时设trainable=false
            self.saver = tf.train.Saver(CAE_vars, max_to_keep=2)

            counter=0
            start_time = time.time()

            with tf.Session(config=self.run_config, graph=self.g2) as sess:
                try:
                    sess.run(tf.global_variables_initializer())
                except:
                    sess.run(tf.initialize_all_variables())

                load_model=1
                if load_model==1:
                    # 重载训练好的参数
                    ckpt = tf.train.get_checkpoint_state(self.pretrain_checkpoint_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        # 如果检查点存在，则加载最新的检查点
                        model_name = os.path.basename(ckpt.model_checkpoint_path)  # 确定最新的参数文件
                        model_path = os.path.join(self.pretrain_checkpoint_dir, model_name)
                        # model_file = tf.train.latest_checkpoint(self.checkpoint_dir)
                        self.saver.restore(sess, model_path)
                        print('successfully load model from: ' + model_path)
                    else:
                        print('No checkpoint file found')
                        return


                for epoch in range(self.max_epoch):
                    data = np.loadtxt(self.pretrain_data_dir, dtype='str')
                    random.shuffle(data)
                    batch_idxs = min(len(data), self.train_size) // self.batch_size

                    print('current_learning rate is:' + str(lr))
                    lr = sess.run(learning_rate, feed_dict={global_: epoch})
                    lrs.append(lr)
                    for idx in range(0, batch_idxs):
                        try:
                            batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                            batch = self.get_data(batch_files)
                            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                            batch_labels = [self.get_label(batch_file) for batch_file in batch_files]

                            # 运行优化器
                            CAE_optim.run(feed_dict={self.inputs: batch_images,})

                            # 计算损失（训练集带augment）
                            CAE_cost = cost.eval({self.inputs: batch_images,})

                            print(self.thread_name + " Epoch: [%2d] [%4d/%4d] time: %4.4f, mean loss: %.8f" % (
                            epoch, idx, batch_idxs, time.time() - start_time, CAE_cost))

                            # 自动保存检查点，每隔10步自动保存一个检查点，在最后一步之前自动保存一次
                            if counter % 10 == 0:
                                self.save_model(sess, counter,self.pretrain_checkpoint_dir) # 这里有个参数默认值max_to_keep=5，表示保留最新的5个检查点

                            if counter % 100== 0:
                                #保存重建的图像
                                imgs=self.decoded.eval({self.inputs: batch_images,})
                                img = imgs[0, :, :, 0]
                                full_save_name=os.path.join(self.pretrain_checkpoint_dir,'recon_imgs','img_'+str(counter)+'.jpg')
                                self.save_img(full_save_name,img)
                                #
                                imgs2=self.encoder.eval({self.inputs: batch_images,})
                                img2 = imgs2[0, :, :, 0]
                                full_save_name2=os.path.join(self.pretrain_checkpoint_dir,'feature_imgs','img_'+str(counter)+'.jpg')
                                self.save_img(full_save_name2,img2)

                                imgs3=batch_images
                                img3 = imgs3[0, :, :, 0]
                                full_save_name3=os.path.join(self.pretrain_checkpoint_dir,'origin_imgs','img_'+str(counter)+'.jpg')
                                self.save_img(full_save_name3,img3)

                            counter += 1
                        except KeyboardInterrupt:  # 这里是可以在键盘按ctrl+c终止模型时保存一下当前模型
                            self.save_model(sess, counter, self.pretrain_checkpoint_dir)
                            sys.stdout.flush()
                            continue

    def train(self):
        # 开始创建模型
        self.build_model()

            # data_X, data_y = self.load_mnist();
            # cnn_optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=self.t_vars)  # 创建优化器
            # cnn_optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)  # 创建优化器
            # cnn_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum).minimize(self.loss)  # 调用优化器优化
        with self.g1.as_default():
            with tf.control_dependencies(self.update_ops):
                self.cnn_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss,var_list=self.t_vars)  # 调用优化器优化


        #初始化变量
        with tf.Session(config=self.run_config, graph=self.g1) as sess:
            try:
                sess.run(tf.global_variables_initializer())
            except:
                sess.run(tf.initialize_all_variables())


            # 是否接着上次保存的模型进行训练
            if self.use_saved_model:
                # 重载训练好的参数
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # 如果检查点存在，则加载最新的检查点
                    model_name = os.path.basename(ckpt.model_checkpoint_path)  # 确定最新的参数文件
                    model_path = os.path.join(self.checkpoint_dir, model_name)
                    # model_file = tf.train.latest_checkpoint(self.checkpoint_dir)
                    self.saver.restore(sess, model_path)
                    print('successfully load model from: ' + model_path)
                else:
                    print('No checkpoint file found')
                    return

            #集中所有的summary
            # self.c_sum = tf.summary.merge([self.cnn_sum,self.loss_sum,self.sum_c1,self.sum_c2,self.sum_c3,self.sum_c4,self.sum_c5,self.sum_c6])
            self.c_sum = tf.summary.merge(
                [self.cnn_sum, self.loss_sum,self.accu_sum, self.sum_c1, self.sum_c2, self.sum_c3])
            self.writer = tf.summary.FileWriter("./logs", sess.graph)



            ########这里新创建的test和train模型，是为了设置模型保存条件
            ########这里的test实际上传入的是充当valid集的509数据集之外的数据，由于之前这里是用test集作为模型选择条件的,之后换为valid集之后变量名就沿用了test
            CNN_test,accuracy_test=self.build_model_test()
            CNN_train, accuracy_train = self.build_model_train()
            # data_test = np.loadtxt(self.valid_data_dir, dtype='str')# 获取测试集数据
            data_train = np.loadtxt(self.SVM_tr_data_dir, dtype='str')#获取训练数据（不带augment）

            train_data = self.get_data(data_train)
            train_images = np.array(train_data).astype(np.float32)[:, :, :, None]
            train_labels = [self.get_label(d) for d in data_train]

            # test_data = self.get_data(data_test)
            # test_images = np.array(test_data).astype(np.float32)[:, :, :, None]
            # test_labels = [self.get_label(d) for d in data_test]
            #设置全局计数器
            counter = 0
            #设置valid数据集上
            best_acc_test=0

            start_time = time.time()
            for epoch in range(self.max_epoch):
                #每隔10个epoch学习率减少10倍

                #计算有多少个batch
                # data = glob(os.path.join(self.data_dir+'\\train\\aug\\', "*.jpg"))
                # data = glob(os.path.join(self.train_data_dir, "*.jpg"))
                data = np.loadtxt(self.train_data_dir, dtype='str')
                random.shuffle(data)
                batch_idxs = min(len(data), self.train_size) // self.batch_size

                ##validation DATA


                # 获取验证集数据
                # data_v = glob(os.path.join(self.valid_dir, "*.jpg"))
                # data_v = np.loadtxt(self.SVM_tr_data_dir, dtype='str')
                # random.shuffle(data_v)
                for idx in range(0, batch_idxs):
                    try:


                        #读取jpg数据
                        batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch=self.get_data(batch_files)
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                        batch_labels=[self.get_label(batch_file) for batch_file in batch_files]

                        #运行优化器（设置dropout率）
                        self.cnn_optim.run(feed_dict={self.inputs:batch_images,self.labels:batch_labels,self.keep_prob:0.6})

                        #计算损失（训练集带augment）
                        err_CNN = self.loss.eval({self.inputs: batch_images,self.labels: batch_labels,self.keep_prob:1.0})
                        mean_err=np.mean(err_CNN)
                        print(self.thread_name+" Epoch: [%2d] [%4d/%4d] time: %4.4f, mean loss: %.8f" % (epoch, idx, batch_idxs,time.time() - start_time, mean_err))


                        #打印准确率
                        if counter % 100 == 0:
                            # 执行summary
                            summary_str = sess.run(self.c_sum,
                                                        feed_dict={self.inputs: batch_images, self.labels: batch_labels,self.keep_prob:1.0})
                            self.writer.add_summary(summary_str, global_step=counter)
                            #计算准确率
                            predict_labels = sess.run(self.CNN,feed_dict={self.inputs: batch_images, self.labels: batch_labels,self.keep_prob:1.0})
                            pred=np.argmax(predict_labels,1)
                            real=np.argmax(batch_labels,1)

                            TP=TN=FP=FN=SEN=SPC=0
                            for i in range(self.batch_size):
                                # print("image name:%s,\t real_label:%d,\t predict_label:%d" % (file_names[i], real[i], pred[i]))
                                #计算敏感度，特异度等
                                    # True Positive：本来是正样例，分类成正样例
                                    # True Negative：本来是负样例，分类成负样例
                                    # False Positive ：本来是负样例，分类成正样例，通常叫误报。
                                    # False Negative：本来是正样例，分类成负样例，通常叫漏报。
                                if real[i] == 0 and pred[i] == 0:
                                    TP += 1
                                elif real[i] == 1 and pred[i] == 1:
                                    TN += 1
                                elif real[i] == 1 and pred[i] == 0:
                                    FP += 1
                                elif real[i] == 0 and pred[i] == 1:
                                    FN += 1
                            SEN=TP/(TP+FN)      #敏感性
                            SPC=TN/(TN+FP)      #特异性
                            ACC=(TP+TN)/(TP+TN+FP+FN)  #准确率
                            print(self.thread_name+" Epoch: [%2d] [%4d/%4d] time: %4.4f,train sensitivity: %.8f,specificity: %.8f,accuracy: %.8f,"
                                  % (epoch, idx, batch_idxs, time.time() - start_time,SEN,SPC,ACC))



                            ##计算当前模型在训练集（不带augment）上的准确率#################
                            acc_train = sess.run(accuracy_train,
                                                     feed_dict={self.inputs_train: train_images, self.labels_train: train_labels,
                                                                self.keep_prob: 1.0})
                            print("#############"+self.thread_name+" Epoch: [%2d] [%4d/%4d] time: %4.4f,train set accuracy:    %.8f"
                                  % (epoch, idx, batch_idxs, time.time() - start_time, acc_train))


                        #自动保存检查点，每隔10步自动保存一个检查点，在最后一步之前自动保存一次
                        if counter % 10 == 0 or (counter + 1) == batch_idxs:
                            self.save_model(sess,counter)            #这里有个参数默认值max_to_keep=5，表示保留最新的5个检查点
                        counter += 1
                    except KeyboardInterrupt:               #这里是可以在键盘按ctrl+c终止模型时保存一下当前模型
                        self.save_model(sess,counter)
                        sys.stdout.flush()
                        # continue
                        exit()


    def test(self, sess):
        # 开始创建模型
        self.build_model()
        # print("Start test ...")
        #设置默认graph,初始化变量
        with tf.Session(graph=self.g1) as sess:
            try:
                sess.run(tf.global_variables_initializer())
            except:
                sess.run(tf.initialize_all_variables())
        #这里要选择是否只使用最后一个best模型进行测试
            if self.use_best_one_model:
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                # 如果检查点存在，则加载最新的检查点
                if ckpt and ckpt.model_checkpoint_path:
                    # 确定最新的参数文件
                    model_name=os.path.basename(ckpt.model_checkpoint_path)
                    #这里是为了将checkpoint里的文件名取出，然后和当前文件夹路径拼接，因为checkpoint文件是从它外层的文件夹复制过来的，里面记录的文件路径与当前文件夹不一致
                    model_path=os.path.join(self.checkpoint_dir,model_name)
                    #重载最好的模型
                    self.saver.restore(sess, model_path)
                else:
                    print('No checkpoint file found')
                    return
            else:
                #如果要用保存的多个模型做测试，则要提供要加载的模型名
                self.saver.restore(sess, self.model_name)


            # 根据文件路径获取测试数据
            data= np.loadtxt(self.test_data_dir, dtype='str')
            #对数据排序，这里很重要，因为多个fold里面的文件名顺序要能一一对应，后面才能做Ensembel
            data=sorted(data)

            # 打乱数据
            # random.shuffle(data)
            # 防止测试数据个数不够一个batch，这里是之前做的检测。后来在外面传参数时batch_size就设为了test的样本数，所以所有test数据都会划分到同一个batch里
            self.batch_size=min(self.batch_size,len(data))
            # 计算数据batch数
            batch_idxs = min(len(data), self.train_size) // self.batch_size
            # 记录测试开始时间
            start_time = time.time()
            for idx in range(0, batch_idxs):
                # 获取test图像及标签
                batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch = self.get_data(batch_files)
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                batch_labels = [self.get_label(batch_file) for batch_file in batch_files]

                # 获取预测值和预测概率（注意，测试阶段keep_prob设为1，即不dropout结点）
                predict_prob,predict_logits = sess.run([self.CNN,self.CNN_logits], feed_dict={self.inputs: batch_images, self.labels: batch_labels,self.keep_prob:1.0})
                pred = np.argmax(predict_prob, 1)
                real = np.argmax(batch_labels, 1)
                print(real)

                # 获取所有文件名，用于后面打印准确率时显示出文件名字
                file_names = [os.path.basename(batch_file) for batch_file in batch_files]

                # 初始化TP，TN，FP，FN，SEN，SPC（True Positive，True Negative，False Positive，False Negative， sensitivity，specificity）
                TP = TN = FP = FN = SEN = SPC = 0
                for i in range(self.batch_size):
                    # print("image name:%s,\t\t real_label:%d,\t predict_label:%d" % (file_names[i], real[i], pred[i]))
                    # 计算敏感度，特异度等
                    # True Positive：本来是正样例，分类成正样例
                    # True Negative：本来是负样例，分类成负样例
                    # False Positive ：本来是负样例，分类成正样例，通常叫误报。
                    # False Negative：本来是正样例，分类成负样例，通常叫漏报。
                    if real[i] == 0 and pred[i] == 0:
                        TP += 1
                    elif real[i] == 1 and pred[i] == 1:
                        TN += 1
                    elif real[i] == 1 and pred[i] == 0:
                        FP += 1
                    elif real[i] == 0 and pred[i] == 1:
                        FN += 1
                esp = 1e-7
                SEN = TP / (TP + FN+esp)  # 敏感性
                SPC = TN / (TN + FP+esp)  # 特异性
                ACC = (TP + TN) / (TP + TN + FP + FN+esp)  # 准确率
                # print("batch:[%4d/%4d] time: %4.4f,test sensitivity: %.8f,specificity: %.8f,accuracy: %.8f"
                #       % (idx, batch_idxs, time.time() - start_time, SEN, SPC, ACC))

                #计算AUC
                real_labels_binary = [self.get_svm_label(batch_file) for batch_file in batch_files]
                positive_scores=[prob[1] for prob in predict_prob]
                auc = roc_auc_score(np.array(real_labels_binary), positive_scores)
                # print('ACC softmax test:', ACC)
                # print('AUC softmax test:', auc)
                sys.stdout.flush()
                return predict_prob,pred,real,ACC,auc,file_names

    #之前有尝试用SVM做分类器，后面做多折之后该部分代码没有维护了
    def SVM_classify(self, sess):
        print("Start SVM_classify ...")
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        # 重载训练好的参数
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            #如果检查点存在，则加载最新的检查点
            model_file = tf.train.latest_checkpoint(self.checkpoint_dir)
            self.saver.restore(sess, model_file)
        else:
            print('No checkpoint file found')
            return

        # 根据文件路径获取数据（前200行是训练样本，后72行为测试样本，样本无重复）
        data_train = glob(os.path.join(self.SVM_tr_data_dir, "*.jpg"))
        # data_train = glob(os.path.join(self.train_data_dir, "*.jpg"))
        data_test = glob(os.path.join(self.SVM_te_data_dir, "*.jpg"))
        train_num=len(data_train)
        test_num=len(data_test)
        data=data_train+data_test


        #只测试训练集上的准确率
        # data = glob(os.path.join(self.data_dir + "\\train\\", "*.jpg"))
        # train_num = 160
        # test_num = 40
        # random.shuffle(data)

        # 这里直接把所有数据一批输进去
        self.batch_size=len(data)
        # 计算数据batch数
        batch_idxs = min(len(data), self.train_size) // self.batch_size
        # 记录测试开始时间
        start_time = time.time()
        for idx in range(0, batch_idxs):
            # 获取test图像及标签
            # 读取jpg数据
            batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch = self.get_data(batch_files)
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]

            #提取第一层全连接层的输出特征作为svm的输入
            feature = self.sess.run(self.fc1, feed_dict={self.inputs: batch_images,self.keep_prob:1.0})
            # feature2 = self.sess.run(self.CNN_logits, feed_dict={self.inputs: batch_images, self.keep_prob: 1.0})
            # feature = np.hstack((feature1,feature2))
            svm_X=feature.tolist()
            svm_Y=[self.get_svm_label(batch_file) for batch_file in batch_files]

            '''if self.train_flag==3:
                print('use EasyMKL')
                use_EasyMKL(svm_X,svm_Y,train_num,test_num)
            elif self.train_flag==4:
                print('use Normal SVM')
                use_svm(svm_X,svm_Y,train_num,test_num)
            elif self.train_flag == 5:
                print('use Imbalance SVM')
                use_imbalance_SVM(svm_X, svm_Y, train_num, test_num)'''



    #根据文件名列表获取jpg图像,并做了归一化
    def get_data(self, batch_files):
        images=[]
        for batch_file in batch_files:
            image=scipy.misc.imread(batch_file, flatten=True).astype(np.float)
            # image = np.nan_to_num(image)
            if image.max() <= 0.1:
                print(batch_file)
                print(image.max())
                image_scaled = image
            else:
                image_scaled= (image-image.min())/(image.max())                   #归一化
            images.append(image_scaled)
        return images

    #根据文件路径获取文件名，然后根据文件名判断该图像类型，并给出相应标签。jpg文件名中必须包含文件所属类别才能识别出来
    def get_label(self,path):
        filename=os.path.basename(path)
        label = []
        if 'AD' in filename:
            label=[1,0]
        elif 'MCIc' in filename:
            label = [1, 0]
        elif 'HC' in filename:
            label=[0,1]
        elif 'MCInc' in filename:
            label=[0,1]
        else:
            raise NameError('image name cannot be resolved,image name should be "AD_***" or "HC_***" ')
        return label

    #获取SVM标签，与前面的softmax标签不同，svm分类器所用图像标签只需要一个数来表示，-1 || +1
    def get_svm_label(self,path):
        filename=os.path.basename(path)
        if 'AD' in filename:
            label=[-1]
        elif 'MCIc' in filename:
            label = [-1]
        elif 'HC' in filename:
            label=[1]
        elif 'MCInc' in filename:
            label=[1]
        else:
            raise NameError('image name cannot be resolved,image name should be "AD_01" or "HC_01" ')
        return label

    # 用这个函数是因为每个卷积层的输出有多个特征图，每次只能保存其中一张.这里的max_outputs控制每次从一个batch中抽几张图片来进行显示
    def summary_image(self,name,image_mat,max_outputs=5,filter_idx=1):
        image_mat=image_mat[:,:,:,filter_idx]
        image_mat=tf.reshape(image_mat, [image_mat.shape[0], image_mat.shape[1], image_mat.shape[2],1])
        return tf.summary.image(name, image_mat, max_outputs=max_outputs)

    #这个函数根据给出的变量名获取变量的值，调试时使用
    def get_weight_bias(self,name):
        #用来获取各层的权值w和偏置b
        tensor = tf.get_default_graph().get_tensor_by_name(name)
        return self.sess.run(tensor)

    #将当前模型保存到checkpoint_dir里，step是一个全局迭代次数，保证每个模型的文件名最后几位不一样
    def save_model(self,sess,step,checkpoint_dir=''):
        if checkpoint_dir=='':
            checkpoint_dir=self.checkpoint_dir
        saver=self.saver
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        print(self.thread_name+' model saved to %s' % (checkpoint_dir))

    #备份模型，当验证集和训练集上的准确率达到某个条件时，认为当前模型是较好的模型，将其复制到另一个文件夹下。
    #因为checkpoint_dir里面保存的模型数有参数限制，一般保存5个，前面的模型会被新的覆盖掉。所以要备份到另一个文件夹
    def backup_model(self,sess,counter):
        # 保存当前checkpoint
        self.save_model(sess, counter)
        backup_dir = os.path.join(self.checkpoint_dir, 'best_checkpoint')
        # 创建备份文件夹,为了节省空间，只保存最后一个最优模型，因此每次备份模型之前把备份文件夹清空
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        # else:
        #     shutil.rmtree(backup_dir)
        #     os.makedirs(backup_dir)
        # 把当前模型复制到best_checkpoint文件夹下，每个保存点有三个文件，加上保存点列表有4个文件，都要复制
        curr_checkpoint_file0 = os.path.join(self.checkpoint_dir, "checkpoint")
        curr_checkpoint_file1 = os.path.join(self.checkpoint_dir, "model.ckpt-" + str(counter) + ".data-00000-of-00001")
        curr_checkpoint_file2 = os.path.join(self.checkpoint_dir, "model.ckpt-" + str(counter) + ".index")
        curr_checkpoint_file3 = os.path.join(self.checkpoint_dir, "model.ckpt-" + str(counter) + ".meta")
        shutil.copy(curr_checkpoint_file0, backup_dir)
        shutil.copy(curr_checkpoint_file1, backup_dir)
        shutil.copy(curr_checkpoint_file2, backup_dir)
        shutil.copy(curr_checkpoint_file3, backup_dir)

    #辅助函数，用来保存中间卷积层结果
    def save_img(self,full_save_name, img):
        save_dir=os.path.dirname(full_save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        scipy.misc.imsave(full_save_name, img)
######################################################################################################################################
#以上为CNN_classifier类的内容。

def pretrain_thread(run_config,conf):
    print('Start pre train...')
    CNN_classifier(run_config, conf).train_CAE()

#多线程运行train
def train_thread(run_config,conf):
        print('Start ' + conf.thread_name)
        CNN_classifier(run_config, conf).train()

#保存ROC曲线
'''def plot_and_save_ROC(fpr,tpr,roc_auc,savepath):
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.8f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    #是否打印到屏幕上
    # plt.show()'''


#保存test报告文件
def save_report_txt(savepath,filename,content,trunc=False):
    #清空文件夹
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    #创建新文件或追加
    filename=os.path.join(savepath,filename)
    if not os.path.exists(filename) or trunc==True:
        f = open(filename, 'w')
        f.truncate()
        f.write(content + '\n')
        f.close()
    else:
        f = open(filename, 'a')
        f.write(content+'\n')
        f.close()
    print(content)

#获取文件名公共部分（去掉文件名坐标信息如X40,Z39），用于多切片融合后显示公共文件名
def get_subjectID_from_filename(files_names):
    subjectIDs = []
    for name in files_names:
        name_split = name.split('_')
        del name_split[2]
        subjectID = '_'.join(name_split)
        subjectIDs.append(subjectID)
    return subjectIDs

#融合不同坐标轴的分类信息
def merge_different_axises_result(conf,file_names,real_labels,pred_lables,pred_probs):
    ##将三个坐标轴的结果融合起来
    pred_lables_merge=[]
    pred_prob_merge=[]
    ACC_merge=[]
    AUC_merge=[]
    
    #由于pred_lables是按每个坐标轴/每个fold的顺序叠加在一个矩阵里的，所以前5条是X轴的结果，中间5条是Y轴的结果，后面5条是Z轴的结果。
    #现在要按坐标轴将相应的fold结合起来，即将0,5,10三行投票合并为一行作为fold1的判断值;1,6,11三行投票合并作为fold2的判断值...
    #这个循环主要用来处理pred_lables,pred_probs，使其按fold合并，将15行的矩阵merge成5行
    for fold in range(conf.fold_num):
        fold_pred=[]
        fold_pred_prob=[]
        for i in range(len(conf.slices_pos_all)):
            fold_pred.append(pred_lables[fold + i * conf.fold_num])
            fold_pred_prob.append(pred_probs[fold + i * conf.fold_num])

        threshold = len(fold_pred) / 2
        pred_vote = np.sum(fold_pred, axis=0)
        pred_vote[pred_vote < threshold] = 0
        pred_vote[pred_vote >= threshold] = 1
        pred_lables_merge.append(pred_vote)

        pred_prob_vote=np.mean(fold_pred_prob,axis=0)
        pred_prob_merge.append(pred_prob_vote)

    #设置保存文件的文件名，定义第一行(str0)和分割线（str9）
    report_txt_name = 'Test_Report_' + conf.classify_name + '_all_fold_all_axis.txt'
    str0 = '********************* ' + report_txt_name + ' ***********************'
    str9 = '*****************************************************************************************'
    # 第一次向日志文件写入时，要指定是否清空里面的内容，trunc=True时表示本次清空之前调试的信息，trunc=False表示保留之前的调试信息
    save_report_txt(conf.base_report_dir, report_txt_name, str0, trunc=True)

    #该循环用来保存每个fold的三轴Ensembel ACC
    for fold in range(conf.fold_num):
        # curr_filenames=file_names[fold]
        # curr_filenames=get_subjectID_from_filename(curr_filenames)
        curr_real=real_labels[fold]
        curr_pred=pred_lables_merge[fold]
        acc_vote = accuracy_score(curr_real, curr_pred)
        ACC_merge.append(acc_vote)
        str1 = 'fold'+str(fold) +' voted ACC of all axises is :' + str(acc_vote)
        save_report_txt(conf.base_report_dir, report_txt_name, str1)
    #打印ACC的统计信息，包括平均值，方差等
    mean_ACC=np.mean(ACC_merge)
    std_ACC = np.std(ACC_merge)                 # 有偏估计,除以n
    std_ACC_ddof_1 = np.std(ACC_merge, ddof=1)  # 无偏估计，除以n-1
    str_accs='ALL_ACC:'+str(ACC_merge)
    str_mean_acc = 'Mean ACC of all folds is :【' + str(mean_ACC) + '】'
    str_std_acc = 'ACC Standard Deviation(devide n) of all folds is :【' + str(std_ACC) + '】'
    str_std_acc_ddof_1 = 'ACC Standard Deviation(devide n-1) of all folds is :【' + str(std_ACC_ddof_1) + '】'
    save_report_txt(conf.base_report_dir, report_txt_name, str_accs)
    save_report_txt(conf.base_report_dir, report_txt_name, str_mean_acc)
    save_report_txt(conf.base_report_dir, report_txt_name, str_std_acc)
    save_report_txt(conf.base_report_dir, report_txt_name, str_std_acc_ddof_1)
    save_report_txt(conf.base_report_dir, report_txt_name, str9)

    #该循环用来保存每个fold的三轴Ensembel AUC
    for fold in range(conf.fold_num):
        curr_real = real_labels[fold]
        curr_pred_prob = pred_prob_merge[fold]
        auc_vote = roc_auc_score(np.array(curr_real), curr_pred_prob)
        AUC_merge.append(auc_vote)
        str2 = 'fold' + str(fold) + ' voted AUC of all axises is :' + str(auc_vote)
        save_report_txt(conf.base_report_dir, report_txt_name, str2)
        # 画ROC曲线并保存图片
        fpr, tpr, threshold = roc_curve(np.array(curr_real), curr_pred_prob)  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        image_save_path = os.path.join(conf.base_report_dir,conf.classify_name + '_fold' + str(fold)+ '_model_ROC.png')
        # plot_and_save_ROC(fpr, tpr, roc_auc, image_save_path)  # 自定义了一个画图函数

    #打印AUC的统计信息，包括平均值，方差等
    mean_AUC=np.mean(AUC_merge)
    std_AUC = np.std(AUC_merge)                 # 有偏估计,除以n
    std_AUC_ddof_1 = np.std(AUC_merge, ddof=1)  # 无偏估计，除以n-1
    str_aucs='ALL_AUC:'+str(AUC_merge)
    str_mean_auc = 'Mean AUC of all folds is :【' + str(mean_AUC) + '】'
    str_std_auc = 'AUC Standard Deviation(devide n) of all folds is :【' + str(std_AUC) + ']'
    str_std_auc_ddof_1 = 'AUC Standard Deviation(devide n-1) of all folds is :【' + str(std_AUC_ddof_1) + '】'
    save_report_txt(conf.base_report_dir, report_txt_name, str_aucs)
    save_report_txt(conf.base_report_dir, report_txt_name, str_mean_auc)
    save_report_txt(conf.base_report_dir, report_txt_name, str_std_auc)
    save_report_txt(conf.base_report_dir, report_txt_name, str_std_auc_ddof_1)
    save_report_txt(conf.base_report_dir, report_txt_name, str9)


# 融合不同坐标轴的分类信息
def merge_different_axises_result_and_return(conf,real_labels, pred_lables, pred_probs):
    ##将三个坐标轴的结果融合起来
    pred_lables_merge = []
    pred_prob_merge = []
    ACC_merge = []
    AUC_merge = []

    # 由于pred_lables是按每个坐标轴/每个fold的顺序叠加在一个矩阵里的，所以前5条是X轴的结果，中间5条是Y轴的结果，后面5条是Z轴的结果。
    # 现在要按坐标轴将相应的fold结合起来，即将0,5,10三行投票合并为一行作为fold1的判断值;1,6,11三行投票合并作为fold2的判断值...
    # 这个循环主要用来处理pred_lables,pred_probs，使其按fold合并，将15行的矩阵merge成5行
    # for fold in range(conf.fold_num):
    for fold in range(conf.fold_num):
        fold_pred = []
        fold_pred_prob = []
        for i in range(len(conf.slices_pos_all)):
            fold_pred.append(pred_lables[fold + i * conf.fold_num])
            fold_pred_prob.append(pred_probs[fold + i * conf.fold_num])
            # fold_pred.append(pred_lables[i])
            # fold_pred_prob.append(pred_probs[i])

        threshold = len(fold_pred) / 2
        pred_vote = np.sum(fold_pred, axis=0)
        pred_vote[pred_vote < threshold] = 0
        pred_vote[pred_vote >= threshold] = 1
        pred_lables_merge.append(pred_vote)

        pred_prob_vote = np.mean(fold_pred_prob, axis=0)
        pred_prob_merge.append(pred_prob_vote)

    # print(pred_lables_merge)
    # 该循环用来保存每个fold的三轴Ensembel ACC
    for fold in range(conf.fold_num):
    # for fold in conf.fold_num:
        # curr_filenames=file_names[fold]
        # curr_filenames=get_subjectID_from_filename(curr_filenames)
        curr_real = real_labels[fold]
        curr_pred = pred_lables_merge[fold]
        acc_vote = accuracy_score(curr_real, curr_pred)
        ACC_merge.append(acc_vote)
        str1 = 'fold' + str(fold) + ' voted ACC of all axises is :' + str(acc_vote)

    # 打印ACC的统计信息，包括平均值，方差等
    mean_ACC = np.mean(ACC_merge)
    std_ACC = np.std(ACC_merge)  # 有偏估计,除以n
    std_ACC_ddof_1 = np.std(ACC_merge, ddof=1)  # 无偏估计，除以n-1


    # 该循环用来保存每个fold的三轴Ensembel AUC
    for fold in range(conf.fold_num):
    # for fold in conf.fold_num:
        curr_real = real_labels[0]
        curr_pred_prob = pred_prob_merge[0]
        auc_vote = roc_auc_score(np.array(curr_real), curr_pred_prob)
        AUC_merge.append(auc_vote)
        str2 = 'fold' + str(fold) + ' voted AUC of all axises is :' + str(auc_vote)

        # 画ROC曲线并保存图片
        fpr, tpr, threshold = roc_curve(np.array(curr_real), curr_pred_prob)  ###计算真正率和假正率


    # 打印AUC的统计信息，包括平均值，方差等
    mean_AUC = np.mean(AUC_merge)
    std_AUC = np.std(AUC_merge)  # 有偏估计,除以n
    std_AUC_ddof_1 = np.std(AUC_merge, ddof=1)  # 无偏估计，除以n-1
    return mean_ACC,mean_AUC

#如果测试时，要使用保存的多个模型，为了保持原来的test结构不变，则调用这个函数来实现多个模型的Ensembel
def test_on_best_models(CNN,sess,conf):
    G_predict_prob_positive=[]
    G_predict_prob_negative=[]
    G_pred=[]
    G_real=[]
    G_ACC=[]
    G_AUC=[]
    G_file_names=[]
    pred_lables_merge=[]
    pred_prob_merge=[]

    model_paths = glob(os.path.join(conf.checkpoint_dir, "*.index"))
    model_num=len(model_paths)
    for i in range(model_num):
        model_name=model_paths[i].split('.index')[0]
        print('load model:',model_name)
        CNN.model_name=model_name
        predict_prob, pred, real, ACC, AUC, file_names = CNN.test(sess)
        G_real=real
        G_file_names=file_names
        predict_prob_positive=np.array(predict_prob)[:,1]
        predict_prob_negative = np.array(predict_prob)[:, 0]
        G_predict_prob_positive.append(predict_prob_positive)
        G_predict_prob_negative.append(predict_prob_negative)
        # G_predict_prob=predict_prob
        G_pred.append(pred)
        G_ACC.append(ACC)
        G_AUC.append(AUC)
    #Enselbel
    threshold = model_num / 2
    pred_vote = np.sum(G_pred, axis=0)
    pred_vote[pred_vote < threshold] = 0
    pred_vote[pred_vote >= threshold] = 1
    # pred_lables_merge=pred_vote


    pred=pred_vote
    ACC=accuracy_score(G_real,pred_vote)

    positive_prob_mean = np.mean(G_predict_prob_positive, axis=0)
    negative_prob_mean = np.mean(G_predict_prob_negative, axis=0)

    zip_prob=zip(negative_prob_mean,positive_prob_mean)
    predict_prob=np.array([z for z in zip_prob] )
    # auc_vote = roc_auc_score(np.array(G_real), positive_prob_mean)
    # predict_prob=
    # real=
    # ACC=
    # AUC=
    # file_names=
    # pred_prob_vote = np.mean(fold_pred_prob, axis=0)
    # pred_prob_merge.append(pred_prob_vote)
    return predict_prob, pred, real, ACC, AUC, file_names








class user_config:
    def __init__(self):
        #进行多折交叉验证
        self.fold_num =5
        #要训练的slices(x,y,z轴上的切片坐标)
        self.slices_pos_all=[[39, 40, 41]]
        #当前训练的分类
        self.classify_name='AD_HC'
        #所有坐标轴
        self.axises = ['X', 'Y', 'Z']
        #数据主目录(注意，路径中的斜杠linux系统用'/',windows用'\')
        self.base_data_dir=os.path.normpath('./Data/ADNI/slices/AD_HC')
        # 保存点主目录
        self.base_checkpoint_dir = os.path.normpath('./checkpoint/AD_HC')
        # 模型test report文件的保存主目录
        self.base_report_dir = os.path.normpath('./report/AD_HC')
        self.pretrain_checkpoint_dir=''
        #这里表示进行什么操作，1表示train，2表示test(softmax分类器)，3/4/5分别表示用不同的SVM分类器进行分类
        self.train_flag = 2
        #如果train_flag = 1,则该参数生效，表示是否接着保存点里的模型开始训练。
        self.use_saved_model = False
        # 该参数在训练时用到，每训练一段时间就会测试一下在测试集上的准确率，这时需要这个参数来建立测试的模型
        self.test_data_num = 50
        #是否使用多折交叉验证
        self.use_cross_validation = False
        #是否使用进程池
        self.use_process_pool = True
        # 训练的batch大小
        self.batchsize = 200
        # 训练的epoch数
        self.max_epoch=10
        #学习率
        self.learning_rate=0.0001

        self.use_best_one_model = True
        self.base_valid_data_dir=''
        self.train_data_num=100
        self.model_idx=''

        ##以下参数在main里面由函数自动得到
        # 图像所在的文件夹
        self.data_dir = ''
        # 训练图像所在的文件夹(带aug)
        self.train_data_dir = ''
        # 测试图像所在的文件夹
        self.test_data_dir = ''
        # 保存点所在的文件夹
        self.checkpoint_dir = ''
        # 模型测试报告保存的目录
        self.report_dir = ''
        # svm训练图像所在的文件夹（不带aug）
        self.SVM_tr_data_dir = ''
        # svm测试图像所在的文件夹
        self.SVM_te_data_dir = ''
        # 进程名称
        self.thread_name=''



def concat_path(conf,fold='',axis='',z=''):
    ##使用os.path.join()的好处是，可以屏蔽linux和windows中路径斜杠的差异
    if fold=='':
        conf.data_dir = os.path.join(conf.base_data_dir,axis+str(z),'smooth')    # 训练图像所在的文件夹
        conf.train_data_dir = os.path.join(conf.data_dir, 'train.txt')      # 训练图像所在的文件夹
        conf.test_data_dir = os.path.join(conf.data_dir, 'test.txt')            # 测试图像所在的文件夹
        conf.checkpoint_dir = os.path.join(conf.base_checkpoint_dir,axis+str(z)) # 检查点所在文件夹
        conf.report_dir = os.path.join(conf.base_report_dir,axis,'not_use_fold')      # 测试报告所在文件夹
        conf.SVM_tr_data_dir = os.path.join(conf.data_dir, 'train.txt')         # svm训练图像所在的文件夹
        conf.SVM_te_data_dir = os.path.join(conf.data_dir, 'test.txt')          # svm测试图像所在的文件夹
        conf.valid_data_dir = os.path.join(conf.base_valid_data_dir,axis+str(z),'smooth','valid.txt')
        conf.pretrain_checkpoint_dir = os.path.join(conf.base_pretrain_checkpoint_dir, 'PreTrain_6',axis + str(z) )
        conf.pretrain_data_dir=os.path.join(conf.base_pretrain_data_dir,axis+str(z),'smooth','all','pretrain2.txt')
    else:
        conf.data_dir = os.path.join(conf.base_data_dir,axis+str(z),'smooth')                    # 训练图像所在的文件夹
        conf.train_data_dir = os.path.join(conf.data_dir, 'fold'+str(fold)+'_train.txt')        # 训练图像所在的文件夹
        conf.test_data_dir = os.path.join(conf.data_dir, 'fold'+str(fold)+'_train.txt')          # 测试图像所在的文件夹
        conf.checkpoint_dir = os.path.join(conf.base_checkpoint_dir,'fold'+str(fold),axis+str(z))# 检查点所在文件夹
        conf.report_dir = os.path.join(conf.base_report_dir,axis,'use_fold')  # 测试报告所在文件夹
        conf.SVM_tr_data_dir = os.path.join(conf.data_dir, 'fold'+str(fold)+'_train.txt')       # svm训练图像所在的文件夹
        conf.SVM_te_data_dir = os.path.join(conf.data_dir, 'fold'+str(fold)+'_test.txt')        # svm测试图像所在的文件夹
        conf.valid_data_dir = os.path.join(conf.base_valid_data_dir, axis + str(z), 'smooth', 'valid.txt')
        conf.pretrain_checkpoint_dir = os.path.join(conf.base_pretrain_checkpoint_dir, 'PreTrain_6',axis + str(z))
        conf.pretrain_data_dir = os.path.join(conf.base_pretrain_data_dir, axis + str(z), 'smooth', 'all','pretrain2.txt')

        conf.test_data_dir = conf.valid_data_dir          # 输出测试集结果
    return conf


def main(_):
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # 这两句是对GPU的使用进行配置
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    ##控制台：控制运行的关键参数都在这里设置#######################################
    
    # slices_pos_x = [40, 61, 80]
    # slices_pos_y = [69, 71, 72]
    # slices_pos_z = [39, 40, 41]
    #slices_pos_x = [x for x in range(40, 80, 2)]
    #slices_pos_y = [x for x in range(60, 100, 2)]
    #slices_pos_z = [x for x in range(35, 75, 2)]
    # slices_pos_z = [40]
    slices_pos_x = [x for x in range(20, 100, 2)]
    slices_pos_y = [x for x in range(24, 124, 2)]
    slices_pos_z = [x for x in range(30, 96, 2)]

    # slices_pos_x = [72, 74, 70, 78, 48]#, 82, 38, 46, 30, 40]
    # slices_pos_y = [76, 80, 78, 64, 82]#, 94, 74, 62, 48, 56]
    # slices_pos_z = [38, 32, 44, 34, 30]#, 36, 68, 50, 46, 40]

    slices_pos_x = [60,62]
    slices_pos_y = [84,86]
    slices_pos_z = [40,42]

    # slices_pos_x =[56, 76, 42]#, 98, 62]#, 36, 40, 94, 84]
    slices_pos_all = [slices_pos_x, slices_pos_y, slices_pos_z] #x,y,z轴上的切片位置
    axises = ['X', 'Y', 'Z']                                    #文件夹前缀
    # slices_pos_all = [slices_pos_y,slices_pos_z]                             # x,y,z轴上的切片位置
    # axises = ['Y','Z']

    classify_names=['AD_HC','MCIc_HC','MCIc_MCInc']             #所有分类类别
    classify_names_t=['test_AD_HC','test_MCIc_HC','test_MCIc_MCInc']             #所有分类类别

    conf = user_config()                    #新建一个conf对象
    conf.train_flag = 2                     #执行什么操作（1：train，2：test）
    conf.use_saved_model=False              #该参数在train时有用，True表示是否接着原先保存的模型训练
    conf.use_cross_validation=True          #是否使用多折交叉验证，True表示使用，此时还需设置conf.fold_num值
    conf.use_process_pool=True             #train阶段是否使用进程池，True表示使用，此时并发执行训练，并发数由conf.pool_mult_num控制
    conf.use_best_one_model = True         #test阶段是否使用最好的一个模型进行测试，如果为FALSE，则使用保存的最好的k个模型进行测试
    
    conf.max_epoch=30                      #训练次数
    conf.learning_rate=0.0001               #学习率
    conf.fold_num=5                         #进行多少折交叉验证
    conf.pool_mult_num=40                   #进程池最大并发进程数
    conf.batchsize = 200                     #模型训练时训练数据的batchsize
    conf.classify_name=classify_names[2]    #当前训练哪种分类
    conf.classify_name_t = classify_names_t[1]
    ################################################################################

    conf.slices_pos_all = slices_pos_all
    conf.axises=axises
    workdir=os.path.dirname(sys.argv[0])
    conf.base_data_dir= os.path.join(workdir,'Data','ADNI','slices',conf.classify_name)
    conf.base_valid_data_dir = os.path.join(workdir, 'Data', 'ADNI', 'slices', conf.classify_name_t)
    conf.base_checkpoint_dir = os.path.join(workdir,'checkpoint',conf.classify_name)
    conf.base_report_dir=   os.path.join(workdir,'report',conf.classify_name)
    conf.base_pretrain_checkpoint_dir = os.path.join(workdir, 'checkpoint')
    conf.base_pretrain_data_dir=os.path.join(workdir, 'Data','ADNI','slices','PreTrain')

    if conf.train_flag == 0:
        pool = multiprocessing.Pool(conf.pool_mult_num)
        conf.max_epoch = 200  # 训练次数
        conf.batchsize = 10
        conf.learning_rate = 0.001  # 学习率
        for p in range(len(slices_pos_all)):
            slices_pos = conf.slices_pos_all[p]
            axis = conf.axises[p]
            for z in slices_pos:
                # 设置数据及模型保存路径
                conf = concat_path(conf, axis=axis, z=z)

                if not os.path.exists(conf.pretrain_checkpoint_dir):
                    os.makedirs(conf.pretrain_checkpoint_dir)

                conf.thread_name = 'Thread:PreTrain ' + axis + str(z)

                if conf.use_process_pool:
                    conf_thread = copy.deepcopy(conf)  # 保证每个进程独享一个conf参数
                    pool.apply_async(pretrain_thread, (run_config, conf_thread,))  # 加入到进程池，且各进程相互独立，可以同时训练pool_mult_num个模型
                else:
                    conf_thread = copy.deepcopy(conf)  # 保证每个进程独享一个conf参数
                    pool.apply(pretrain_thread, (run_config, conf_thread,))  # 每个进程要等上一个进程运行完才运行，一次只运行一个模型
        # 主进程阻塞，等待子进程的退出， join方法要在close或terminate之后使用
        pool.close()
        pool.join()
        print('All processes finished')

    elif conf.train_flag == 1:
        #如果进程太多，则使用进程池控制并发数
        pool = multiprocessing.Pool(conf.pool_mult_num)
        for p in range(len(slices_pos_all)):
            slices_pos = conf.slices_pos_all[p]
            axis = conf.axises[p]
            if conf.use_cross_validation:
                for fold in range(conf.fold_num):
                    for z in slices_pos:
                        # 设置数据及模型保存路径
                        conf = concat_path(conf, fold=fold,axis=axis,z=z)
                        # 设置test_data_num
                        # test_data = np.loadtxt(conf.valid_data_dir, dtype='str')
                        # conf.test_data_num = len(test_data)

                        train_data = np.loadtxt(conf.SVM_tr_data_dir, dtype='str')
                        conf.train_data_num=len(train_data)

                        # 检查模型保存路径是否存在
                        if not os.path.exists(conf.checkpoint_dir):
                            os.makedirs(conf.checkpoint_dir)
                        # 开始进行多进程训练
                        conf.thread_name='Thread:'+conf.classify_name+' fold'+str(fold)+' '+axis+ str(z)
                        #该变量主要是为了多进程时，各个模型创建的参数能在不同的scope下，不至于混淆
                        conf.model_idx='f'+str(fold)+axis+str(z)


                        # try:
                        if conf.use_process_pool:
                            conf_thread= copy.deepcopy(conf)#保证每个进程独享一个conf参数
                            pool.apply_async(train_thread, (run_config, conf_thread,))#加入到进程池，且各进程相互独立，可以同时训练pool_mult_num个模型
                        else:
                            conf_thread= copy.deepcopy(conf)#保证每个进程独享一个conf参数
                            pool.apply(train_thread, (run_config, conf_thread,))#每个进程要等上一个进程运行完才运行，一次只运行一个模型

                            #实现多进程的另一种方法（不带并发数控制）
                            # conf_thread = copy.deepcopy(conf)  # 保证每个进程独享一个conf参数
                            # p = multiprocessing.Process(target=train_thread, args=(run_config, conf_thread))
                            # p.start()
                            # print("p.pid:", p.pid)
                            # print("p.name:", p.name)
                            # print("p.is_alive:", p.is_alive())
                            # print("**************************")
                        # except:
                        #     print("Error: unable to start thread")

            else:
                # 循环训练3个切片，并分别保存三个切片上最好的模型，这里用了多进程并行，提高运行效率
                for z in slices_pos:
                    #设置数据及模型保存路径
                    conf=concat_path(conf,axis=axis,z=z)
                    #设置test_data_num
                    test_data = np.loadtxt(conf.test_data_dir, dtype='str')
                    conf.test_data_num = len(test_data)
                    #检查模型保存路径是否存在
                    if not os.path.exists(conf.checkpoint_dir):
                        os.makedirs(conf.checkpoint_dir)

                    #开始进行多进程训练
                    conf.thread_name = 'Thread:'+conf.classify_name+' '+axis+str(z)
                    #该变量主要是为了多进程时，各个模型创建的参数能在不同的scope下，不至于混淆
                    conf.model_idx = axis + str(z)
                    try:
                        if conf.use_process_pool:
                            conf_thread = copy.deepcopy(conf)  # 保证每个进程独享一个conf参数
                            pool.apply_async(train_thread, (run_config, conf_thread,)) #加入到进程池，且各进程相互独立，可以同时训练pool_mult_num个模型
                        else:
                            conf_thread= copy.deepcopy(conf)#保证每个进程独享一个conf参数
                            pool.apply(train_thread, (run_config, conf_thread,))#每个进程要等上一个进程运行完才运行，一次只运行一个模型
                    except:
                        print ("Error: unable to start thread")

        # 主进程阻塞，等待子进程的退出， join方法要在close或terminate之后使用
        pool.close()
        pool.join()
        print('All processes finished')


    elif conf.train_flag == 2:

        G_file_names = []
        G_real_lables = []
        G_pred_voted = []
        G_pred_prob_voted=[]
        for p in range(len(slices_pos_all)):
            slices_pos = conf.slices_pos_all[p]
            axis = conf.axises[p]
            if conf.use_cross_validation:
                global_ACCs=[]
                global_AUCs=[]
                global_ACC_reports=[]
                global_AUC_reports=[]
                global_filenames = []
                global_reallabels =[]

                #循环k个fold
                for fold in range(conf.fold_num):
                    # 定义几个全局变量，用来保存测试时返回的结果
                    te_pred_prob = [],
                    te_pred = []
                    te_ACC = []
                    te_AUC = []
                    te_real = []
                    te_filenames=[]
                    te_idx = 0
                    for z in slices_pos:
                        concat_path(conf, fold=fold, axis=axis, z=z)

                        # conf.test_data_dir=conf.valid_data_dir
                        # conf.test_data_dir = conf.SVM_tr_data_dir
                        #确定batchsize.测试时，把所有的图像一次测完
                        test_data = np.loadtxt(conf.test_data_dir, dtype='str')
                        conf.batchsize = len(test_data)

                        #测试时用最好的模型做测试
                        # conf.checkpoint_dir=os.path.join(conf.checkpoint_dir,'best_checkpoint')
                        #此变量是因为多进程训练时，每次训练保存的模型参数前的前缀不同，所以要通过此变量保持一致
                        conf.model_idx='f'+str(fold)+axis+str(z)
                        if not os.path.exists(conf.checkpoint_dir):
                            print('Can not find checkpoint in checkpoint_dir:'+conf.checkpoint_dir)
                        print('*********************************************************************************')
                        print('start test model on ' +conf.classify_name + ' fold' + str(fold) + ' ' + axis + str(z))
                        with tf.Session(config=run_config) as sess:
                            if conf.use_best_one_model==True:
                                CNN = CNN_classifier(sess,conf)
                                predict_prob, pred,real,ACC, AUC,file_names=CNN.test(sess)
                            else:
                                CNN = CNN_classifier(sess, conf)
                                predict_prob, pred, real, ACC, AUC, file_names=test_on_best_models(CNN,sess,conf)
                                print('ACC:' + str(ACC))
                                print('AUC:' + str(AUC))

                            if te_idx==0:
                                te_pred_prob = predict_prob
                                te_pred = pred
                                te_real = real
                                te_ACC=[ACC]
                                te_AUC=[AUC]
                                te_filenames=file_names

                            else:

                                te_pred_prob=np.c_[te_pred_prob,predict_prob]
                                te_pred=np.c_[te_pred,pred]
                                # te_real = np.c_[te_real, real]        #因为不同切片处的real 标签是一样的，只保留一个切片的real label就行了
                                te_ACC.append(ACC)
                                te_AUC.append(AUC)

                            te_idx =te_idx+1

                    #根据预测标签投票，对三个模型的预测标签求和
                    threshold= len(slices_pos) / 2
                    pred_vote=np.sum(te_pred, axis=1)
                    pred_vote[pred_vote < threshold] = 0
                    pred_vote[pred_vote >= threshold] = 1

                    positive_scores_add=[]

                    #这里求正例概率是为了算AUC
                    #把所有切片对应的正例的概率加起来取平均，因为test函数返回的概率pred是有两列的，第一列是判为0的概率，第二列是判为1的概率。
                    #由于将每个切片处的pred按列拼接，就变成了所有奇数列都是正例概率，要将奇数列加起来求平均得到vote后模型得正例概率。
                    for prob in te_pred_prob:
                        positive_scores=0
                        for i in range(len(slices_pos)):
                            positive_scores += prob[2*i+1]
                        positive_scores_add.append(positive_scores / len(slices_pos))

                    acc_vote=accuracy_score(te_real,pred_vote)
                    auc_vote = roc_auc_score(np.array(te_real), positive_scores_add)

                    te_idx = 0

                    # 打印并保存结果信息(每个fold单独保存一个文件)
                    report_txt_name = 'Test_Report_'+conf.classify_name + '_fold' + str(fold) + '_axis_' + axis +'.txt'
                    str1 = '********************* '+report_txt_name+' ***********************'
                    # 第一次向日志文件写入时，要指定是否清空里面的内容，trunc=True时表示本次清空之前调试的信息，trunc=False表示保留之前的调试信息
                    save_report_txt(conf.report_dir, report_txt_name, str1, trunc=True)
                    str_currtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    title = '*********** Classify model:' + conf.classify_name + '    Report time:' + str_currtime + ' ************'
                    save_report_txt(conf.report_dir, report_txt_name, title)
                    for z in slices_pos:
                        str2 = "Test on %s fold%d %s%d--ACC: %.8f, AUC: %.8f" %(conf.classify_name,fold,axis,z, te_ACC[te_idx], te_AUC[te_idx])
                        save_report_txt(conf.report_dir, report_txt_name, str2)
                        te_idx += 1
                    str3 = conf.classify_name+' Fold'+str(fold)+' axis '+axis+' final voted ACC is :' + str(acc_vote)
                    save_report_txt(conf.report_dir, report_txt_name, str3)

                    str4 = conf.classify_name+' Fold'+str(fold)+' axis '+axis+' final voted AUC is :' + str(auc_vote)
                    save_report_txt(conf.report_dir, report_txt_name, str4)
                    str5 = '********************************************************************************'
                    save_report_txt(conf.report_dir, report_txt_name, str5)

                    # 画ROC曲线并保存图片
                    fpr, tpr, threshold = roc_curve(np.array(te_real), positive_scores_add)  ###计算真正率和假正率
                    roc_auc = auc(fpr, tpr)  ###计算auc的值
                    image_save_path = os.path.join(conf.report_dir, conf.classify_name +'_fold'+str(fold)+'_axis_'+axis+'_model_ROC.png')
                    # plot_and_save_ROC(fpr, tpr, roc_auc, image_save_path)  # 自定义了一个画图函数

                    #把每个fold得到的最终结果保存起来做一个总的report
                    global_ACCs.append(acc_vote)
                    global_AUCs.append(auc_vote)
                    global_ACC_reports.append(str3)
                    global_AUC_reports.append(str4)

                    #由于每个fold不同坐标轴切片所用的病人信息是一样的，所以文件名和其对应的真实标签只需要每折保存一份就好了。5折交叉验证就只需要保存5个数组
                    global_filenames.append(te_filenames)
                    global_reallabels.append(te_real)
                    #由于每个fold不同切片坐标得到的预测值和预测值概率都是不一样的，所以预测值和预测值概率需要每个fold的每种axis的结果都保存下来。最后5折3轴的实验需要保存15个数组
                    G_pred_voted.append(pred_vote)
                    G_pred_prob_voted.append(positive_scores_add)

                #算所有fold的ACC，AUC平均值
                mean_ACC=np.mean(global_ACCs)
                mean_AUC =np.mean(global_AUCs)
                
                #计算ACC，AUC标准差
                std_ACC=np.std(global_ACCs)                     #有偏估计,除以n
                std_ACC_ddof_1 = np.std(global_ACCs,ddof=1)     #无偏估计，除以n-1
                std_AUC=np.std(global_AUCs)
                std_AUC_ddof_1=np.std(global_AUCs,ddof=1)

                #将结果转移到全局变量，因为global_filenames，global_reallabels在下一个循环时会被清空
                G_file_names=global_filenames
                G_real_lables =global_reallabels


                #打印并保存多折交叉验证的平均结果
                #save_report_txt函数里自带了print效果，调用save_report_txt保存信息的同时也会打印到屏幕上。后续如果节约时间可以把函数里的print注释掉
                str1 = 'Mean ACC of all folds is :【' + str(mean_ACC) + '】'
                str2 = 'Mean AUC of all folds is :【' + str(mean_AUC) + '】'

                str10='All ACC:'+str(global_ACCs)
                str11='ACC Standard Deviation(devide n) of all folds is :【' + str(std_ACC)+'】'
                str12='ACC Standard Deviation(devide n-1) of all folds is :【' + str(std_ACC_ddof_1)+'】'

                str20 = 'All AUC:' + str(global_AUCs)
                str21='AUC Standard Deviation(devide n) of all folds is :【' + str(std_AUC)+'】'
                str22='AUC Standard Deviation(devide n-1) of all folds is :【' + str(std_AUC_ddof_1)+'】'

                report_txt_name='Report_Summary_'+conf.classify_name +'_axis_'+axis+'.txt'

                str_currtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                str3 = '***** Classify model:' + conf.classify_name+ '_axis_'+axis+' ,Report time:' + str_currtime + '***Summary*****'
                str4 = '*********************************************************************************'

                save_report_txt(conf.report_dir, report_txt_name, str3, trunc=True)
                for i in range(len(global_ACC_reports)):
                    save_report_txt(conf.report_dir, report_txt_name, global_ACC_reports[i])
                save_report_txt(conf.report_dir, report_txt_name, str10)
                save_report_txt(conf.report_dir, report_txt_name, str1)
                save_report_txt(conf.report_dir, report_txt_name, str11)
                save_report_txt(conf.report_dir, report_txt_name, str12)
                save_report_txt(conf.report_dir, report_txt_name, str4)

                for i in range(len(global_AUC_reports)):
                    save_report_txt(conf.report_dir, report_txt_name, global_AUC_reports[i])
                save_report_txt(conf.report_dir, report_txt_name, str20)
                save_report_txt(conf.report_dir, report_txt_name, str2)
                save_report_txt(conf.report_dir, report_txt_name, str21)
                save_report_txt(conf.report_dir, report_txt_name, str22)
                save_report_txt(conf.report_dir, report_txt_name, str4)

            else:
                # 定义几个全局变量，用来保存测试时返回的结果
                te_pred_prob = [],
                te_pred = []
                te_ACC = []
                te_AUC = []
                te_real = []
                te_idx = 0
                # 循环测试三个切片，用softmax分类器
                for z in slices_pos:
                    concat_path(conf, axis=axis, z=z)
                    # 确定batchsize.测试时，把所有的图像一次测完
                    test_data = np.loadtxt(conf.test_data_dir, dtype='str')
                    conf.batchsize = len(test_data)

                    # 测试时用最好的模型做测试
                    conf.checkpoint_dir = os.path.join(conf.checkpoint_dir, 'best_checkpoint')
                    # 此变量是因为多进程训练时，每次训练保存的模型参数前的前缀不同，所以要通过此变量保持一致
                    conf.model_idx = axis + str(z)
                    # conf.model_idx =''#最开始一批训练好的模型该参数为空


                    if not os.path.exists(conf.checkpoint_dir):
                        print('Can not find checkpoint in checkpoint_dir:' + conf.checkpoint_dir)
                    print('********************************************************************************')
                    print('start test model on ' + conf.classify_name + ' ' + axis + str(z))
                    with tf.Session(config=run_config) as sess:
                        CNN = CNN_classifier(sess, conf)
                        predict_prob, pred, real, ACC, AUC = CNN.test(sess)
                        if te_idx == 0:
                            te_pred_prob = predict_prob
                            te_pred = pred
                            te_real = real
                            te_ACC = [ACC]
                            te_AUC = [AUC]
                        else:
                            te_pred_prob = np.c_[te_pred_prob, predict_prob]
                            te_pred = np.c_[te_pred, pred]
                            # te_real = np.c_[te_real, real]        #因为不同切片处的real 标签是一样的，只保留一个切片的real label就行了
                            te_ACC.append(ACC)
                            te_AUC.append(AUC)
                        te_idx = te_idx + 1

                # 根据预测标签投票，对三个模型的预测标签求和
                threshold = len(slices_pos) / 2
                pred_vote = np.sum(te_pred, axis=1)
                pred_vote[pred_vote < threshold] = 0
                pred_vote[pred_vote >= threshold] = 1

                positive_scores_add = []
                # 把所有切片对应的正例的概率加起来
                for prob in te_pred_prob:
                    positive_scores = 0
                    for i in range(len(slices_pos)):
                        positive_scores += prob[2 * i + 1]
                    positive_scores_add.append(positive_scores / len(slices_pos))

                acc_vote = accuracy_score(te_real, pred_vote)
                auc_vote = roc_auc_score(np.array(te_real), positive_scores_add)

                #计数器置0
                te_idx = 0

                # 打印并保存结果信息(每个fold单独保存一个文件)
                report_txt_name = 'Test_Report_' + conf.classify_name + '_axis_' + axis + '.txt'
                str1 = '************************ ' + report_txt_name + ' **************************'
                # 第一次向日志文件写入时，要指定是否清空里面的内容，trunc=True时表示本次清空之前调试的信息，trunc=False表示保留之前的调试信息
                save_report_txt(conf.report_dir, report_txt_name, str1, trunc=True)
                str_currtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                title = '*********** Classify model:' + conf.classify_name + '    Report time:' + str_currtime + ' ************'
                save_report_txt(conf.report_dir, report_txt_name, title)
                for z in slices_pos:
                    str2 = "Test on %s %s%d--ACC: %.8f, AUC: %.8f" % (conf.classify_name,axis, z, te_ACC[te_idx], te_AUC[te_idx])
                    save_report_txt(conf.report_dir, report_txt_name, str2)
                    te_idx += 1
                str3 = conf.classify_name + ' axis ' + axis + ' final voted ACC is :' + str(acc_vote)
                save_report_txt(conf.report_dir, report_txt_name, str3)

                str4 = conf.classify_name + ' axis ' + axis + ' final voted AUC is :' + str(auc_vote)
                save_report_txt(conf.report_dir, report_txt_name, str4)
                str5 = '********************************************************************************'
                save_report_txt(conf.report_dir, report_txt_name, str5)

                # 画ROC曲线并保存图片
                fpr, tpr, threshold = roc_curve(np.array(te_real), positive_scores_add)  ###计算真正率和假正率
                roc_auc = auc(fpr, tpr)  ###计算auc的值
                image_save_path = os.path.join(conf.report_dir, conf.classify_name + '_axis_' + axis + '_model_ROC.png')
                # plot_and_save_ROC(fpr, tpr, roc_auc, image_save_path)  # 自定义了一个画图函数

                # print('Test end')

        # 根据预测标签投票，对三个模型的预测标签求和


        merge_different_axises_result(conf,G_file_names, G_real_lables, G_pred_voted, G_pred_prob_voted)
        print('Test end')




    # with tf.Session(config=run_config) as sess:
    #     #设置各种路径
    #     data_dir = ".\Data\ADNI\slices\AD_HC\Z40\smooth"         # 训练图像所在的文件夹
    #     train_data_dir = data_dir+"\\train_aug.txt"  # 训练图像所在的文件夹
    #     test_data_dir = data_dir+"\\test.txt"         # 测试图像所在的文件夹
    #     checkpoint_dir=".\AD_HC_checkpoint"
    #
    #     SVM_tr_data_dir = data_dir+"\\train.txt"      # svm训练图像所在的文件夹
    #     SVM_te_data_dir = data_dir+"\\test.txt"        # svm测试图像所在的文件夹
    #
    #     train_flag =1
    #     use_saved_model=False
    #     # use_saved_model = True
    #
    #     if train_flag==1:
    #         CNN = CNN_classifier(sess,train_flag,data_dir,train_data_dir,test_data_dir,SVM_tr_data_dir,SVM_te_data_dir,checkpoint_dir,batchsize=200)
    #         CNN.train(use_saved_model)
    #     elif train_flag==2:
    #         CNN = CNN_classifier(sess,train_flag,data_dir,train_data_dir,test_data_dir,SVM_tr_data_dir,SVM_te_data_dir,checkpoint_dir,batchsize=72)
    #         CNN.test(sess)
    #     elif train_flag==3 or train_flag==4 or train_flag==5:
    #         CNN = CNN_classifier(sess,train_flag,data_dir,train_data_dir,test_data_dir,SVM_tr_data_dir,SVM_te_data_dir,checkpoint_dir,batchsize=612)
    #         CNN.SVM_classify(sess)
    #     else:
    #         print("waring:don't have this option!")

# GWAS and CNN Ensemble demo.py
def get_ACC_AUC(classify_flag,X_indexes,Y_indexes,Z_indexes):
    print ('calculate acc and auc...')
    cf = parse.parse_args()

# def get_ACC_AUC():
    ##classify_flag=0:AD_HC; 1:MCIc_HC;  2:MCIc_MCInc
    ## X_indexes = [40, 61, 80]
    ## Y_indexes = [69, 71, 72]
    ## Z_indexes = [39, 40, 41]

    ##classify_flag=0:AD_HC; 1:MCIc_HC;  2:MCIc_MCInc
    ## X_indexes = [40, 61, 80]
    ## Y_indexes = [69, 71, 72]
    ## Z_indexes = [39, 40, 41]

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # 这两句是对GPU的使用进行配置
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    ##控制台：控制运行的关键参数都在这里设置#######################################
    slices_pos_x = X_indexes
    slices_pos_y = Y_indexes
    slices_pos_z = Z_indexes
    # slices_pos_x = [60,62]
    # slices_pos_y = [84,86]
    # slices_pos_z = [40,42]

    slices_pos_all = [slices_pos_x, slices_pos_y, slices_pos_z]  # x,y,z轴上的切片位置
    axises = ['X', 'Y', 'Z']  # 文件夹前缀

    classify_names = ['AD_HC', 'MCIc_HC', 'MCIc_MCInc']  # 所有分类类别
    classify_names_t = ['test_AD_HC', 'test_MCIc_HC', 'test_MCIc_MCInc']  # 所有分类类别

    conf = user_config()  # 新建一个conf对象
    conf.train_flag = 2  # 执行什么操作（1：train，2：test）
    conf.use_saved_model = False  # 该参数在train时有用，True表示是否接着原先保存的模型训练
    conf.use_cross_validation = True  # 是否使用多折交叉验证，True表示使用，此时还需设置conf.fold_num值
    conf.use_process_pool = True  # train阶段是否使用进程池，True表示使用，此时并发执行训练，并发数由conf.pool_mult_num控制
    conf.use_best_one_model = True  # test阶段是否使用最好的一个模型进行测试，如果为FALSE，则使用保存的最好的k个模型进行测试

    conf.max_epoch = 30  # 训练次数
    conf.learning_rate = 0.0001  # 学习率
    conf.fold_num = 5  # 进行多少折交叉验证
    conf.pool_mult_num = 40  # 进程池最大并发进程数
    conf.batchsize = 200  # 模型训练时训练数据的batchsize
    conf.classify_name = classify_names[classify_flag]  # 当前训练哪种分类
    conf.classify_name_t = classify_names_t[classify_flag]
    ################################################################################

    conf.slices_pos_all = slices_pos_all
    conf.axises = axises
    cv = cf.cv
    # workdir = os.path.dirname(sys.argv[0])
    workdir ='/home/anzeng/jlf/project/CNN_sMRI_2D'
    t_dir = os.path.join('new_corss','cv_{}'.format(cv),conf.classify_name)
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
        print('{} make success'.format(t_dir))
    conf.base_data_dir = os.path.join(workdir, 'Data_leakage{}'.format(cv), 'ADNI', 'slices', conf.classify_name)
    conf.base_valid_data_dir = os.path.join(workdir, 'new_cross', 'ADNI', 'slices', conf.classify_name_t)
    conf.base_checkpoint_dir = os.path.join(workdir, 'cv_checkpoint/cv{}'.format(0), conf.classify_name)
    conf.base_report_dir = os.path.join(workdir, 'report', conf.classify_name)
    conf.base_pretrain_checkpoint_dir = os.path.join(workdir, 'checkpoint')
    conf.base_pretrain_data_dir = os.path.join(workdir, 'Data', 'ADNI', 'slices', 'PreTrain')

    print(conf.base_checkpoint_dir)
    # begin test
    G_file_names = []
    G_real_lables = []
    G_pred_voted = []
    G_pred_prob_voted = []
    G_z_result = []
    G_select_slices = [[] for i in range(5)]
    for p in range(len(slices_pos_all)):
        slices_pos = conf.slices_pos_all[p]
        axis = conf.axises[p]
        if conf.use_cross_validation:
            global_ACCs = []
            global_AUCs = []
            global_ACC_reports = []
            global_AUC_reports = []
            global_filenames = []
            global_reallabels = []

            # 循环k个fold
            # for fold in range(conf.fold_num):
            for fold in range(conf.fold_num):
                # 定义几个全局变量，用来保存测试时返回的结果

                te_pred_prob = [],
                te_pred = []
                te_ACC = []
                te_AUC = []
                te_real = []
                te_filenames = []
                te_idx = 0
                for z in slices_pos:
                    concat_path(conf, fold=fold, axis=axis, z=z)

                    # conf.test_data_dir=conf.valid_data_dir
                    # conf.test_data_dir = conf.SVM_tr_data_dir
                    # 确定batchsize.测试时，把所有的图像一次测完
                    test_data = np.loadtxt(conf.test_data_dir, dtype='str')
                    conf.batchsize = len(test_data)

                    # 测试时用最好的模型做测试
                    # conf.checkpoint_dir=os.path.join(conf.checkpoint_dir,'best_checkpoint')
                    # 此变量是因为多进程训练时，每次训练保存的模型参数前的前缀不同，所以要通过此变量保持一致
                    conf.model_idx = 'f' + str(fold) + axis + str(z)
                    if not os.path.exists(conf.checkpoint_dir):
                        print('Can not find checkpoint in checkpoint_dir:' + conf.checkpoint_dir)
                    # print('*********************************************************************************')
                    # print('start test model on ' + conf.classify_name + ' fold' + str(fold) + ' ' + axis + str(z))
                    with tf.Session(config=run_config) as sess:
                        if conf.use_best_one_model == True:
                            CNN = CNN_classifier(sess, conf)
                            predict_prob, pred, real, ACC, AUC, file_names = CNN.test(sess)
                        else:
                            CNN = CNN_classifier(sess, conf)
                            predict_prob, pred, real, ACC, AUC, file_names = test_on_best_models(CNN, sess, conf)
                            # print('ACC:' + str(ACC))
                            # print('AUC:' + str(AUC))
                        # np.set_printoptions(threshold='nan')
                        # c45_pred = copy.copy(pred)
                        # if(axis=='Z'):
                        #     print(z,fold)
                        #     print(pred)
                        #     print(predict_prob[:,1])
                        c45 = open('c45_'+str(fold)+'.txt', 'a+')
                        c45.write(str(pred)+'\n')
                        c45.close()
                        if te_idx == 0:
                            te_pred_prob = predict_prob
                            #te_pred_prob = np.reshape(te_pred_prob,(len(predict_prob),1))
                            te_pred = pred
                            te_pred = np.reshape(te_pred,(len(pred),1))
                            te_real = real
                            te_ACC = [ACC]
                            te_AUC = [AUC]
                            te_filenames = file_names

                        else:
                            te_pred_prob = np.c_[te_pred_prob, predict_prob]
                            te_pred = np.c_[te_pred, pred]
                            # te_real = np.c_[te_real, real]        #因为不同切片处的real 标签是一样的，只保留一个切片的real label就行了
                            te_ACC.append(ACC)
                            te_AUC.append(AUC)

                        te_idx = te_idx + 1



                # 根据预测标签投票，对三个模型的预测标签求和
                threshold = len(slices_pos) / 2
                print(te_pred.shape)
                pred_vote = np.sum(te_pred, axis=1)
                pred_vote[pred_vote < threshold] = 0
                pred_vote[pred_vote >= threshold] = 1

                for i in range(1,te_pred_prob.shape[1],2):
                    G_select_slices[fold].append(te_pred_prob[:,i])
                # if (axis == 'Z'):
                #     print(fold)
                #     print((te_pred_prob[:,1]+te_pred_prob[:,3]+te_pred_prob[:,5]+te_pred_prob[:,7]+te_pred_prob[:,9])/5)
                #     slices_z = [te_pred_prob[:,1],te_pred_prob[:,3],te_pred_prob[:,5],te_pred_prob[:,7],te_pred_prob[:,9]]
                #     G_z_result.append(slices_z)
                print("============================================")

                positive_scores_add = []

                # 这里求正例概率是为了算AUC
                # 把所有切片对应的正例的概率加起来取平均，因为test函数返回的概率pred是有两列的，第一列是判为0的概率，第二列是判为1的概率。
                # 由于将每个切片处的pred按列拼接，就变成了所有奇数列都是正例概率，要将奇数列加起来求平均得到vote后模型得正例概率。
                for prob in te_pred_prob:
                    positive_scores = 0
                    for i in range(len(slices_pos)):
                        positive_scores += prob[2 * i + 1]
                    positive_scores_add.append(positive_scores / len(slices_pos))


                acc_vote = accuracy_score(te_real, pred_vote)
                print(te_real)
                save_filename = 'valid_label_{}_{}_fold_{}'.format(conf.classify_name,cv,fold)
                save_filename = os.path.join(t_dir,save_filename)
                np.save(save_filename,te_real)
                print('save_successful {}'.format(save_filename))
                auc_vote = roc_auc_score(np.array(te_real), positive_scores_add)
                # print(axis)
                # print(pred_vote)
                # print("----------------------------------------------------------")

                te_idx = 0

                # 打印并保存结果信息(每个fold单独保存一个文件)
                # report_txt_name = 'Test_Report_' + conf.classify_name + '_fold' + str(fold) + '_axis_' + axis + '.txt'
                # str1 = '********************* ' + report_txt_name + ' ***********************'
                # 第一次向日志文件写入时，要指定是否清空里面的内容，trunc=True时表示本次清空之前调试的信息，trunc=False表示保留之前的调试信息
                # save_report_txt(conf.report_dir, report_txt_name, str1, trunc=True)
                # str_currtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                # title = '*********** Classify model:' + conf.classify_name + '    Report time:' + str_currtime + ' ************'
                # save_report_txt(conf.report_dir, report_txt_name, title)
                # for z in slices_pos:
                #     str2 = "Test on %s fold%d %s%d--ACC: %.8f, AUC: %.8f" % (
                #     conf.classify_name, fold, axis, z, te_ACC[te_idx], te_AUC[te_idx])
                #     save_report_txt(conf.report_dir, report_txt_name, str2)
                #     te_idx += 1
                # str3 = conf.classify_name + ' Fold' + str(fold) + ' axis ' + axis + ' final voted ACC is :' + str(
                #     acc_vote)
                # save_report_txt(conf.report_dir, report_txt_name, str3)
                #
                # str4 = conf.classify_name + ' Fold' + str(fold) + ' axis ' + axis + ' final voted AUC is :' + str(
                #     auc_vote)
                # save_report_txt(conf.report_dir, report_txt_name, str4)
                # str5 = '********************************************************************************'
                # save_report_txt(conf.report_dir, report_txt_name, str5)

                # 画ROC曲线并保存图片
                # fpr, tpr, threshold = roc_curve(np.array(te_real), positive_scores_add)  ###计算真正率和假正率
                # roc_auc = auc(fpr, tpr)  ###计算auc的值
                # image_save_path = os.path.join(conf.report_dir, conf.classify_name + '_fold' + str(
                #     fold) + '_axis_' + axis + '_model_ROC.png')
                # # plot_and_save_ROC(fpr, tpr, roc_auc, image_save_path)  # 自定义了一个画图函数

                # 把每个fold得到的最终结果保存起来做一个总的report
                global_ACCs.append(acc_vote)
                global_AUCs.append(auc_vote)
                # global_ACC_reports.append(str3)
                # global_AUC_reports.append(str4)

                # 由于每个fold不同坐标轴切片所用的病人信息是一样的，所以文件名和其对应的真实标签只需要每折保存一份就好了。5折交叉验证就只需要保存5个数组
                global_filenames.append(te_filenames)
                global_reallabels.append(te_real)
                # 由于每个fold不同切片坐标得到的预测值和预测值概率都是不一样的，所以预测值和预测值概率需要每个fold的每种axis的结果都保存下来。最后5折3轴的实验需要保存15个数组
                G_pred_voted.append(pred_vote)
                G_pred_prob_voted.append(positive_scores_add)
            # print(axis)
            # print(np.array(G_pred_voted))

            # 算所有fold的ACC，AUC平均值
            mean_ACC = np.mean(global_ACCs)
            mean_AUC = np.mean(global_AUCs)

            # 计算ACC，AUC标准差
            std_ACC = np.std(global_ACCs)  # 有偏估计,除以n
            std_ACC_ddof_1 = np.std(global_ACCs, ddof=1)  # 无偏估计，除以n-1
            std_AUC = np.std(global_AUCs)
            std_AUC_ddof_1 = np.std(global_AUCs, ddof=1)

            # 将结果转移到全局变量，因为global_filenames，global_reallabels在下一个循环时会被清空
            G_file_names = global_filenames
            G_real_lables = global_reallabels

            # 打印并保存多折交叉验证的平均结果
            # save_report_txt函数里自带了print效果，调用save_report_txt保存信息的同时也会打印到屏幕上。后续如果节约时间可以把函数里的print注释掉
            # str1 = 'Mean ACC of all folds is :【' + str(mean_ACC) + '】'
            # str2 = 'Mean AUC of all folds is :【' + str(mean_AUC) + '】'
            #
            # str10 = 'All ACC:' + str(global_ACCs)
            # str11 = 'ACC Standard Deviation(devide n) of all folds is :【' + str(std_ACC) + '】'
            # str12 = 'ACC Standard Deviation(devide n-1) of all folds is :【' + str(std_ACC_ddof_1) + '】'
            #
            # str20 = 'All AUC:' + str(global_AUCs)
            # str21 = 'AUC Standard Deviation(devide n) of all folds is :【' + str(std_AUC) + '】'
            # str22 = 'AUC Standard Deviation(devide n-1) of all folds is :【' + str(std_AUC_ddof_1) + '】'
            #
            # report_txt_name = 'Report_Summary_' + conf.classify_name + '_axis_' + axis + '.txt'
            #
            # str_currtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # str3 = '***** Classify model:' + conf.classify_name + '_axis_' + axis + ' ,Report time:' + str_currtime + '***Summary*****'
            # str4 = '*********************************************************************************'

            # save_report_txt(conf.report_dir, report_txt_name, str3, trunc=True)
            # for i in range(len(global_ACC_reports)):
            #     save_report_txt(conf.report_dir, report_txt_name, global_ACC_reports[i])
            # save_report_txt(conf.report_dir, report_txt_name, str10)
            # save_report_txt(conf.report_dir, report_txt_name, str1)
            # save_report_txt(conf.report_dir, report_txt_name, str11)
            # save_report_txt(conf.report_dir, report_txt_name, str12)
            # save_report_txt(conf.report_dir, report_txt_name, str4)
            #
            # for i in range(len(global_AUC_reports)):
            #     save_report_txt(conf.report_dir, report_txt_name, global_AUC_reports[i])
            # save_report_txt(conf.report_dir, report_txt_name, str20)
            # save_report_txt(conf.report_dir, report_txt_name, str2)
            # save_report_txt(conf.report_dir, report_txt_name, str21)
            # save_report_txt(conf.report_dir, report_txt_name, str22)
            # save_report_txt(conf.report_dir, report_txt_name, str4)

    save_filename = 'valid_xyz_porb_result{}_{}_cross'.format(conf.classify_name,cv)
    save_filename = os.path.join(t_dir,save_filename)
    np.save(save_filename,G_select_slices)
    print(save_filename + ' save success')
    
    Mean_ACC,Mean_AUC=merge_different_axises_result_and_return(conf, G_real_lables, G_pred_voted, G_pred_prob_voted)
    # print('Test end')
    return Mean_ACC,Mean_AUC

if __name__ == '__main__':
    tf.app.run()



