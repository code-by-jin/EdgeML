from layers import *
import tensorflow as tf
import numpy as np

mu = 0
sigma = 0.1

class B_VGGNet(object):

    def __init__(self, num_class=10):
        self.num_class = num_class

    def model(self, x, is_train):
        # conv layer 1
        with tf.name_scope("Inference_1"):
            self.conv1 = Conv2d(x, filters=64, k_size=3, stride=1,name='conv1')
            self.conv1 = BN(self.conv1, phase_train=is_train,name='conv1_bn')
            self.conv1 = Relu(self.conv1,name='conv1_relu')
            self.max_pool1 = max_pooling(self.conv1, k_size=2, stride=2,name='conv1_maxpool')

            self.conv2 = Conv2d(self.max_pool1, filters=128, k_size=3, stride=1,name='conv2')
            self.conv2 = BN(self.conv2, phase_train=is_train,name='conv2_bn')
            self.conv2 = Relu(self.conv2,name='conv2_relu')
            self.max_pool2 = max_pooling(self.conv2, k_size=2, stride=2,name='conv2_maxpool')
            #exit`````````````````
            self.conv3 = Conv2d(self.max_pool2, filters=256, k_size=3, stride=1,name='conv3')
            self.conv3 = BN(self.conv3, phase_train=is_train,name='conv3_bn')
            self.conv3 = Relu(self.conv3,name='conv3_relu')
            self.conv4 = Conv2d(self.conv3, filters=256, k_size=3, stride=1,name='conv4')
            self.conv4 = BN(self.conv4, phase_train=is_train,name='conv4_bn')
            self.conv4 = Relu(self.conv4,name='conv4_relu')
            self.max_pool3 = max_pooling(self.conv4, k_size=2, stride=2,name='conv4_maxpool')

            self.conv5 = Conv2d(self.max_pool3, filters=512, k_size=3, stride=1,name='conv5')
            self.conv5 = BN(self.conv5, phase_train=is_train,name='conv5_bn')
            self.conv5 = Relu(self.conv5,name='conv5_relu')
            self.conv6 = Conv2d(self.conv5, filters=512, k_size=3, stride=1,name='conv6')
            self.conv6 = BN(self.conv6, phase_train=is_train,name='conv6_bn')
            self.conv6 = Relu(self.conv6,name='conv6_relu')
            self.max_pool4 = max_pooling(self.conv6, k_size=2, stride=2,name='conv6_maxpool')
            #exit`````````````````
            self.conv7 = Conv2d(self.max_pool4, filters=512, k_size=3, stride=1,name='conv7')
            self.conv7 = BN(self.conv7, phase_train=is_train,name='conv7_bn')
            self.conv7 = Relu(self.conv7,name='conv7_relu')
            self.conv8 = Conv2d(self.conv7, filters=512, k_size=3, stride=1,name='conv8')
            self.conv8 = BN(self.conv8, phase_train=is_train,name='conv8_bn')
            self.conv8 = Relu(self.conv8,name='conv8_relu')

            self.fc1 = Flatten(self.conv8)
            self.fc1 = fc_layer(self.fc1, 4096,name='fc3')
            self.fc1 = Relu(self.fc1,name='fc3_relu')
            self.fc1 = Drop_out(self.fc1, 0.2, training=is_train)
            self.fc2 = fc_layer(self.fc1, 4096,name='fc4')
            self.fc2 = Relu(self.fc2,name='fc4_relu')
            self.fc2 = Drop_out(self.fc2, 0.2, training=is_train)
            logits_exit2 = fc_layer(self.fc2, self.num_class,name='logits_exit2')

        with tf.name_scope("exit0"):
            #tf.reset_default_graph()
            self.exit0 = Conv2d(self.max_pool2, filters=256, k_size=3, stride=2,name='conv11')
            self.exit0 = BN(self.exit0, phase_train=is_train,name='conv1_bn')
            self.exit0 = Relu(self.exit0,name='conv1_relu')
            self.exit0 = Conv2d(self.exit0, filters=256, k_size=3, stride=2,name='conv21')
            self.exit0 = BN(self.exit0, phase_train=is_train,name='conv2_bn')
            self.exit0 = Relu(self.exit0,name='conv2_relu')
            self.exit0 = max_pooling(self.exit0, k_size=2, stride=2,name='conv2_maxpool')
            self.exit0 = Flatten(self.exit0)
            self.exit0 = fc_layer(self.exit0, 4096,name='fc1')
            self.exit0 = Relu(self.exit0,name='fc1_relu')
            self.exit0 = Drop_out(self.exit0, 0.2, training=is_train)
            self.exit0 = fc_layer(self.exit0, 4096,name='fc2')
            self.exit0 = Relu(self.exit0,name='fc2_relu')
            self.exit0 = Drop_out(self.exit0, 0.2, training=is_train)
            logits_exit0 = fc_layer(self.exit0, self.num_class,name='logits_exit0')

        with tf.name_scope("exit1"):
            self.exit1 = Conv2d(self.max_pool4, filters=512, k_size=3, stride=2,name='conv22')
            self.exit1 = BN(self.exit1, phase_train=is_train,name='conv2_bn')
            self.exit1 = Relu(self.exit1,name='conv2_relu')
            self.exit1 = Flatten(self.exit1)
            self.exit1 = fc_layer(self.exit1, 4096,name='fc31')
            self.exit1 = Relu(self.exit1,name='fc3_relu')
            self.exit1 = Drop_out(self.exit1, 0.2, training=is_train)
            self.exit1 = fc_layer(self.exit1, 4096,name='fc41')
            self.exit1 = Relu(self.exit1,name='fc4_relu')
            logits_exit1 = fc_layer(self.exit1, self.num_class,name='logits_exit1')



        return [logits_exit0, logits_exit1, logits_exit2]

