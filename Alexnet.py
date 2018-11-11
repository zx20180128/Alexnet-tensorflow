import tensorflow as tf
import numpy as np
import datetime

class Alexnet(object):
    """ Alexnet """
    
    def __init__(self, x, class_num, dropout_keep_prob=0.5, model_path=''):
        super (Alexnet, self).__init__()
        self.X = x
        self.CLASS_NUM = class_num
        self.DROPOUT = dropout_keep_prob # train = 0.5 || test = 1
        self.MODEL_PATH = model_path
        
        self.create()

    def create(self):
    
        """Create the network graph."""
        start_time = datetime.datetime.now()
        
        #layer_1: 227*227*3 *batches input
        conv_1 = self.conv(self.X, 11, 96, 4, 'VALID', 'conv_1', 1)
        lrn_1 = self.lrn(conv_1, 1e-4, 0.75, 5, 2.0, 'lrn_1') # 2e-05 0.75 2 1.0
        pool_1 = self.max_pool(lrn_1, 3, 2, 'VALID', 'pool_1')
        #layer_2: 27*27*96 *batches input
        conv_2 = self.conv(pool_1, 5, 256, 1, 'SAME', 'conv_2', 2)
        lrn_2 = self.lrn(conv_2, 1e-4, 0.75, 5, 2.0, 'lrn_2')
        pool_2 = self.max_pool(lrn_2, 3, 2, 'VALID', 'pool_2')
        #layer_3: 13*13*256 *batches input
        conv_3 = self.conv(pool_2, 3, 384, 1, 'SAME', 'conv_3', 1)
        #layer_4: 13*13*384 *batches input
        conv_4 = self.conv(conv_3, 3, 384, 1, 'SAME', 'conv_4', 2)
        #layer_5: 13*13*384 *batches input
        conv_5 = self.conv(conv_4, 3, 256, 1, 'SAME', 'conv_5', 2)
        pool_5 = self.max_pool(conv_5, 3, 2, 'VALID', 'pool_5')
        #layer_6: 6*6*256 *batches input 
        layer_to_line = tf.reshape(pool_5, [-1, 6*6*256])
        fc_6 = self.fc(layer_to_line, 6*6*256, 4096, 'fc_6', True)
        dropout_6 = self.dropout(fc_6, self.DROPOUT, 'dropout_6')
        #layer_7: 4096 to 4096
        fc_7 = self.fc(dropout_6, 4096, 4096, 'fc_7', True)
        dropout_7 = self.dropout(fc_7, self.DROPOUT, 'dropout_7')
        #layer_8: 4096 to class_num
        self.fc_8 = self.fc(dropout_7, 4096, self.CLASS_NUM, 'fc_8', False)
        #class_prob = tf.nn.softmax(fc_8, name="class_prob") #in cross-entropy loss
        time = (datetime.datetime.now() - start_time)
        print("build model finished: {}s".format(time))
        
    def conv(self, x, kernel_size, ker_num, stride, padding, name, groups):
        channel_num = int(int(x.get_shape()[-1])/groups)
        # Create tf variables for the weights and biases of the conv layer
        with tf.name_scope(name) as scope:
            w = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, channel_num, ker_num], 
                                                dtype=tf.float32, stddev=1e-2, mean=0.0),
                            name='kernel', trainable=True)
            b = tf.Variable(tf.constant(0.1, shape=[ker_num], dtype=tf.float32),
                            name='biases', trainable=True)
        
        # Separate into groups
        x_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        w_groups = tf.split(axis=3, num_or_size_splits=groups, value=w)
        o_groups = []
        for i, j in zip(x_groups, w_groups):
            o_groups += [tf.nn.conv2d(i, j, [1, stride, stride, 1], padding=padding)]
        conv = tf.concat(axis=3, values=o_groups)
        bias = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias)
        # y=relu(wx+b)
        return relu
    
    def lrn(self, x, alpha, beta, depth_radius, bias, name):
        return tf.nn.local_response_normalization(x,
                                           alpha=alpha,
                                           beta=beta,
                                           depth_radius=depth_radius,
                                           bias=bias,
                                           name=name)

    def max_pool(self, x, ker_size, strides, padding, name):
        return tf.nn.max_pool(x, 
                              ksize=[1, ker_size, ker_size, 1],
                              strides=[1, strides, strides, 1],
                              padding=padding, 
                              name=name)

    def fc(self, x, i_num, o_num, name, relu):
        with tf.name_scope(name) as scope:
            w = tf.Variable(tf.truncated_normal([i_num, o_num], dtype=tf.float32, stddev=1e-2, mean=0.0),
                                name='weights', 
                                trainable=True)
            b = tf.Variable(tf.constant(0.1, shape=[o_num], dtype=tf.float32),
                                name='biases', 
                                trainable=True)
        out = tf.nn.xw_plus_b(x, w, b)
        if relu:
            relu = tf.nn.relu(out)
            return relu
        else:
            return out
        
    def dropout(self, x, keep_prob, name):
        return tf.nn.dropout(x, keep_prob, name=name)

    def loadModel(self, sess, checkfile):
        saver = tf.train.Saver()
        saver.restore(sess, checkfile)

        # wDict = np.load(self.MODELPATH, encoding = "bytes").item()
        # for name in wDict:
            # if name not in self.SKIP:
                # with tf.variable_scope(name, reuse = True):
                    # for p in wDict[name]:
                        # if len(p.shape) == 1:
                            # sess.run(tf.get_variable('b', trainable = False).assign(p))
                        # else:
                            # sess.run(tf.get_variable('w', trainable = False).assign(p))
