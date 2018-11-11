import tensorflow as tf
import datetime
from Alexnet import Alexnet
from data import DATA
import argparse
import sys
import os

FLAGS = None

def train():
    # load data
    train_path = './tfrecord'
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    # load network
    num_classes=2
    checkfile = 'checkpoint/model_{:03d}.ckpt'.format(FLAGS.checkpoint)
    with tf.Graph().as_default():
        data = DATA(train_path, batch_size, num_epochs)
        img = data.data[0]
        label = data.data[1]
        network = Alexnet(img, 2, 0.5)
        
        learning_rate=0.001
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=network.fc_8, labels=label))
        #cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) 
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        with tf.Session() as sess:
            # init
            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess=sess,coord = coord)
            if FLAGS.checkpoint >=0 and os.path.exists(checkfile):
                network.loadModel(sess, checkfile)
            for epoch in range(FLAGS.checkpoint+1, num_epochs):
                num_batch = 25000//batch_size
                if 25000 % batch_size == 0:
                    num_batch += 1
                for batch in range(num_batch):
                    start_time = datetime.datetime.now()
                    if coord.should_stop():
                        break
                    _, loss_print = sess.run([train,loss])
                    time = (datetime.datetime.now() - start_time)
                    print('[TRAIN] Epoch[{}]({}/{});  Loss: {:.6f};  Backpropagation: {} sec; '.
                        format(epoch, batch + 1, num_batch, loss_print, time))
                if not os.path.exists('checkpoint'):
                    os.mkdir('checkpoint')
                saver = tf.train.Saver()
                saver.save(sess,'checkpoint/model_{:03d}.ckpt'.format(epoch))
            coord.request_stop()
            coord.join(thread)
def main(_):
    train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 512,
        help = 'Batch size.'
    )
    parser.add_argument(
        '--num_epochs',
        type = int,
        default = 500,
        help = 'Number of batches to run.'
    )
    parser.add_argument(
        '--checkpoint',
        type = int,
        default = -1,
        help = 'Load the model, input epoch num.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
