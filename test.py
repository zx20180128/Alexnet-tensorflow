import tensorflow as tf
from Alexnet import Alexnet
import datetime
import os
import argparse
import sys
from PIL import Image
import numpy as np

FLAGS = None


def test():
    num_classes=2
    classes = {0:'cat', 1:'dog'}
    modelname = 'checkpoint/model_{:03d}.ckpt'.format(FLAGS.model)
    # load network
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    network = Alexnet(x, num_classes, 1)
    with tf.Session() as sess:
        network.loadModel(sess, modelname)
        for name in os.listdir(FLAGS.data):
            path = FLAGS.data + name
            img = Image.open(path).convert('RGB')
            img = img.resize((227, 227))
            img_input = np.asarray(img) * (1. / 255) - 0.5
            result = sess.run(network.fc_8, feed_dict={x:[img_input]})
            classname = classes[np.argmax(result,1)[0]]
            if not os.path.exists('result'):
                os.mkdir('result')
            imgname_new = classname+'.'+name
            img.save('./result/'+imgname_new)
            print(name,"    ", result, "    ",classname)

def main(_):
    test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type = int,
        default = 0,
        help = 'Load the model, input epoch num.'
    )    
    parser.add_argument(
        '--data',
        type = str,
        default = './test/',
        help = 'test img.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
