import tensorflow as tf
import os
from PIL import Image
import random

TFRECORD_SIZE = 10000
NET_INPUT_SIZE = 227
def write_to_tfrecord(train_data, start, end, tf_file):
    #writer = tf.python_io.TFRecordWriter(tf_file)
    with tf.python_io.TFRecordWriter(tf_file) as writer:
        for num in range(start, end):
            img_name = train_data[0][num]
            label = train_data[1][num]
            img_path = r'./train/'+img_name
            
            # image_raw_data = tf.gfile.FastGFile(img_path, 'r').read() 
            # img = tf.decode_raw(image_raw_data, tf.uint8)
            # img = tf.cast(img, tf.float32)
            # tf.image.resize_images(image, (NET_INPUT_SIZE, NET_INPUT_SIZE), method=0)
            # img = tf.cast(img, tf.uint8)
            # img = tf.cast(img, tf.string)
            
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize((NET_INPUT_SIZE, NET_INPUT_SIZE))
            img_raw = img.tobytes() 
            
            tf.cast(label, tf.int64)
            
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    print(tf_file+" done")
    
    
    
def makelabel(train_path):
    
    classes = {'cat':0, 'dog':1}
    imgname = []
    label = []
    for img_name in os.listdir(train_path):
        label_name = img_name.split('.')[0]
        label.append(classes[label_name])
        imgname.append(img_name)
    
    return imgname, label


def shuffle(count_img, train_data):
    index = [i for i in range(count_img)] 
    random.shuffle(index)
    data = []
    label = []
    for _ in index:
        data.append(train_data[0][_])
        label.append(train_data[1][_])
    return data, label
    
if __name__ == '__main__':

    tfrecord_path = r'./tfrecord/'
    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)
    train_path = r'./train'
    train_data = makelabel(train_path)
    
    count_img = len(train_data[1])
    shuffled_data = shuffle(count_img, train_data)
    
    tfrecord_file_num = int(count_img/TFRECORD_SIZE)
    if count_img % TFRECORD_SIZE != 0:
        tfrecord_file_num += 1
    
    for i in range(tfrecord_file_num):
        start = i*TFRECORD_SIZE
        end = start + TFRECORD_SIZE
        if i == tfrecord_file_num-1:
            end = count_img
        tf_file_name = r'train_'+str(start)+'-'+str(end)+'.tfrecord'
        write_to_tfrecord(shuffled_data, start, end, tfrecord_path+tf_file_name)