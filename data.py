import tensorflow as tf
import os

class DATA(object):
    
    def __init__(self, train_path, batch_size, num_epochs):

        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = num_epochs
        self.img, self.label = self.read_tfrecord(train_path)
        self.data = self.to_batch(self.img, self.label)
        
    def to_batch(self, image, label):
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=self.BATCH_SIZE,
            #num_threads=20,
            #allow_smaller_final_batch=True,
            capacity=1000+3*self.BATCH_SIZE,
            min_after_dequeue=1000)
            
        return image_batch, label_batch
    
    def read_tfrecord(self, train_path):
        #dataset = tf.data.TFRecordDataset(filelist)
        filelist = os.listdir(train_path)
        file_path = []
        for i in range(len(filelist)):
            file_path.append('./tfrecord/' + filelist[i])
        filename_deque = tf.train.string_input_producer(file_path, num_epochs=self.NUM_EPOCHS, shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_deque)
        features = tf.parse_single_example(serialized_example, features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)})
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [227, 227, 3])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(features['label'], tf.uint8)
        label = tf.one_hot(label, 2)
    
        return img, label
