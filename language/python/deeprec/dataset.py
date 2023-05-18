import tensorflow as tf
import numpy as np


x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])

data = tf.data.Dataset.from_tensor_slices({"feature": x,
                                               "label": y})


a = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
b = a.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(5),cycle_length=3, block_length=4)
b = b.batch(3)
c=tf.data.make_one_shot_iterator(b)
d = c.get_next()
with tf.Session() as sess:
    try:
        while(True):
            print(sess.run(d))
            d = c.get_next()
            print("---------")
    except tf.errors.OutOfRangeError:
        print("read down")