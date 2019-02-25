import tensorflow as tf
a = tf.constant(3)
b = tf.constant(4)
with tf.Session() as sess:
    print("相加：%i" % sess.run(a+b))
    print("相乘：%i" % sess.run(a*b))