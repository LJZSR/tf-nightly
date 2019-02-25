import tensorflow as tf
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    #计算具体数值
    print("相加：%i" % sess.run(add, feed_dict={a: 3, b: 4}))
    print("相乘：%i" % sess.run(mul, feed_dict={a: 3, b: 4}))
