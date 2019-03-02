import tensorflow as tf
with tf.variable_scope('test1', ):
    var1 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)

with tf.variable_scope('test2', ):
    var2 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)

print("var1:", var1.name)
print("var2:", var2.name)