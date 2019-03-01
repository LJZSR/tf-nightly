import tensorflow as tf

var1 = tf.Variable(1.0, name='firstvar')
print('var1:', var1.name)
var1 = tf.Variable(2.0, name='firstvar')
print('var1:', var1.name)
var2 = tf.Variable(3.0)
print('var2:', var2.name)
var2 = tf.Variable(4.0)
print('var2:', var2.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('var1=', var1.eval())
    print('var2=', var2.eval())

