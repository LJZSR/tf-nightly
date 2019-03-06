import tensorflow as tf

labels = [[0,0,1],[0,1,0]]
logits = [[2,0.5,6],[0.1,0,3]]
logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
result2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_scaled, labels=labels)
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)
result4 = -tf.reduce_sum(labels*tf.log(logits_scaled2),1)

with tf.Session() as sess:
    print('scaled=', sess.run(logits_scaled))
    print('scaled2=', sess.run(logits_scaled2))

    print('rel1=', sess.run(result1))
    print('rel2=', sess.run(result2))
    print('rel3=', sess.run(result3))
    print('rel4=', sess.run(result4))