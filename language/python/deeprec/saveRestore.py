import tensorflow as tf
  
var = tf.get_embedding_variable("var_0",
                                embedding_dim=5,
#                                initializer=tf.ones_initializer(tf.float32),
                                partitioner=tf.fixed_size_partitioner(num_shards=4))


var2 = tf.get_embedding_variable("var_23",
                                embedding_dim=3,
                                partitioner=tf.fixed_size_partitioner(num_shards=4))

shape = [var1.total_count() for var1 in var]

emb = tf.nn.embedding_lookup(var, tf.cast([0,1,2,5,6,7], tf.int64))
emb2 = tf.nn.embedding_lookup(var2, tf.cast([0,1,2,5,6,7], tf.int64))
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.AdagradOptimizer(0.1)

g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)

init = tf.global_variables_initializer()

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#saver = tf.train.Saver([var], max_to_keep=2)
saver2 = tf.train.Saver( max_to_keep=2)
#下面注释的为保持ev的逻辑
with tf.Session(config=sess_config) as sess: 
#  sess.run([init])
  print('----------------------------')
#  print(sess.run([emb]))
#  print(sess.run([emb2]))
#  saver2.save(sess, "Saved_model/test.ckpt", global_step=0) //restore前必须sess.run([emb])
  saver2.restore(sess, "Saved_model/test.ckpt-0")

  print('----------------------------')
  print(sess.run([emb]))
  print(sess.run([emb2]))