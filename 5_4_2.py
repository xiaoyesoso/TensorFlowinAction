import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result = v1 + v2

saver = tf.train.Saver()

saver.export_meta_graph("/home/soso/PycharmProjects/TensorFlowinAction/meta_graph.json",as_text=True)
