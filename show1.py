import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
print('Finished imports')

admits = pd.read_csv('admit.csv')
gre_x = np.array(admits.gre.tolist())
admit_y = np.array(admits.admit.tolist())

# plt.grid(True)

# for (gre, admit) in zip(gre_x, admit_y):
# 	plt.plot(gre, admit, '.', color='orange')

# plt.ylim(-0.5, 1.5)
# plt.show()

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(0.0, tf.float32)
b = tf.Variable(0.0, tf.float32)

a = 1/1 + tf.exp(-tf.add(tf.multiply(x, W), b))

cross_entropy = tf.reduce_mean(-(y * tf.log(a) + (1 - y) * tf.log(1 - a)))
cross_entropy = tf.Print(cross_entropy, [a])
train = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(1000):
	idx = np.random.choice(len(admit_y), 20, replace=False)
	_, l = sess.run([train, cross_entropy], feed_dict={x:gre_x[idx], y:admit_y[idx]})
	if epoch % 100 == 0:
		print (l)
