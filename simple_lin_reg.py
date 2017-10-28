import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

adv = pd.read_csv('Advertising.csv')
tv_budget_x = adv.TV.tolist()
sales_y = adv.Sales.tolist()

# plt.scatter(tv_budget_x, sales_y)
# plt.show()
y_tensor = tf.convert_to_tensor(sales_y, dtype=np.float32, name="Y_INPUT")
x_tensor = tf.convert_to_tensor(tv_budget_x, dtype=np.float32, name="X_INPUT")

with tf.name_scope("MeanCalculation"):
	x_mean = tf.reduce_mean(x_tensor, name="X_MEAN")
	y_mean = tf.reduce_mean(y_tensor, name="Y_MEAN")

with tf.name_scope("Numerator"):
	def gen_term(tensor, mean):
		return tf.reduce_sum([tensor, tf.fill(tensor.shape, -mean)], 0)
	x_term = gen_term(x_tensor, x_mean)
	y_term = gen_term(y_tensor, y_mean)
	numerator = tf.reduce_sum(tf.reduce_prod([x_term, y_term], 0), name="NUMERATOR")

with tf.name_scope("Denominator"):
	x_term = gen_term(x_tensor, x_mean)
	denominator = tf.reduce_sum(tf.pow(x_term, 2), name="DENOMINATOR")

with tf.name_scope("Slope"):
	slope = numerator / denominator

with tf.name_scope("Intercept"):
	intercept = y_mean - (slope * x_mean)

with tf.Session() as sess:
	# sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(numerator))
