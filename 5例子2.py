import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
print('x_data',x_data)
y_data = x_data*0.1 + 0.3
print('y_data',y_data)
### create tensorflow structure start ###


Weights = tf.Variable(tf.random_uniform([1], -10000.0, 10000.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)          # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
