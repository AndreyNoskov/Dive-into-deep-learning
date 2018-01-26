""" Глава 1. Линейная регрессия """

import numpy as np
import tensorflow as tf

n_samples = 1000
batch_size = 100
num_steps = 20000

# тренировочные данные
x_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2 * x_data + 1 + np.random.normal(0, 2, (n_samples, 1))

# заглушки
x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# оптимизируемые переменные
with tf.variable_scope('linear-regression'):
    k = tf.Variable(tf.random_normal((1, 1), stddev=0.001), name='slope')
    b = tf.Variable(tf.zeros(1,), name='bias')

# предсказанное значние
y_pred = tf.matmul(x, k) + b

# функция потерь
loss = tf.reduce_mean(tf.square(y_pred - y))

# способ оптимизации
optimizer = tf.train.GradientDescentOptimizer(1.0e-2).minimize(loss)

display_step = 200
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(num_steps):
        indices = np.random.choice(n_samples, batch_size)
        x_batch = x_data[indices]
        y_batch = y_data[indices]
        _, loss_val, k_val, b_val = session.run([optimizer, loss, k, b],
                                                feed_dict={x: x_batch, y: y_batch})
        if (i + 1) % display_step == 0:
            print(f'Эпоха {i+1}: loss = {loss_val:.3f}, ' + 
                  f'k = {np.sum(k_val).item():.3f}, b = {np.sum(b_val).item():.3f}')
