""" Глава 1. Распознавание рукописных символов """

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# загружаем набор тестовых данных базы MNIST
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

image_size = 784  # image size = 28 * 28
relu_neurons = 100

# заглушки
x = tf.placeholder(tf.float32, [None, image_size])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_probability = tf.placeholder(tf.float32)

# оптимизируемые переменные
# слой logit
w = tf.Variable(tf.zeros([relu_neurons, 10]))
b = tf.Variable(tf.zeros([10]))
# слой ReLU
w_relu = tf.Variable(tf.truncated_normal([image_size, relu_neurons], stddev=0.1))
b_relu = tf.Variable(tf.truncated_normal([relu_neurons], stddev=0.1))

# слой ReLU
h = tf.nn.relu(tf.matmul(x, w_relu) + b_relu)
# слой dropout
h_drop = tf.nn.dropout(h, keep_probability)
# слой logit и выходной слой с SoftMax-нормализацией
y = tf.nn.softmax(tf.matmul(h_drop, w) + b)

# кросс-энтропия как функция потерь
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# градиентный спуск как метода оптимизации
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

# инициализируем переменные и запускаем сессию
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

# запускаем обучение
for i in range(1000):
    x_batch, y_batch = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: x_batch, y_: y_batch, keep_probability: 0.5})

# считаем точность
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# выводим точность
print("Точность: %s" % session.run(accuracy, feed_dict={x: mnist.test.images,
                                                        y_: mnist.test.labels,
                                                        keep_probability: 0.5}))

# закрываем сессию
session.close()
