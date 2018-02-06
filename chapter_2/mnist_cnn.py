""" Глава 2. Сверточная нейронная сеть для распознавания рукописных символов """

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Загружаем данные сета MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Заглушки для тренировочных данных
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# приводим изображение к квадрату 28*28 пикселей с одним каналом
# -1 показывает, что рамер минибатча неизвестен заранее
x_image = tf.reshape(x, [-1, 28, 28, 1])

# первый сверточный слой
# веса сверточных слоев
window_1 = 5  # размер окна свертки по стороне
filters_1 = 32  # количество фильтров
# веса сверток
w_conv_1 = tf.Variable(tf.truncated_normal([window_1, window_1, 1, filters_1], stddev=0.1))
# свободные членв
b_conv_1 = tf.Variable(tf.constant(0.1, shape=[filters_1]))

# применим фильтры и свободные веса ко входному изображению
conv_1 = tf.nn.conv2d(x_image, w_conv_1, strides=[1, 1, 1, 1], padding="SAME") + b_conv_1
# нелинейная функция активации (ReLU)
h_conv_1 = tf.nn.relu(conv_1)
# слой субдискретизации уменьшит размер изображений вдвое по каждой стороне
h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# второй сверточный слой (все аналогично первому)
window_2 = 5  # размер окна свертки по стороне
filters_2 = 64  # количество фильтров
w_conv_2 = tf.Variable(tf.truncated_normal([window_2, window_2, filters_1, filters_2], stddev=0.1))
b_conv_2 = tf.Variable(tf.constant(0.1, shape=[filters_2]))
conv_2 = tf.nn.conv2d(h_pool_1, w_conv_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv_2
h_conv_2 = tf.nn.relu(conv_2)
h_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# развернем двумерную картинку в вектор 7 = 64 / 2 / 2 (два слоя субдискретизации)
h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])

# обычная нейроная сеть (1024 нейрона) связывается 
# полным графом с выходами второго сверточного слоя
w_fc_1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, w_fc_1) + b_fc_1)

# слой регуляризации Dropout'ом
keep_probability = tf.placeholder(tf.float32)
h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_probability)

# последний слой с 10 выходами и softmax
w_fc_2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc_2 = tf.Variable(tf.constant(0.1, shape=[10]))
logit_conv = tf.matmul(h_fc_1_drop, w_fc_2) + b_fc_2
y_conv = tf.nn.softmax(logit_conv)

# кросс-энтропия как функция потерь
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_conv, labels=y))
# Adam как оптимизирующий алгоритм
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# вычислим точность
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# обучение
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(64)
    session.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_probability: 0.5})

# выводим результат
print(session.run(accuracy, {x: mnist.test.images, y: mnist.test.labels, keep_probability: 1}))
