import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


##################################################################
batch_size = 64
z_dim = 100
learning_rate_d = 0.0002
learning_rate_g = 0.0002
image_width = 32
image_height = 32
ndf = 16
ngf = 16
beta1 = 0.5
beta2 = 0.9
max_iter_step = 20000
channels = 1
log_path = './log_cgan'
ckpt_path = './ckpt_cgan'
ckpt_step_path = ckpt_path + '.step'
dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
##################################################################


def get_batches():
    X, y = dataset.train.next_batch(batch_size)
    X = 2 * X - 1
    X = np.reshape(X, (-1, 28, 28))
    X = np.pad(X, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=-1)
    X = np.expand_dims(X, -1)

    return X, y


def leaky_relu(x, leak=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def generator(z, y, channels, training=True):
    with tf.variable_scope("generator", reuse=(not training)):
        # y_ = tf.reshape(y, shape=[batch_size, 1, 1, 10])
        z = tf.concat([z, y], axis=1)
        x = tf.layers.dense(z, 4 * 4 * ngf * 8)

        deconv1 = tf.reshape(x, (-1, 4, 4, ngf * 8))
        bn1 = tf.layers.batch_normalization(deconv1, training=training)
        relu1 = tf.nn.relu(bn1)
        # out1 = conv_cond_concat(relu1, y_)

        deconv2 = tf.layers.conv2d_transpose(relu1, ngf * 4, 3, strides=2, padding='SAME')
        bn2 = tf.layers.batch_normalization(deconv2, training=training)
        relu2 = tf.nn.relu(bn2)
        # out2 = conv_cond_concat(relu2, y_)

        deconv3 = tf.layers.conv2d_transpose(relu2, ngf * 2, 3, strides=2, padding='SAME')
        bn3 = tf.layers.batch_normalization(deconv3, training=training)
        relu3 = tf.nn.relu(bn3)
        # out3 = conv_cond_concat(relu3, y_)

        deconv4 = tf.layers.conv2d_transpose(relu3, ngf, 3, strides=2, padding='SAME')
        bn4 = tf.layers.batch_normalization(deconv4, training=training)
        relu4 = tf.nn.relu(bn4)
        # out4 = conv_cond_concat(relu4, y_)

        deconv5 = tf.layers.conv2d_transpose(relu4, channels, 3, strides=1, padding='SAME')
        out = tf.nn.tanh(deconv5)

        return out


def discriminator(image, y, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        y_ = tf.reshape(y, shape=(batch_size, 1, 1, 10))
        image = conv_cond_concat(image, y_)

        conv1 = tf.layers.conv2d(image, ndf, 3, strides=2, padding='SAME')
        lrelu1 = leaky_relu(conv1)
        # out1 = conv_cond_concat(lrelu1, y_)

        conv2 = tf.layers.conv2d(lrelu1, ndf * 2, 3, strides=2, padding='SAME')
        bn2 = tf.layers.batch_normalization(conv2, training=True)
        lrelu2 = leaky_relu(bn2)
        # out2 = conv_cond_concat(lrelu2, y_)

        conv3 = tf.layers.conv2d(lrelu2, ndf * 4, 3, strides=2, padding='SAME')
        bn3 = tf.layers.batch_normalization(conv3, training=True)
        lrelu3 = leaky_relu(bn3)
        # out3 = conv_cond_concat(lrelu3, y_)

        conv4 = tf.layers.conv2d(lrelu3, ndf * 8, 3, strides=1, padding='SAME')
        bn4 = tf.layers.batch_normalization(conv4, training=True)
        lrelu4 = leaky_relu(bn4)
        # out4 = conv_cond_concat(lrelu4, y_)

        flat = tf.reshape(lrelu4, [batch_size, -1])

        # flat = tf.concat([flat, y], axis=1)

        logits = tf.layers.dense(flat, 1)

        out = tf.sigmoid(logits)

        return out, logits


def model_inputs():
    inputs_real = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, channels))
    inputs_y = tf.placeholder(tf.float32, shape=(batch_size, 10))
    inputs_z = tf.placeholder(tf.float32, shape=(batch_size, z_dim))

    return inputs_real, inputs_y, inputs_z


def model_loss(input_real, input_y, input_z):
    label_smoothing = 0.9

    g = generator(input_z, input_y, channels)
    d_real, d_logits_real = discriminator(input_real, input_y)
    d_fake, d_logits_fake = discriminator(g, input_y, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_real, labels=tf.ones_like(d_real) * label_smoothing))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_fake, labels=tf.zeros_like(d_fake) * label_smoothing))

    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_fake, labels=tf.ones_like(d_fake) * label_smoothing))

    return d_loss, g_loss


def model_opt(d_loss, g_loss):
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_opt = tf.train.AdamOptimizer(learning_rate_d, beta1, beta2).minimize(d_loss, var_list=d_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate_g, beta1, beta2).minimize(g_loss, var_list=g_vars)

    return d_opt, g_opt


def main():
    input_real, input_y, input_z = model_inputs()
    d_loss, g_loss = model_loss(input_real, input_y, input_z)
    d_opt, g_opt = model_opt(d_loss, g_loss)

    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    merged_all = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(log_path, tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    fixed_z = np.random.normal(0.0, 1.0, size=(batch_size, z_dim))

    with tf.Session(config=config) as sess:
        if os.path.isfile(ckpt_step_path):
            with open(ckpt_step_path, 'rb') as f:
                start_step = int(f.read())
            print('Training was interrupted. Continuing at step', start_step)
            saver.restore(sess, ckpt_path)
        else:
            start_step = 0
            sess.run(init)

        sample_image_id = 0

        for step in range(start_step, max_iter_step):
            x, y = get_batches()
            z = np.random.normal(0.0, 1.0, size=(batch_size, z_dim))
            sess.run(d_opt, feed_dict={input_real: x, input_y: y, input_z: z})
            sess.run(g_opt, feed_dict={input_real: x, input_y: y, input_z: z})

            if step % 50 == 0:
                d_loss_val, g_loss_val, merged_summary = sess.run([d_loss, g_loss, merged_all],
                                                                  feed_dict={input_real: x, input_y: y, input_z: z})
                print('step: %d d_loss: %f, g_loss: %f' % (step, d_loss_val, g_loss_val))
                summary_writer.add_summary(merged_summary, step)

            if (step > 0 and ((step < 500 and step % 25 == 0) or (step % 500 == 0))):
                overall = []
                g = generator(input_z, input_y, channels, training=False)
                for r in range(0, 10):
                    y = np.zeros((batch_size, 10))
                    y[:, r] = 1
                    gen_images = g.eval(feed_dict={input_z: fixed_z, input_y: y})
                    temp = []
                    for c in range(10):
                        temp.append(gen_images[c])
                    overall.append(np.concatenate(temp, axis=1))
                res = np.concatenate(overall, axis=0)
                res = np.squeeze(res)
                plt.figure(figsize=[10, 10])
                plt.axis('off')
                res = 1 - (res + 1) / 2
                plt.imshow(res, cmap='binary')
                plt.tight_layout()
                plt.savefig('./result/%d.png' % sample_image_id, format='png', dpi=51.2)
                print('Saving %d.png' % sample_image_id)
                plt.close('all')
                sample_image_id += 1

            if (step != 0 and step % 500 == 0):
                saver.save(sess, ckpt_path)
                print('Save model at step', step)
                with open(ckpt_step_path, 'wb') as f:
                    f.write(b'%d' % (step + 1))

        os.remove(ckpt_step_path)


if __name__ == '__main__':
    main()
