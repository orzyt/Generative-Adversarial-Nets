import os
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from load_svhn import load_svhn
from load_celeba import download_celeb_a
from tensorflow.examples.tutorials.mnist import input_data


##################################################################
dataset_name = 'svhn'  # [mnist, celeba, svhn]
##################################################################
batch_size = 64
z_dim = 128
learning_rate_c = 1e-4
learning_rate_g = 1e-4
image_width = 32
image_height = 32
ndf = 32
ngf = 32
Citers = 5
lam = 10
beta1 = 0.5
beta2 = 0.9
max_iter_step = 20000
channels = 1 if dataset_name == 'mnist' else 3
log_path = './' + dataset_name + '/log_wgan'
ckpt_path = './' + dataset_name + '/ckpt_wgan'
ckpt_step_path = ckpt_path + '.step'
##################################################################
if dataset_name == 'mnist':
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    ndf = ngf = 16
elif dataset_name == 'celeba':
    download_celeb_a()
    dataset = glob(os.path.join('./data/', 'celebA/*.jpg'))
    count = len(dataset)
elif dataset_name == 'svhn':
    dataset = load_svhn()
##################################################################


def get_image(image_path):
    image = Image.open(image_path)
    if image.size != (image_width, image_height):
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([image_width, image_height], Image.BILINEAR)
    return np.array(image.convert('RGB'))


def get_batches():
    if dataset_name == 'mnist':
        batch_images = dataset.train.next_batch(batch_size)[0]
        batch_images = 2 * batch_images - 1
        batch_images = np.reshape(batch_images, (-1, 28, 28))
        batch_images = np.pad(batch_images, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=-1)
        batch_images = np.expand_dims(batch_images, -1)
    elif dataset_name == 'celeba':
        index = np.random.choice(count, batch_size, replace=False)
        batch_images = np.array([get_image(dataset[idx]) for idx in index]).astype(np.float32)
        batch_images = 2 * batch_images / 255 - 1
    elif dataset_name == 'svhn':
        batch_images = dataset.train.next_batch(batch_size)[0]
        batch_images = 2 * batch_images - 1

    return batch_images


def leaky_relu(x, leak=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def generator(z, channels, training=True):
    with tf.variable_scope("generator", reuse=(not training)):

        x = tf.layers.dense(z, 4 * 4 * ngf * 8)

        deconv1 = tf.reshape(x, (-1, 4, 4, ngf * 8))
        bn1 = tf.layers.batch_normalization(deconv1, training=training)
        relu1 = tf.nn.relu(bn1)

        deconv2 = tf.layers.conv2d_transpose(relu1, ngf * 4, 3, strides=2, padding='SAME',
                                             kernel_initializer=tf.random_normal_initializer(0, 0.02))
        bn2 = tf.layers.batch_normalization(deconv2, training=training)
        relu2 = tf.nn.relu(bn2)

        deconv3 = tf.layers.conv2d_transpose(relu2, ngf * 2, 3, strides=2, padding='SAME',
                                             kernel_initializer=tf.random_normal_initializer(0, 0.02))
        bn3 = tf.layers.batch_normalization(deconv3, training=training)
        relu3 = tf.nn.relu(bn3)

        deconv4 = tf.layers.conv2d_transpose(relu3, ngf, 3, strides=2, padding='SAME',
                                             kernel_initializer=tf.random_normal_initializer(0, 0.02))
        bn4 = tf.layers.batch_normalization(deconv4, training=training)
        relu4 = tf.nn.relu(bn4)

        deconv5 = tf.layers.conv2d_transpose(relu4, channels, 3, strides=1,
                                             padding='SAME', kernel_initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.tanh(deconv5)

        return out


def critic(image, reuse=False):
    with tf.variable_scope('critic', reuse=reuse):
        conv1 = tf.layers.conv2d(image, ndf, 3, strides=2, padding='SAME')
        lrelu1 = leaky_relu(conv1)

        conv2 = tf.layers.conv2d(lrelu1, ndf * 2, 3, strides=2, padding='SAME')
        lrelu2 = leaky_relu(conv2)

        conv3 = tf.layers.conv2d(lrelu2, ndf * 4, 3, strides=2, padding='SAME')
        lrelu3 = leaky_relu(conv3)

        conv4 = tf.layers.conv2d(lrelu3, ndf * 8, 3, strides=1, padding='SAME')
        lrelu4 = leaky_relu(conv4)

        flat = tf.reshape(lrelu4, [batch_size, 4 * 4 * ndf * 8])

        logits = tf.layers.dense(flat, 1)

        return logits


def model_inputs():
    inputs_real = tf.placeholder(tf.float32, shape=(None, image_width, image_height, channels))
    inputs_z = tf.placeholder(tf.float32, (None, z_dim))

    return inputs_real, inputs_z


def model_loss(input_real, input_z):
    g = generator(input_z, channels)
    c_real = critic(input_real)
    c_fake = critic(g, reuse=True)

    c_loss = tf.reduce_mean(c_fake - c_real)
    g_loss = tf.reduce_mean(-c_fake)

    alpha_dist = tf.distributions.Uniform()
    alpha = alpha_dist.sample((batch_size, 1, 1, 1))
    interpolated = input_real + alpha * (g - input_real)
    inte_loss = critic(interpolated, reuse=True)
    gradients = tf.gradients(inte_loss, [interpolated, ])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)

    c_loss += lam * gradient_penalty

    return c_loss, g_loss, gradients


def model_opt(c_loss, g_loss):
    c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        c_opt = tf.train.AdamOptimizer(learning_rate_c, beta1, beta2).minimize(c_loss, var_list=c_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate_g, beta1, beta2).minimize(g_loss, var_list=g_vars)

    return c_opt, g_opt


def main(mode='train'):
    input_real, input_z = model_inputs()
    c_loss, g_loss, gradients = model_loss(input_real, input_z)
    c_opt, g_opt = model_opt(c_loss, g_loss)

    c_loss_sum = tf.summary.scalar("c_loss", c_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    grad = tf.summary.scalar("grad_norm", tf.sqrt(tf.nn.l2_loss(gradients) * 2 / batch_size))
    img_sum = tf.summary.image("images", generator(input_z, channels, False))
    merged_all = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(log_path, tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        if mode == 'train':
            if os.path.isfile(ckpt_step_path):
                with open(ckpt_step_path, 'rb') as f:
                    start_step = int(f.read())
                print('Training was interrupted. Continuing at step', start_step)
                saver.restore(sess, ckpt_path)
            else:
                start_step = 0
                sess.run(init)

            for step in range(start_step, max_iter_step):
                citers = 100 if (step < 25 or step % 500 == 0) else Citers
                for i in range(citers):
                    x = get_batches()
                    z = np.random.normal(0.0, 1.0, size=(batch_size, z_dim))
                    sess.run(c_opt, feed_dict={input_real: x, input_z: z})

                x = get_batches()
                z = np.random.normal(0.0, 1.0, size=(batch_size, z_dim))
                sess.run(g_opt, feed_dict={input_real: x, input_z: z})

                if step % 50 == 0:
                    c_loss_val, g_loss_val, merged_summary = sess.run(
                        [c_loss, g_loss, merged_all], feed_dict={input_real: x, input_z: z})
                    print('step: %d c_loss: %f, g_loss: %f' % (step, c_loss_val, g_loss_val))
                    summary_writer.add_summary(merged_summary, step)

                if step % 500 == 0:
                    saver.save(sess, ckpt_path)
                    print('Save model at step', step)
                    with open(ckpt_step_path, 'wb') as f:
                        f.write(b'%d' % (step + 1))

            os.remove(ckpt_step_path)

        elif mode == 'test':
            saver.restore(sess, ckpt_path)
            g = generator(input_z, channels, training=False)

            for idx in range(3):
                z = np.random.normal(0.0, 1.0, size=(batch_size, z_dim))
                gen_images = g.eval(feed_dict={input_z: z})

                overall = []
                for i in range(8):
                    temp = []
                    for j in range(8):
                        temp.append(gen_images[i * 8 + j])
                    overall.append(np.concatenate(temp, axis=1))

                res = np.concatenate(overall, axis=0)
                res = np.squeeze(res)
                plt.figure(figsize=[8, 8])
                plt.axis('off')
                if dataset_name == 'mnist':
                    res = 1 - (res + 1) / 2
                    plt.imshow(res, cmap='binary')
                else:
                    res = (res + 1) / 2
                    plt.imshow(res)
                plt.tight_layout()
                save_path = './%s/result/' % (dataset_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(save_path + '%d.png' % idx)
                print('Saving %d.png' % idx)


if __name__ == '__main__':
    main(mode='test')
