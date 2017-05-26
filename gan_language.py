from __future__ import print_function
from builtins import next
from builtins import range
import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import argparse

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!

ITERS = 200000 # How many iterations to train for
CRITIC_ITERS = 10 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).
parser = argparse.ArgumentParser(description="gan_language.py")
parser.add_argument('-model_dir', required=True,
                    help='path to save the models.')
parser.add_argument('-data_dir', required=True,
                    help='data path.')
parser.add_argument('-dim', type=int,
                    help='model dimensionality.')
parser.add_argument('-batch_size', type=int,
                    help='batch size.')
parser.add_argument('-seq_len', type=int,
                    help='sequence length.')
parser.add_argument('-vocab_size', type=int,
                    help='vocabulary size.')
opt = parser.parse_args()
DIM = opt.dim if opt.dim else 64 
# Model dimensionality. This is fairly slow and overfits, even on
# Billion Word. Consider decreasing for smaller datasets.
BATCH_SIZE = opt.batch_size if opt.batch_size else 64 # Batch size
SEQ_LEN = opt.seq_len if opt.seq_len else 20 # Sequence length in characters
VOCAB_SIZE = opt.vocab_size if opt.vocab_size else 6000 #  Vocabulary size


lib.print_model_settings(locals().copy())

lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    max_vocab_size=VOCAB_SIZE,
    data_dir=opt.data_dir
)

def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random_normal(shape)

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    return output

def Discriminator(inputs):
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', SEQ_LEN*DIM, 1, output)
    return output

real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

disc_real = Discriminator(real_inputs) 
disc_fake = Discriminator(fake_inputs)

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1,1], 
    minval=0.,
    maxval=1.
)
differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

lr = tf.Variable(1e-4, trainable=False)
lr_decay_op = tf.assign(lr, lr * 0.9)

gen_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]], 
                dtype='int32'
            )

# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(4)]
for i in range(4):
    print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]

global_step = tf.Variable(0, trainable=False)
global_step_increment = global_step.assign(global_step + 1)
saver = tf.train.Saver()

with tf.Session() as session:
    ckpt = tf.train.get_checkpoint_state(opt.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())


    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    gen = inf_train_gen()

    js1_ema = [0]
    ema_alpha = 0.5
    while global_step.eval() < ITERS:
        start_time = time.time()

        # Train generator
        if global_step.eval() > 0:
            _ = session.run(gen_train_op)

        # Train critic
        for i in range(CRITIC_ITERS):
            _data = next(gen)
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete:_data}
            )

        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)

        if global_step.eval() % 100 == 99:
            saver.save(session, os.path.join(opt.model_dir, "model"), global_step=global_step)
            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            for i in range(4):
                lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
                js = lm.js_with(true_char_ngram_lms[i])
                if i == 0:
                    js1_ema.append(ema_alpha * js + (1 - ema_alpha) * js1_ema[-1])
                    print("js1_ema: {0}".format(js1_ema[-1]))
                lib.plot.plot(os.path.join(opt.model_dir, 'js{}'.format(i+1)), js)
            if global_step.eval() >= 299 and js1_ema[-1] > js1_ema[-2] and js1_ema[-2] > js1_ema[-3]:
                session.run(lr_decay_op)
                print("no improvement seen, decay learning rate to {0}".format(lr.eval()))



            with open(os.path.join(opt.model_dir, 'samples_{}.txt'.format(global_step.eval())), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")

        if global_step.eval() % 100 == 99:
            lib.plot.flush()
        session.run(global_step_increment)
        
        lib.plot.tick()
