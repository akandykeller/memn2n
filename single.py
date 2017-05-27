"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data, vectorize_data_ordertest
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import copy
import csv

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 2, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 20, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en-10k/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)

# Add time words/indexes
for i in range(memory_size):
    word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position
sentence_size += 1  # +1 for time words

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

sentence_size_w_order = max(query_size, sentence_size) # 

# train/validation/test sets
S, S_lens, Q, Q_lens, A = vectorize_data(train, word_idx, sentence_size, memory_size)
trainS, valS, trainS_lens, valS_lens, trainQ, valQ, trainQ_lens, valQ_lens, trainA, valA = cross_validation.train_test_split(S, S_lens, Q, Q_lens, A, test_size=.1, random_state=FLAGS.random_state)
testS, testS_lens, testQ, testQ_lens, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

print("TestS:", testS[0])
print("TestS_lens:", testS_lens[0])
print("TrainS:", trainS[0])
print("TrainS_lens:", trainS_lens[0])

# Vectorize data for encoder study
S_order, S_order_lens, Q_order, A_order = vectorize_data_ordertest(train, word_idx, sentence_size_w_order)
train_S_order, val_S_order, train_S_order_lens, val_S_order_lens, train_Q_order, val_Q_order, train_A_order, val_A_order = cross_validation.train_test_split(S_order, S_order_lens, Q_order, A_order, test_size=.1, random_state=FLAGS.random_state)
test_S_order, test_S_order_lens, test_Q_order, test_A_order = vectorize_data_ordertest(test, word_idx, sentence_size_w_order)
print("S_order[0]: ", train_S_order[0])
print("Q_order[0]: ", train_Q_order[0])
print("A_order[0]: ", train_A_order[0])
print("S_order shape: ", train_S_order.shape)
print("Percent Positive Train: ", np.mean(train_A_order))
print("Percent Positive Val: ", np.mean(val_A_order))
print("Percent Positive Test: ", np.mean(test_A_order))
n_train_order = train_S_order.shape[0]


print("Test S[0]: ", testS[0])
print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]
# Get list of ordered batches for eval before shuffling
train_eval_batches = copy.copy(batches)

order_batches = zip(range(0, n_train_order-batch_size, batch_size), range(batch_size, n_train_order, batch_size))
order_batches = [(start, end) for start, end in order_batches]

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, 
                   sentence_size_w_order=sentence_size_w_order, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)

    for t in range(1, FLAGS.epochs+1):
        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in tqdm(batches, desc='Epoch {}: '.format(t)):
            s = trainS[start:end]
            s_lens = trainS_lens[start:end]
            q = trainQ[start:end]
            q_lens = trainQ_lens[start:end]
            a = trainA[start:end]

            cost_t = model.batch_fit(s, s_lens, q, q_lens, a)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            #for start in range(0, n_train, batch_size):
            for start, end in tqdm(train_eval_batches, desc='Train Eval: '):
                s = trainS[start:end]
                s_lens = trainS_lens[start:end]
                q = trainQ[start:end]
                q_lens = trainQ_lens[start:end]
                pred = model.predict(s, s_lens, q, q_lens)
                train_preds += list(pred)

            val_preds = model.predict(valS, valS_lens, valQ, valQ_lens)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels[:len(train_preds)])
            val_acc = metrics.accuracy_score(val_preds, val_labels[:len(val_preds)])

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

            with open('results_ordertest/rnn_no_posenc/task_{}.csv'.format(FLAGS.task_id), 'a') as csvfile:
                fieldnames = ['Epoch', 'Total Cost', 'Train Acc', 'Validation Acc', 'Test Acc']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'Epoch':t, 
                                 'Total Cost':total_cost,
                                 'Train Acc':train_acc,
                                 'Validation Acc':val_acc,
                                 'Test Acc':-1})


    test_preds = model.predict(testS, testS_lens, testQ, testQ_lens)
    test_acc = metrics.accuracy_score(test_preds, test_labels[:len(test_preds)])
    print("Testing Accuracy:", test_acc)
    with open('results_ordertest/rnn_no_posenc/task_{}_QA.csv'.format(FLAGS.task_id), 'a') as csvfile:
        fieldnames = ['Epoch', 'Total Cost', 'Train Acc', 'Validation Acc', 'Test Acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Epoch':t, 
                         'Total Cost':total_cost,
                         'Train Acc':train_acc,
                         'Validation Acc':val_acc,
                         'Test Acc':test_acc})


    print("Beginning Order test")
    best_val_acc = 0.0
    num_worse = 0
    # Run encoder study experiments using trained model
    for t in range(1, 500):
        np.random.shuffle(order_batches)
        total_cost = 0.0

        for start, end in order_batches:
            s = train_S_order[start:end]
            s_lens = train_S_order_lens[start:end]
            q = train_Q_order[start:end]
            a = train_A_order[start:end]
            cost_t = model.order_batch_fit(s, s_lens, q, a)
            total_cost += cost_t


        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train_order, batch_size):
                end = start + batch_size
                s = train_S_order[start:end]
                s_lens = train_S_order_lens[start:end]
                q = train_Q_order[start:end]
                pred = model.order_predict(s, s_lens, q)
                train_preds += list(pred)


            val_preds = model.order_predict(val_S_order, val_S_order_lens, val_Q_order)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_A_order[:, 1])
            val_acc = metrics.accuracy_score(val_preds, val_A_order[:, 1])

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

            if val_acc > best_val_acc:
                best_val_acc = val_acc 
                num_worse = 0   
            else:
                num_worse += 1

            if num_worse > 2:
                break

    test_preds = model.order_predict(test_S_order, test_S_order_lens, test_Q_order)
    test_acc = metrics.accuracy_score(test_preds, test_A_order[:, 1])
    print("Testing Accuracy:", test_acc)
    with open('results_ordertest/rnn_no_posenc/task_{}_ordertest.csv'.format(FLAGS.task_id), 'a') as csvfile:
        csvfile.write('{}, {}, {}\n'.format(t, train_acc, val_acc, test_acc))



