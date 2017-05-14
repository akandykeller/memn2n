"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_integer("num_curric_steps", 4, "Number of sets to divide the training set into based on length")
tf.flags.DEFINE_integer("curric_stop_epoch", 50, "Number of sets to divide the training set into based on length")

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

# train/validation/test sets
S, Q, A, m_lens = vectorize_data(train, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA, train_m_lens, val_m_lens = cross_validation.train_test_split(S, Q, A, m_lens, test_size=.1, random_state=FLAGS.random_state)
testS, testQ, testA, test_m_lens = vectorize_data(test, word_idx, sentence_size, memory_size)

print(testS[0])

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


print("Splitting training into length batches")
train_index_by_len = {l:[] for l in np.unique(train_m_lens)}
  
for m_idx, m_l in enumerate(train_m_lens):
    train_index_by_len[m_l].append(m_idx)

unique_lens = sorted(np.unique(train_m_lens))

split_pct = 100.0 / FLAGS.num_curric_steps
percentiles = [i*split_pct for i in range(1, FLAGS.num_curric_steps+1)]

percentile_lens = [np.percentile(train_m_lens, p) for p in percentiles]

train_index_groups = []

for pl in percentile_lens:
    train_index_groups.append([])
    for key_len in train_index_by_len.keys():
        if key_len <= pl:
            train_index_groups[-1] += train_index_by_len[key_len]


print("Unique Lens: {}".format(unique_lens))
print("Percentile Lens: {}".format(percentile_lens))
print("Group lengths: {}".format(map(len, train_index_groups)))

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)
    for t in range(1, FLAGS.epochs+1):
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        group_idx = min(FLAGS.num_curric_steps - 1, int(np.floor(t / ((FLAGS.curric_stop_epoch+1) / float(FLAGS.num_curric_steps)))))
        batch_idxs = train_index_groups[group_idx]

        batches = zip(range(0, len(batch_idxs)-batch_size, batch_size), range(batch_size, len(batch_idxs), batch_size))
        batches = [batch_idxs[start:end] for start, end in batches]

        np.random.shuffle(batches)
        total_cost = 0.0
        for idxs in batches:
            s = trainS[idxs]
            q = trainQ[idxs]
            a = trainA[idxs]
            cost_t = model.batch_fit(s, q, a, lr)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, q)
                train_preds += list(pred)

            val_preds = model.predict(valS, valQ)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
            val_acc = metrics.accuracy_score(val_preds, val_labels)
            
            with open('results_personal/master_curric/task_{}.csv'.format(FLAGS.task_id), 'a') as f:
                f.write("{}, {}, {}, -1\n".format(t, train_acc, val_acc))
                
            print('-----------------------')
            print('Epoch', t)
            print('Current group index:', group_idx)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

    test_preds = model.predict(testS, testQ)
    test_acc = metrics.accuracy_score(test_preds, test_labels)
    print("Testing Accuracy:", test_acc)
    with open('results_personal/master_curric/task_{}.csv'.format(FLAGS.task_id), 'a') as f:
        f.write("{}, {}, {}, {}\n".format(t, train_acc, val_acc, test_acc))
            
