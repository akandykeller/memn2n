"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from wiki_data_utils import load_wiki, vectorize_data, get_word_idx, get_max_lens, build_hash, multi_acc_score
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce
from tqdm import tqdm
from random import randint

import tensorflow as tf
import numpy as np
import pickle

tf.flags.DEFINE_float("learning_rate", 0.005, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_evaluation_interval", 20, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 2, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 500, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 500, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 1000, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "movieqa/", "Directory containing movie QA dataset")
tf.flags.DEFINE_string("processed_data_dir", 'processed_data/', "Directory containing processed movie QA dataset")
tf.flags.DEFINE_boolean("save_processed_data", True, "Flag to determine if data should be saved agter preprocessing")
tf.flags.DEFINE_integer("max_key_len", 100, "Clip sentences to this length")
tf.flags.DEFINE_boolean("reprocess_data", False, "Flag to load or reprocess data")
tf.flags.DEFINE_boolean("reprocess_raw", False, "Flag to load or reprocess data from raw")

FLAGS = tf.flags.FLAGS

if FLAGS.reprocess_data:
    if FLAGS.reprocess_raw:
        print("Reprocessing raw data from scratch...")
        print("Loading Wiki Data")
        docs, questions, ent_lists = load_wiki(FLAGS.data_dir)

        print("Creating Word Index")
        word_idx = get_word_idx(docs, questions)

        print("Building Hash")
        train_hash, dev_hash, test_hash = build_hash(docs, questions, word_idx)
        
        if FLAGS.save_processed_data:
            with open(FLAGS.processed_data_dir + 'word_idx.pkl', 'w') as f:
                pickle.dump(word_idx, f)

            with open(FLAGS.processed_data_dir + 'load_wiki.pkl', 'w') as f:
                pickle.dump((docs, questions, ent_lists), f)
            
            with open(FLAGS.processed_data_dir + 'hash.pkl', 'w') as f:
                pickle.dump((train_hash, dev_hash, test_hash), f)
    else:
        print("Loading preprocessed raw data from {}: word_idx, docs, questions, ent_lists, hash".format(FLAGS.processed_data_dir))
        with open(FLAGS.processed_data_dir + 'word_idx.pkl', 'r') as f:
            word_idx = pickle.load(f)

        with open(FLAGS.processed_data_dir + 'load_wiki.pkl', 'r') as f:
            docs, questions, ent_lists = pickle.load(f)

        with open(FLAGS.processed_data_dir + 'hash.pkl', 'r') as f:
            train_hash, dev_hash, test_hash = pickle.load(f)
    
    key_length, value_length, query_length = get_max_lens(docs, questions, FLAGS.max_key_len)
    re_list, entities, ent_rev, ent_idx = ent_lists
    memory_size = min(FLAGS.memory_size, max(map(len, train_hash)))
    vocab_size = len(word_idx) + 1 # +1 for nil word

    print("Longest key length", key_length)
    print("Longest value length", value_length)
    print("Longest query length", query_length)
    print("Vocab Size", vocab_size)
    print("Memory Size", memory_size)
    print("Mean hash size", np.mean(map(len, train_hash)))
    print("Max hash size", np.max(map(len, train_hash)))
    print("Min hash size", np.min(map(len, train_hash)))
    print("Number of Entitites", len(entities))

    print("Vectorizing data Train")
    trainS, trainQ, trainA = vectorize_data(docs, questions[0], ent_idx, word_idx, train_hash, memory_size, key_length, query_length)
    print("TrainS shape:{}".format(trainS.shape))

    print("Vectorizing data val")
    valS, valQ, valA = vectorize_data(docs, questions[1], ent_idx, word_idx, dev_hash, memory_size, key_length, query_length)
    print("ValS shape:{}".format(valS.shape))

    print("Vectorizing data test")
    testS, testQ, testA = vectorize_data(docs, questions[2], ent_idx, word_idx, test_hash, memory_size, key_length, query_length)
    print("testS shape:{}".format(testS.shape))

    if FLAGS.save_processed_data:
        print("Data processing complete. Saving to {}.".format(FLAGS.processed_data_dir))

        print("Saving Train")
        with open(FLAGS.processed_data_dir + 'trainS_1000m.npy', 'w') as f:
            np.save(f, trainS)

        with open(FLAGS.processed_data_dir + 'trainQ_1000m.npy', 'w') as f:
            np.save(f, trainQ)

        with open(FLAGS.processed_data_dir + 'trainA_1000m.npy', 'w') as f:
            np.save(f, trainA)
        
        print("Saving Val")
        with open(FLAGS.processed_data_dir + 'valS_1000m.npy', 'w') as f:
            np.save(f, valS)

        with open(FLAGS.processed_data_dir + 'valQ_1000m.npy', 'w') as f:
            np.save(f, valQ)

        with open(FLAGS.processed_data_dir + 'valA_1000m.npy', 'w') as f:
            np.save(f, valA)
        
        print("Saving Test")
        with open(FLAGS.processed_data_dir + 'testS_1000m.npy', 'w') as f:
            np.save(f, testS)

        with open(FLAGS.processed_data_dir + 'testQ_1000m.npy', 'w') as f:
            np.save(f, testQ)

        with open(FLAGS.processed_data_dir + 'testA_1000m.npy', 'w') as f:
            np.save(f, testA)

else:
    print("Loading saved data from: {}".format(FLAGS.processed_data_dir))
    
    with open(FLAGS.processed_data_dir + 'word_idx.pkl', 'r') as f:
        word_idx = pickle.load(f)

    with open(FLAGS.processed_data_dir + 'load_wiki.pkl', 'r') as f:
        docs, questions, ent_lists = pickle.load(f)

    key_length, value_length, query_length = get_max_lens(docs, questions, FLAGS.max_key_len)
    re_list, entities, ent_rev, ent_idx = ent_lists

    print("Loading Train")
    with open(FLAGS.processed_data_dir + 'trainS_1000m.npy', 'r') as f:
        trainS = np.load(f)

    with open(FLAGS.processed_data_dir + 'trainQ_1000m.npy', 'r') as f:
        trainQ = np.load(f)

    with open(FLAGS.processed_data_dir + 'trainA_1000m.npy', 'r') as f:
        trainA = np.load(f)
    
    print("Loading Val")
    with open(FLAGS.processed_data_dir + 'valS_1000m.npy', 'r') as f:
        valS = np.load(f)

    with open(FLAGS.processed_data_dir + 'valQ_1000m.npy', 'r') as f:
        valQ = np.load(f)

    with open(FLAGS.processed_data_dir + 'valA_1000m.npy', 'r') as f:
        valA = np.load(f)
    
    print("Loading Test")
    with open(FLAGS.processed_data_dir + 'testS_1000m.npy', 'r') as f:
        testS = np.load(f)

    with open(FLAGS.processed_data_dir + 'testQ_1000m.npy', 'r') as f:
        testQ = np.load(f)

    with open(FLAGS.processed_data_dir + 'testA_1000m.npy', 'r') as f:
        testA = np.load(f)

    memory_size = trainS.shape[1]
    key_length = value_length = trainS.shape[2]
    query_length = trainQ.shape[1]
    vocab_size = len(word_idx) + 1

num_entities = len(ent_idx) + 1

print(testS[0])
print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

# There can be multiple labels for an answer, figure this out. 
# train_labels = np.argmax(trainA, axis=1)
# test_labels = np.argmax(testA, axis=1)
# val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, key_length, value_length, query_length, memory_size, FLAGS.embedding_size, 
                   num_entities, session=sess, hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)
    for t in range(1, FLAGS.epochs+1):
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        batch_num = 0
        for start, end in tqdm(batches):
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]

            # Encode answers as valid probability distribution
            y = np.zeros([end - start, len(ent_idx) + 1])
            
            for b_idx, ans in enumerate(a):
                for ans_idx in ans:
                    # Use uniform distribution for multiple answers
                    y[b_idx][ans_idx] = 1.0 / len(ans)

            cost_t = model.batch_fit(s, s, q, y, lr)
            total_cost += cost_t
            batch_num += 1

            if batch_num % FLAGS.batch_evaluation_interval == 0:
                print("Running Eval.")

                t_start = randint(0, n_train - batch_size)
                t_end = t_start + batch_size
                s = trainS[t_start:t_end]
                q = trainQ[t_start:t_end]
                train_preds = list(model.predict(s, s, q))

                v_start = randint(0, n_val - batch_size)
                v_end = v_start + batch_size
                s = valS[v_start:v_end]
                q = valQ[v_start:v_end]

                val_preds = model.predict(s, s, q)

                train_acc = multi_acc_score(np.array(train_preds), trainA[t_start:t_end])
                val_acc = multi_acc_score(val_preds, valA[v_start:v_end])

                print('-----------------------')
                print('Batch', batch_num)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')

        if t % FLAGS.evaluation_interval == 0:
            print("Running Eval.")
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, s, q)
                train_preds += list(pred)

            val_preds = model.predict(valS, valS, valQ)

            train_acc = multi_acc_score(np.array(train_preds), trainA)
            val_acc = multi_acc_score(val_preds, valA)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

    test_preds = model.predict(testS, testS, testQ)
    test_acc = multi_acc_score(test_preds, testA)
    print("Testing Accuracy:", test_acc)
