"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
        hops=3,
        max_grad_norm=40.0,
        nonlin=None,
        num_filters=2,
        initializer=tf.random_normal_initializer(stddev=0.1),
        encoding=position_encoding,
        session=tf.Session(),
        name='MemN2N'):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._num_filters = num_filters
        self._init = initializer
        self._name = name

        self.filter_sizes = [2, 3, 4]

        self._build_inputs()
        self._build_vars()

        self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr)

        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        logits = self._inference(self._stories, self._queries) # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)


    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")

    def _build_vars(self):
        # Use conv-filters of 2, 3, and 4 words

        with tf.variable_scope(self._name):
            num_filters_total = self._num_filters * len(self.filter_sizes)
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])

            self.A_1 = tf.Variable(A, name="A")
            self.TA_1 = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TA')

            self.A_conv_W = []
            self.A_conv_b = []

            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("A1_conv_{}".format(filter_size)):
                    filter_shape = [filter_size, self._embedding_size, 1, self._num_filters]
                    self.A_conv_W.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W"))
                    self.A_conv_b.append(tf.Variable(tf.constant(0.1, shape=[self._num_filters]), name="b"))

            self.W_A_proj = tf.tile(tf.Variable(self._init([1, num_filters_total, self._embedding_size]), name="W_A_proj"), [tf.shape(self._stories)[0], 1, 1])

            self.C = []
            self.TC = []

            self.C_conv_W = []
            self.C_conv_b = []

            self.W_C_proj = []

            for hopn in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name="C"))
                    self.TC.append(tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TC'))

                    self.C_conv_W.append([])
                    self.C_conv_b.append([])

                    for i, filter_size in enumerate(self.filter_sizes):
                        with tf.variable_scope("C_conv_{}".format(filter_size)):
                            filter_shape = [filter_size, self._embedding_size, 1, self._num_filters]
                            self.C_conv_W[hopn].append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W"))
                            self.C_conv_b[hopn].append(tf.Variable(tf.constant(0.1, shape=[self._num_filters]), name="b"))
                    
                    self.W_C_proj.append(tf.tile(tf.Variable(self._init([1, num_filters_total, self._embedding_size]), name="W_C_proj"), [tf.shape(self._stories)[0], 1, 1]))

            # Dont use projection for layerwise weight sharing
            # self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            
            # Use final C as replacement for W
            # self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")
        
        self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name):
            # Use A_1 for thee question embedding as per Adjacent Weight Sharing
            q_emb = tf.nn.embedding_lookup(self.A_1, queries)

            pooled_outputs = []

            # Apply conv accross all filter sizes and collect outputs
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("A1_conv_{}".format(filter_size)):
                    conv = tf.nn.conv2d(
                        tf.expand_dims(q_emb, -1),
                        self.A_conv_W[i],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, self.A_conv_b[i]), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self._sentence_size - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")

                    pooled_outputs.append(pooled)

            import ipdb
            ipdb.set_trace()

            num_filters_total = self._num_filters * len(self.filter_sizes)
            h_pool = tf.concat(3, pooled_outputs)
            h_pool_flat = tf.expand_dims(tf.reshape(h_pool, [-1, num_filters_total]), 1)

            q_A_proj = tf.batch_matmul(h_pool_flat, self.W_A_proj)

            u = [q_A_proj[:, 0]]

            for hopn in range(self._hops):
                if hopn == 0:
                    m_emb_A = tf.nn.embedding_lookup(self.A_1, stories)
                    m_emb_A_seq = tf.unpack(m_emb_A, axis=1)

                    all_h_pooled = []
                    for sentence in m_emb_A_seq:
                        pooled_outputs = []

                        # Apply conv accross all filter sizes and collect outputs
                        for i, filter_size in enumerate(self.filter_sizes):
                            with tf.variable_scope("A1_conv_{}".format(filter_size)):
                                conv = tf.nn.conv2d(
                                    tf.expand_dims(sentence, -1),
                                    self.A_conv_W[i],
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                                # Apply nonlinearity
                                h = tf.nn.relu(tf.nn.bias_add(conv, self.A_conv_b[i]), name="relu")
                                # Maxpooling over the outputs
                                pooled = tf.nn.max_pool(
                                    h,
                                    ksize=[1, self._sentence_size - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="pool")

                                pooled_outputs.append(pooled)
                        
                        num_filters_total = self._num_filters * len(self.filter_sizes)
                        h_pool = tf.concat(3, pooled_outputs)
                        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                        all_h_pooled.append(h_pool_flat)

                    m_A = tf.pack(all_h_pooled, axis=1)

                    m_A_proj = tf.batch_matmul(m_A, self.W_A_proj)

                    m_A = m_A_proj + self.TA_1

                else:
                    with tf.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], stories)

                        m_emb_A_seq = tf.unpack(m_emb_A, axis=1)

                        all_h_pooled = []
                        for sentence in m_emb_A_seq:
                            pooled_outputs = []

                            # Apply conv accross all filter sizes and collect outputs
                            for i, filter_size in enumerate(self.filter_sizes):
                                with tf.variable_scope("C_conv_{}".format(filter_size)):
                                    conv = tf.nn.conv2d(
                                        tf.expand_dims(sentence, -1),
                                        self.C_conv_W[hopn - 1][i],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="conv")
                                    # Apply nonlinearity
                                    h = tf.nn.relu(tf.nn.bias_add(conv, self.C_conv_b[hopn - 1][i]), name="relu")
                                    # Maxpooling over the outputs
                                    pooled = tf.nn.max_pool(
                                        h,
                                        ksize=[1, self._sentence_size - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")

                                    pooled_outputs.append(pooled)
                            
                            num_filters_total = self._num_filters * len(self.filter_sizes)
                            h_pool = tf.concat(3, pooled_outputs)
                            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                            all_h_pooled.append(h_pool_flat)

                        m_A = tf.pack(all_h_pooled, axis=1)

                        m_A_proj = tf.batch_matmul(m_A, self.W_C_proj[hopn - 1]) 
                        m_A = m_A_proj + self.TC[hopn - 1]

                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m_A * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])

                m_emb_C = tf.nn.embedding_lookup(self.C[hopn], stories)
                m_emb_C_seq = tf.unpack(m_emb_C, axis=1)

                with tf.variable_scope('hop_{}'.format(hopn)):
                    all_h_pooled = []
                    for sentence in m_emb_C_seq:
                        pooled_outputs = []

                        # Apply conv accross all filter sizes and collect outputs
                        for i, filter_size in enumerate(self.filter_sizes):
                            with tf.variable_scope("C_conv_{}".format(filter_size)):
                                conv = tf.nn.conv2d(
                                    tf.expand_dims(sentence, -1),
                                    self.C_conv_W[hopn][i],
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                                # Apply nonlinearity
                                h = tf.nn.relu(tf.nn.bias_add(conv, self.C_conv_b[hopn][i]), name="relu")
                                # Maxpooling over the outputs
                                pooled = tf.nn.max_pool(
                                    h,
                                    ksize=[1, self._sentence_size - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="pool")

                                pooled_outputs.append(pooled)
                        
                        num_filters_total = self._num_filters * len(self.filter_sizes)
                        h_pool = tf.concat(3, pooled_outputs)
                        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                        all_h_pooled.append(h_pool_flat)

                    m_C_pooled = tf.pack(all_h_pooled, axis=1)
                    m_C_proj = tf.batch_matmul(m_C_pooled, self.W_C_proj[hopn]) 
                    m_C = m_C_proj + self.TC[hopn]

                c_temp = tf.transpose(m_C, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                # Dont use projection layer for adj weight sharing
                # u_k = tf.matmul(u[-1], self.H) + o_k

                u_k = u[-1] + o_k

                # nonlinearity
                if self._nonlin:
                    u_k = nonlin(u_k)

                u.append(u_k)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(self._hops)):
                return tf.matmul(u_k, tf.transpose(self.C[-1], [1,0]))

    def batch_fit(self, stories, queries, answers, learning_rate):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._lr: learning_rate}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)
