"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

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
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    # encoding[:, -1] = 1.0
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
        use_proj=False,
        rnn_input_keep_prob=0.8,
        rnn_output_keep_prob=1.0,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
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
        self._rnn_input_keep_prob = rnn_input_keep_prob
        self._rnn_output_keep_prob = rnn_output_keep_prob
        self._init = initializer
        self._opt = optimizer
        self._name = name
        self._use_proj = use_proj
        self._use_rnn_comb = False

        self._build_inputs()
        self._build_vars()
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        logits, q_rec_loss, m_rec_loss, c_rec_loss = self._inference(self._stories, self._stories_len, self._queries, self._queries_len) # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = (1.0 - self._ae_lw) * cross_entropy_sum + self._ae_lw * (q_rec_loss + m_rec_loss + c_rec_loss)
        # loss_op = cross_entropy_sum + self._ae_lw * (q_rec_loss + m_rec_loss + c_rec_loss)

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
        self._stories_len = tf.placeholder(tf.int32, [None, self._memory_size], name="stories_lens")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._queries_len = tf.placeholder(tf.int32, [None], name="queries_lens")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._ae_lw = tf.placeholder(tf.float32, ())

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            
            if self._use_proj:
                self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])

            self.A_1 = tf.Variable(A, name="A")

            self.C = []

        for hopnum in range(self._hops):
            with tf.variable_scope('hop_{}'.format(hopnum)):
                self.C.append(tf.Variable(C, name="C_{}".format(hopnum)))

        self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])


    def _inference(self, stories, stories_lens, queries, queries_lens):
        # Adjacent weight sharing scheme
        rnn_A_scopes = ['RNN_A_1', 'RNN_C_1', 'RNN_C_2']
        rnn_C_scopes = ['RNN_C_1', 'RNN_C_2', 'RNN_C_3']
        
        # Layerwise weight sharing scheme
        # rnn_A_scopes = ['RNN_A_1', 'RNN_A_1', 'RNN_A_2']
        # rnn_C_scopes = ['RNN_C_1', 'RNN_C_1', 'RNN_C_1']
        # self._use_proj = True

        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.A_1, queries)

        # Let B = A_1
        with tf.variable_scope('RNN_Q', reuse=None):
            # Encode Query with GRU & Take last state as u_0
            q_cell = rnn_cell.GRUCell(self._embedding_size)
            # Add dropout
            q_cell = rnn_cell.DropoutWrapper(q_cell,
                                             input_keep_prob=self._rnn_input_keep_prob,
                                             output_keep_prob=self._rnn_output_keep_prob)
            q_init_state = q_cell.zero_state(tf.shape(stories)[0], tf.float32)
            q_outputs, q_state = rnn.dynamic_rnn(q_cell, q_emb, initial_state=q_init_state,
                                                 sequence_length=queries_lens)

        with tf.variable_scope(rnn_A_scopes[0] + '_dec', reuse=None):
            # Decode query with 2nd GRU as autoencoder for regularization
            q_dec_cell = rnn_cell.GRUCell(self._embedding_size)
            q_dec_cell = rnn_cell.DropoutWrapper(q_dec_cell,
                                             input_keep_prob=self._rnn_input_keep_prob,
                                             output_keep_prob=self._rnn_output_keep_prob)

            # only feed non-padded q_emb to decoder... cut by lengths, flip & repad
            q_emb_revs = []
            # Need to cut sentences individually for each batch since lengths differ by batch
            for i in range(self._batch_size):
                q_emb_rev_nopad = tf.concat(0, [tf.zeros((1, self._embedding_size), dtype=tf.float32), q_emb[i, :queries_lens[i], :][::-1, :]])[:-1, :]
                ql = self._sentence_size - queries_lens[i]
                q_emb_revs.append(tf.concat(0,  [q_emb_rev_nopad, tf.zeros(([ql, self._embedding_size]), dtype=tf.float32)]))
            # Re-combine over the batch
            q_emb_rev = tf.pack(q_emb_revs)

            q_dec_outputs, q_dec_state = rnn.dynamic_rnn(q_dec_cell, q_emb_rev, 
                                                         initial_state=q_state,
                                                         sequence_length=queries_lens)

            # compute reconstruction loss for regularization
            q_dec_outputs_rev = q_dec_outputs[:, ::-1, :]
            q_rec_loss = tf.reduce_sum(tf.square(tf.sub(q_dec_outputs_rev, q_emb))) / tf.to_float(tf.reduce_sum(queries_lens))

        u = [q_state]
        m_rec_loss = tf.constant(0.0)
        c_rec_loss = tf.constant(0.0)
        
        for hopn in range(self._hops):
            if hopn == 0:
                with tf.variable_scope(self._name):
                    m_emb_A = tf.nn.embedding_lookup(self.A_1, stories)
            else:
                with tf.variable_scope('hop_{}'.format(hopn - 1)):
                    m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], stories)
            
            # Duplicate u[-1] so we can append it to each rnn input by concat
            u_dupe = tf.reshape(tf.tile(u[-1], [self._sentence_size, 1]), 
                                [self._batch_size, self._sentence_size, self._embedding_size])

            # m_emb_a is shape (bsz, num_sentences, sentence_length, embedding_size)
            # We need to feed into rnn individual sentence at a time
            m_emb_A_sentences = tf.unpack(m_emb_A, axis=1)

            m_A_states_all_sent = [] 

            m_cell = rnn_cell.GRUCell(self._embedding_size)
            m_cell = rnn_cell.DropoutWrapper(m_cell,
                                             input_keep_prob=self._rnn_input_keep_prob,
                                             output_keep_prob=self._rnn_output_keep_prob)

            # Decode query with 2nd GRU as autoencoder for regularization
            m_dec_cell = rnn_cell.GRUCell(self._embedding_size)
            m_dec_cell = rnn_cell.DropoutWrapper(m_dec_cell,
                                            input_keep_prob=self._rnn_input_keep_prob,
                                            output_keep_prob=self._rnn_output_keep_prob)

            for i, sentence in enumerate(m_emb_A_sentences):
                reuse = None if ((i == 0) and (hopn == 0)) else True

                # Use adj-weight sharing 
                with tf.variable_scope(rnn_A_scopes[hopn], reuse=reuse):
                    m_init_state = m_cell.zero_state(tf.shape(stories)[0], tf.float32)

                    sentence_u = tf.concat(2, [sentence, u_dupe])

                    m_states = rnn.dynamic_rnn(m_cell, sentence_u, initial_state=m_init_state,
                                               sequence_length=stories_lens[:, i])
                    m_A_states_all_sent.append(m_states)

                reuse_dec = True
                with tf.variable_scope(rnn_A_scopes[hopn] + '_dec', reuse=reuse_dec):
                    # only feed non-padded q_emb to decoder... cut by lengths, flip & repad
                    m_emb_revs = []
                    # Need to cut sentences individually for each batch since lengths differ by batch
                    for b in range(self._batch_size):
                        m_emb_rev_nopad = tf.concat(0, [tf.zeros((1, self._embedding_size), dtype=tf.float32), sentence[b, :stories_lens[b, i], :][::-1, :]])[:-1, :]
                        ml = self._sentence_size - stories_lens[b, i]
                        m_emb_revs.append(tf.concat(0,  [m_emb_rev_nopad, tf.zeros(([ml, self._embedding_size]), dtype=tf.float32)]))
                    # Re-combine over the batch
                    m_emb_rev = tf.pack(m_emb_revs)

                    m_dec_outputs, m_dec_state = rnn.dynamic_rnn(m_dec_cell, m_emb_rev, 
                                                                 initial_state=m_states[-1],
                                                                 sequence_length=stories_lens[:, i])

                    # compute reconstruction loss for regularization
                    m_dec_outputs_rev = m_dec_outputs[:, ::-1, :]
                    m_rec_loss += tf.reduce_sum(tf.square(tf.sub(m_dec_outputs_rev, sentence))) / tf.to_float(tf.reduce_sum(stories_lens[:, i]))

            m_A_states_h = tf.pack([h for c, h in m_A_states_all_sent])
            m_A_states_h_t = tf.transpose(m_A_states_h, [1, 0 ,2])

            m_A = m_A_states_h_t

            # hack to get around no reduce_dot
            u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])

            dotted = tf.reduce_sum(m_A * u_temp, 2)
            
            # Calculate probabilities
            probs = tf.nn.softmax(dotted)
                
            with tf.variable_scope('hop_{}'.format(hopn)):
                # Use LSTM cell to generate new m (aka c) from m_emb again?
                m_emb_C = tf.nn.embedding_lookup(self.C[hopn], stories)

            m_emb_C_sentences = tf.unpack(m_emb_C, axis=1)

            m_C_states_all_sent = [] 
            
            c_cell = rnn_cell.GRUCell(self._embedding_size)

            c_cell = rnn_cell.DropoutWrapper(c_cell,
                                             input_keep_prob=self._rnn_input_keep_prob,
                                             output_keep_prob=self._rnn_output_keep_prob)

            # Decode query with 2nd GRU as autoencoder for regularization
            c_dec_cell = rnn_cell.GRUCell(self._embedding_size)
            c_dec_cell = rnn_cell.DropoutWrapper(c_dec_cell,
                                                 input_keep_prob=self._rnn_input_keep_prob,
                                                 output_keep_prob=self._rnn_output_keep_prob)

            for i, sentence in enumerate(m_emb_C_sentences):
                reuse = None if (i == 0) else True
                c_init_state = c_cell.zero_state(tf.shape(stories)[0], tf.float32)

                # Again use adj-weight sharing for rnn weights
                with tf.variable_scope(rnn_C_scopes[hopn], reuse=reuse):
                    sentence_u = tf.concat(2, [sentence, u_dupe])

                    c_states = rnn.dynamic_rnn(c_cell, sentence_u, initial_state=c_init_state,
                                               sequence_length=stories_lens[:, i])
                    m_C_states_all_sent.append(c_states[-1])

                reuse_dec = None if (i == 0) else True
                with tf.variable_scope(rnn_C_scopes[hopn] + '_dec', reuse=reuse_dec):
                    # only feed non-padded q_emb to decoder... cut by lengths, flip & repad
                    c_emb_revs = []
                    # Need to cut sentences individually for each batch since lengths differ by batch
                    for b in range(self._batch_size):
                        c_emb_rev_nopad = tf.concat(0, [tf.zeros((1, self._embedding_size), dtype=tf.float32), sentence[b, :stories_lens[b, i], :][::-1, :]])[:-1, :]
                        cl = self._sentence_size - stories_lens[b, i]
                        c_emb_revs.append(tf.concat(0,  [c_emb_rev_nopad, tf.zeros(([cl, self._embedding_size]), dtype=tf.float32)]))
                    # Re-combine over the batch
                    c_emb_rev = tf.pack(c_emb_revs)

                    c_dec_outputs, c_dec_state = rnn.dynamic_rnn(c_dec_cell, c_emb_rev, 
                                                                 initial_state=c_states[-1],
                                                                 sequence_length=stories_lens[:, i])

                    # compute reconstruction loss for regularization
                    c_dec_outputs_rev = c_dec_outputs[:, ::-1, :]
                    c_rec_loss += tf.reduce_sum(tf.square(tf.sub(c_dec_outputs_rev, sentence))) / tf.to_float(tf.reduce_sum(stories_lens[:, i]))


            m_C_states_h = tf.pack(m_C_states_all_sent)
            m_C_states_h_t = tf.transpose(m_C_states_h, [1, 0 ,2])
           
            with tf.variable_scope('hop_{}'.format(hopn)):
                m_C = m_C_states_h_t

            m_C_t = tf.transpose(m_C, [0, 2, 1])

            # Take weighted sum of embedded memories
            o_k = tf.reduce_sum(m_C_t * tf.expand_dims(probs, 1), 2)
                    
            u_cell = rnn_cell.GRUCell(self._embedding_size)

            #import ipdb
            #ipdb.set_trace()
            reuse = None if hopn == 0 else True
            with tf.variable_scope('RNN_U_Scope', reuse=reuse):
                if self._use_rnn_comb:
                    u_k_states = rnn.dynamic_rnn(u_cell, tf.expand_dims(o_k, -1), initial_state=u[-1], sequence_length=tf.ones_like(stories_lens[:,0]))
                    u_k = u_k_states[-1]
                elif self._use_proj:
                    u_k = tf.matmul(u[-1], self.H) + o_k
                else:
                    u_k = u[-1] + o_k

                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

        with tf.variable_scope('hop_{}'.format(self._hops)):
            return tf.matmul(u_k, tf.transpose(self.C[-1], [1,0])), q_rec_loss, m_rec_loss, c_rec_loss

    def batch_fit(self, stories, s_lens, queries, q_lens, answers, ae_lw):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._stories_len: s_lens, 
                     self._queries: queries, self._queries_len: q_lens, 
                     self._answers: answers, self._ae_lw: ae_lw}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, s_lens, queries, q_lens):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._stories_len: s_lens, 
                     self._queries: queries, self._queries_len: q_lens} 
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
