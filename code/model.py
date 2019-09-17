import os
import time
import tensorflow as tf
import numpy as np

import gmm

class NoneAttentionWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, cell, fixed_c):
        """
        fixed_c: shape: (batch_size, c_size)
        """
        super(NoneAttentionWrapper, self).__init__()
        self._cell=cell
        self._fixed_c = fixed_c
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size  
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)
    
    def __call__(self, inputs, state, scope=None):
        actual_inputs = tf.concat([inputs, self._fixed_c], axis=1)
        return self._cell(actual_inputs, state, scope=scope)
        

def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())
    

def get_default_hparams():
    """Return default HParams for FontRNN."""
    hparams = tf.contrib.training.HParams(
        data_set="FZTLJW_775.npz",  # Our dataset.
        data_dir="../data",  # The directory in which to find the data_set.
        log_root="../log/demo",  # Directory to store model checkpoints, tensorboard.
        num_steps=10000,  # Max number of steps of training. Keep large.
        save_every=100,   # Number of batches per checkpoint creation.
        max_seq_len=300,  # Not used. Will be changed by data_set.
        enc_rnn_size=256,
        dec_rnn_size=256,
        batch_size=128, 
        grad_clip=1.0,  # Gradient clipping.
        num_mixture=20,  # Number of mixtures in Gaussian mixture model.
        learning_rate=0.001, # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        min_learning_rate=0.000001,  # Minimum learning rate.
        random_scale_factor=0.10,  # Random scaling data augmention proportion.
        augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
        scale_factor=300.0,  # Scale the delta_x and delta_y, constant.
        rnn_dropout_keep_prob=0.6,  # recurrent dropout keep prob, only used for training
        use_layer_norm=False,
        is_training=True,  # set to False if and only if for testing
        attention_method='LM',  # L: LuongAttention, B: BahdanauAttention, LM: LuongMonotonicAttention, N: not use attention
        lc_weight=2.0,  # The lambda_c in Seq(19) in paper.
    )
    return hparams


class FontRNN(object):
    def __init__(self, hps, reuse=tf.AUTO_REUSE):
        self.hps = hps
        self.build_model(reuse=reuse)

    def encoder(self, inputs, sequence_lengths, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("Encoder", reuse=reuse):
            # enc_cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            enc_cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell

            with tf.variable_scope("rnn", reuse=reuse):
                enc_cell_fw = enc_cell_fn(num_units=self.hps.enc_rnn_size, layer_norm=self.hps.use_layer_norm,
                                            dropout_keep_prob=self.hps.rnn_dropout_keep_prob)  # forward RNNCell
                enc_cell_bw = enc_cell_fn(num_units=self.hps.enc_rnn_size, layer_norm=self.hps.use_layer_norm,
                                            dropout_keep_prob=self.hps.rnn_dropout_keep_prob)  # backward RNNCell

                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                    enc_cell_fw,
                    enc_cell_bw,
                    inputs,
                    sequence_length=sequence_lengths,
                    swap_memory=True,
                    dtype=tf.float32)

            output_fw, output_bw = outputs  # output_fw shape: ( batch_size, max_seq_len, enc_size)
            last_state_fw, last_state_bw = output_states
            last_h_fw = last_state_fw[1] # we only use the hidden state
            last_h_bw = last_state_bw[1]

            all_h = tf.concat([output_fw, output_bw], 2)  # shape: (batch_size, max_seq_len, 2*enc_size)
            last_h = tf.concat([last_h_fw, last_h_bw], 1)  # shape: (batch_size, 2*enc_size)

            return all_h, last_h

    def decoder(self, enc_last_h, enc_all_h, enc_seq_lens, dec_input, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("Decoder", reuse=reuse):
            dec_cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell

            dec_cell = dec_cell_fn(num_units=self.hps.dec_rnn_size, layer_norm=self.hps.use_layer_norm,
                                              dropout_keep_prob=self.hps.rnn_dropout_keep_prob)

            # the shape of hidden and cell state are both (batch_size, dec_rnn_size)
            init_state_output_size = 2 * self.hps.dec_rnn_size 
            initial_state = tf.nn.tanh(  # use the last h of encoder to initiate the state of decoder
                tf.layers.dense(
                    inputs=enc_last_h,
                    units=init_state_output_size,
                    kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                    name="init_state_fc"
                )) # shape = (batch_size, dec_rnn_size * 2)

            c0, h0 = tf.split(initial_state, 2, 1)
            decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c0, h0)


            # ----------- define the three functions that the CustomHelper will use -------
            batch_size = self.hps.batch_size
            max_seq_len = self.hps.max_seq_len
            def initial_fn():
                initial_elements_finished = (1 < np.zeros(batch_size))  # initial false, shape:(batch_size, )
                initial_input = np.zeros((batch_size, 5), dtype=np.float32)
                initial_input[:, 2] = 1  # initial, P1 = 1
                return initial_elements_finished, initial_input

            def sample_fn(time, outputs, state): 
                unused_sample_ids = tf.zeros([outputs.shape[0]])
                return unused_sample_ids

            def next_inputs_fn(time, outputs, state, sample_ids):  # define the next input by the current output
                [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits] = gmm.get_mixture_coef(outputs)

                idx_eos = tf.argmax(pen, axis=1)
                eos = tf.one_hot(idx_eos, depth=3)

                next_x1 = tf.reduce_sum(tf.multiply(mu1, pi), axis=1, keepdims=True)
                next_x2 = tf.reduce_sum(tf.multiply(mu2, pi), axis=1, keepdims=True)
                next_x = tf.concat([next_x1, next_x2], axis=1)

                next_inputs = tf.concat([next_x, eos], axis=1) # shape: (batch_size, 5)

                tmp = tf.ones([next_x.shape[0]])
                elements_finished_1 = tf.equal(tmp, eos[:, -1])  # this operation produces boolean tensor of [batch_size]
                elements_finished_2 = (time >= max_seq_len)

                elements_finished = tf.logical_or(elements_finished_1, elements_finished_2)
                next_state = state
                return elements_finished, next_inputs, next_state

            my_inference_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

            # Create an attention mechanism
            if self.hps.attention_method not in ['L', 'B', 'LM', 'BM', 'N']:
                raise ValueError("The attention_method should be one of [L, B, LM, BM, N], but get %s."%str(self.hps.attention_method))
            if self.hps.attention_method == 'L':
                tf.logging.info('Model using LuongAttention.')
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                                            num_units=self.hps.dec_rnn_size,
                                                            memory=enc_all_h, 
                                                            memory_sequence_length=enc_seq_lens,
                                                            name='LuongAttention'
                                                        )
            elif self.hps.attention_method == 'B':
                tf.logging.info('Model using BahdanauAttention.')
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                                            num_units=self.hps.dec_rnn_size,
                                                            memory=enc_all_h, 
                                                            memory_sequence_length=enc_seq_lens,
                                                            name='BahdanauAttention'
                                                        )
            elif self.hps.attention_method == 'LM':
                tf.logging.info('Model using LuongMonotonicAttention.')
                attention_mechanism = tf.contrib.seq2seq.LuongMonotonicAttention(
                                                            num_units=self.hps.dec_rnn_size,
                                                            memory=enc_all_h,
                                                            memory_sequence_length=enc_seq_lens,
                                                            name='LuongMonotonicAttention'
                                                        )
            elif self.hps.attention_method == 'BM':
                tf.logging.info('Model using BahdanauMonotonicAttention.')
                attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(
                                                            num_units=self.hps.dec_rnn_size,
                                                            memory=enc_all_h, 
                                                            memory_sequence_length=enc_seq_lens,
                                                            normalize=True,
                                                            name='BahdanauMonotonicAttention'
                                                        )
            elif self.hps.attention_method == "N":
                tf.logging.info('DO NOT use Attention!!!')
            
            
            # Helper
            if self.hps.is_training:
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_input,
                                                           sequence_length=tf.convert_to_tensor([max_seq_len for _ in range(batch_size)]),
                                                           name='TrainingHelper')
            else:
                helper = my_inference_helper


            if self.hps.attention_method == "N":
                dec_cell_with_attn = NoneAttentionWrapper(dec_cell, enc_last_h)
                att_wrapper_initial_state = decoder_initial_state

            else:
                dec_cell_with_attn = tf.contrib.seq2seq.AttentionWrapper(
                                                dec_cell,
                                                attention_mechanism,
                                                attention_layer_size=self.hps.dec_rnn_size, 
                                                alignment_history=True,
                                                name='AttentionWrapper'
                                                )
                # Decoder
                att_wrapper_initial_state = dec_cell_with_attn.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32).clone(
                                                                                            cell_state=decoder_initial_state)
                                                                                            
            n_out = (3 + self.hps.num_mixture * 6)  # the dim of output of decoder
            fc_layer = tf.layers.Dense(n_out, name='output_fc')
            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell_with_attn, helper,
                                                      initial_state=att_wrapper_initial_state,
                                                      output_layer=fc_layer)
            # Dynamic decoding
            decoder_final_outputs, decoder_final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                                                      decoder=decoder,
                                                      scope='dynamic_decode')


            output, sample_id = decoder_final_outputs # this is a AttentionWrapperState

            if self.hps.attention_method == "N":
                self.timemajor_alignment_history = tf.ones((self.hps.max_seq_len, self.hps.batch_size, self.hps.max_seq_len))
            else:
                cell_state, att, time, alignments, alignment_history, attention_state = decoder_final_state
                # alignment_history is a tensorArray, use stack() to convert it to a tensor
                self.timemajor_alignment_history = alignment_history.stack()

            return output, self.timemajor_alignment_history

  
    def build_model(self, reuse=tf.AUTO_REUSE):
        """Define model architecture."""
        self.enc_seq_lens = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size])  # encoder actual input data length
        self.dec_seq_lens = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size])  # decoder actual input data length

        # input of encoder, reference data. We insert (0, 0, 1, 0, 0) at timestep_0, so "max_seq_len + 1"
        self.enc_input_data = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])
        # input of decoder, target data
        self.dec_input_data = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5]) 

        # encoding
        enc_input_x = self.enc_input_data[:, 1:self.hps.max_seq_len + 1, :] # R_1 ~ R_{max_seq_len}
        enc_all_h, enc_last_h = self.encoder(enc_input_x, self.enc_seq_lens, reuse=reuse)

        # decoding
        dec_input = self.dec_input_data[:, :self.hps.max_seq_len, :]  # T0 ~ T_{max_seq_len-1}
        dec_out, timemajor_attn_hist = self.decoder(enc_last_h, enc_all_h, self.enc_seq_lens, dec_input)

        batch_major_attn_hist = tf.transpose(timemajor_attn_hist, perm=[1, 0, 2]) 

        n_out = (3 + self.hps.num_mixture * 6)  # decoder output dimension
        dec_out = tf.reshape(dec_out, [-1, n_out])  # shape = (batch_size * max_seq_len, n_out)

        # shape of first 6 tensors: (batch_size * max_seq_len, num_mixture), 
        # shape of last 2 tensors: (batch_size * max_seq_len, 3)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = gmm.get_mixture_coef(dec_out)
        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr
        self.pen_logits = o_pen_logits
        self.pen = o_pen

        # reshape target data so that it is compatible with prediction shape
        target = tf.reshape(self.dec_input_data[:, 1:self.hps.max_seq_len + 1, :], [-1, 5])  # (batch_size * max_seq_le, 5)
        [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
        pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

        # Seq(16) and Seq(17) in paper
        Ld, Lc = gmm.get_loss(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits, 
                                x1_data, x2_data, pen_data)
        self.Ld = tf.reduce_sum(Ld) / tf.to_float(tf.reduce_sum(self.dec_seq_lens))
        self.Lc = tf.reduce_mean(Lc)

        self.Loss = self.Ld + self.hps.lc_weight * self.Lc # Seq(19) in paper


        if self.hps.is_training:
            with tf.variable_scope("optimizer", reuse=reuse):
                self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
                optimizer = tf.train.AdamOptimizer(self.lr)

                gvs = optimizer.compute_gradients(self.Loss)
                g = self.hps.grad_clip

                capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gvs)

            with tf.name_scope("summary"):
                Loss_summ = tf.summary.scalar("Loss", self.Loss)
                Ld_summ = tf.summary.scalar("Ld", self.Ld)
                Lc_summ = tf.summary.scalar("Lc", self.Lc)
                lr_summ = tf.summary.scalar("lr", self.lr)
                self.summ = tf.summary.merge([Loss_summ, Ld_summ, Lc_summ, lr_summ])
        else:
            assert self.hps.rnn_dropout_keep_prob == 1.0






