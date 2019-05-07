# Copyright 2019 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import math
import numpy as np

import qnarre.neura as Q

from tensor2tensor.layers import common_layers

from tensorflow.python.util import nest

# Default value for INF
INF = 1. * 1e7


class Keys:
    # Score = log probability / length norm
    DONE_SCORES = 'DONE_SCORES'
    # Flags indicating which sequences in the finished sequences are finished.
    # At the beginning, all of the sequences in DONE_SEQ are filler values.
    # True -> finished sequence, False -> filler. Shape [batch_size, beam_size]
    DONE_FLAGS = 'DONE_FLAGS'


class Beam(Q.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.symbols_to_logits_fn = symbols_to_logits_fn
        # self.vocab_size = vocab_size
        # PS.batch_size = batch_size
        #  PS.beam_size = beam_size
        # self.alpha = alpha
        # PS.tgt_len = tgt_len
        # self.eos_id = eos_id

    def inflate_beam(self, x):
        y = Q.expand_dims(x, axis=1)
        dims = [1] * x.shape.ndims
        dims[1] = self.PS.beam_size
        y = Q.tile(y, dims)
        return y

    def build(self, input_shape):
        PS = self.PS
        logps = Q.constant([[0.] + [PS.big_neg] * (PS.beam_size - 1)])
        self.logps = Q.tile(logps, [input_shape[:1], 1])
        return super().build(input_shape)

    def call(self, inputs, **kw):
        tgt = inputs
        PS = self.PS
        self.alive = Q.expand_dims(self.inflate_beam(tgt), axis=2)
        self.cache = nest.map_structure(lambda t: self.inflate_beam(t))
        self.done = Q.zeros(Q.shape(self.alive), Q.int32)
        self.scores = Q.ones([PS.batch_size, PS.beam_size]) * -INF
        self.flags = Q.zeros([PS.batch_size, PS.beam_size], Q.bool)
        i = 0
        while self.not_done(i):
            seq, logps = self._grow_alive()
            alive = self._get_new_alive_state(seq, logps)
            done = self._get_new_finished_state(seq, logps)
            i += 1
        done = Q.where(Q.reduce_any(self.flags, 1), self.done, self.alive)
        scores = Q.where(Q.reduce_any(self.flags, 1), self.scores, self.logps)
        return done, scores

    def not_done(self, i):
        PS = self.PS
        new = self.logps[:, 0] / self.len_penalty(PS.tgt_len)
        self.scores *= Q.cast(self.flags, Q.floatx())
        old = Q.reduce_min(self.scores, axis=1)
        fs = Q.reduce_any(self.flags, axis=1)
        old += (1. - Q.cast(fs, Q.floatx())) * self.PS.big_neg
        old_better = Q.reduce_all(Q.greater(old, new))
        return Q.logical_and(Q.less(i, PS.tgt_len), Q.logical_not(old_better))

    def _grow_alive(self, state):
        """Grow alive sequences by one token, and collect top 2*beam_size sequences.
    2*beam_size sequences are collected because some sequences may have reached
    the EOS token. 2*beam_size ensures that at least beam_size sequences are
    still alive.
    Args:
      state: A dictionary with the current loop state.
    Returns:
      Tuple of
      (Top 2*beam_size sequences [batch_size, 2 * beam_size, cur_index + 1],
       Scores of returned sequences [batch_size, 2 * beam_size],
       New alive cache, for each of the 2 * beam_size sequences)
    """
        i = state[Keys.CUR_IDX]
        alive = state[Keys.ALIVE_SEQ]
        logps = state[Keys.ALIVE_LPS]
        cache = state[Keys.ALIVE_CACHE]

        beams_to_keep = 2 * PS.beam_size

        # Get logits for the next candidate IDs for the alive sequences. Get the new
        # cache values at the same time.
        flat_ids = flatten_beam_dim(alive)  # [batch_size * beam_size]
        flat_cache = nest.map_structure(_flatten_beam_dim, cache)

        flat_logits, flat_cache = self.symbols_to_logits_fn(
            flat_ids, i, flat_cache)

        # Unflatten logits to shape [batch_size, beam_size, vocab_size]
        logits = unflatten_beam_dim(flat_logits, PS.batch_size, PS.beam_size)
        new_cache = nest.map_structure(
            lambda t: unflatten_beam_dim(t, PS.batch_size, PS.beam_size),
            flat_cache)

        # Convert logits to normalized log probs
        candidate_log_probs = lps_from_logits(logits)

        # Calculate new log probabilities if each of the alive sequences were
        # extended # by the the candidate IDs.
        # Shape [batch_size, beam_size, vocab_size]
        log_probs = candidate_log_probs + tf.expand_dims(logps, axis=2)

        # Each batch item has beam_size * vocab_size candidate sequences. For each
        # batch item, get the k candidates with the highest log probabilities.
        flat_log_probs = tf.reshape(log_probs,
                                    [-1, PS.beam_size * self.vocab_size])
        topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs,
                                                   k=beams_to_keep)

        # Extract the alive sequences that generate the highest log probabilities
        # after being extended.
        topk_beam_indices = topk_indices // self.vocab_size
        topk_seq, new_cache = _gather_beams([alive, new_cache],
                                            topk_beam_indices, PS.batch_size,
                                            beams_to_keep)

        # Append the most probable IDs to the topk sequences
        topk_ids = topk_indices % self.vocab_size
        topk_ids = tf.expand_dims(topk_ids, axis=2)
        topk_seq = tf.concat([topk_seq, topk_ids], axis=2)
        return topk_seq, topk_log_probs, new_cache

    def _get_new_alive_state(self, new_seq, new_log_probs, new_cache):
        """Gather the top k sequences that are still alive.
    Args:
      new_seq: New sequences generated by growing the current alive sequences
        int32 tensor with shape [batch_size, 2 * beam_size, cur_index + 1]
      new_log_probs: Log probabilities of new sequences
        float32 tensor with shape [batch_size, beam_size]
      new_cache: Dict of cached values for each sequence.
    Returns:
      Dictionary with alive keys from Keys:
        {Top beam_size sequences that are still alive (don't end with eos_id)
         Log probabilities of top alive sequences
         Dict cache storing decoder states for top alive sequences}
    """
        # To prevent finished sequences from being considered, set log probs to -INF
        new_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
        new_log_probs += tf.to_float(new_flags) * -INF

        top_alive, top_logps, top_cache = _gather_topk_beams(
            [new_seq, new_log_probs, new_cache], new_log_probs, PS.batch_size,
            PS.beam_size)

        return {
            Keys.ALIVE_SEQ: top_alive,
            Keys.ALIVE_LPS: top_logps,
            Keys.ALIVE_CACHE: top_cache
        }

    def _get_new_finished_state(self, state, new_seq, new_log_probs):
        """Combine new and old finished sequences, and gather the top k sequences.
    Args:
      state: A dictionary with the current loop state.
      new_seq: New sequences generated by growing the current alive sequences
        int32 tensor with shape [batch_size, beam_size, i + 1]
      new_log_probs: Log probabilities of new sequences
        float32 tensor with shape [batch_size, beam_size]
    Returns:
      Dictionary with finished keys from Keys:
        {Top beam_size finished sequences based on score,
         Scores of finished sequences,
         Finished flags of finished sequences}
    """
        i = state[Keys.CUR_IDX]
        done = state[Keys.DONE_SEQ]
        scores = state[Keys.DONE_SCORES]
        flags = state[Keys.DONE_FLAGS]

        # First append a column of 0-ids to done to increment the length.
        # New shape of done: [batch_size, beam_size, i + 1]
        done = tf.concat(
            [done, Q.zeros([PS.batch_size, PS.beam_size, 1], Q.int32)], axis=2)

        # Calculate new seq scores from log probabilities.
        length_norm = self.len_penalty(i + 1)
        new_scores = new_log_probs / length_norm

        # Set the scores of the still-alive seq in new_seq to large negative values.
        new_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
        new_scores += (1. - tf.to_float(new_flags)) * -INF

        # Combine sequences, scores, and flags.
        done = tf.concat([done, new_seq], axis=1)
        scores = tf.concat([scores, new_scores], axis=1)
        flags = tf.concat([flags, new_flags], axis=1)

        # Return the finished sequences with the best scores.
        top_done, top_scores, top_flags = (_gather_topk_beams(
            [done, scores, flags], scores, PS.batch_size, PS.beam_size))

        return {
            Keys.DONE_SEQ: top_done,
            Keys.DONE_SCORES: top_scores,
            Keys.DONE_FLAGS: top_flags
        }


def sequence_beam_search(symbols_to_logits_fn, initial_ids, initial_cache,
                         vocab_size, beam_size, alpha, tgt_len, eos_id):
    """Search for sequence of subtoken ids with the largest probability.
  Args:
    symbols_to_logits_fn: A function that takes in ids, index, and cache as
      arguments. The passed in arguments will have shape:
        ids -> [batch_size * beam_size, index]
        index -> [] (scalar)
        cache -> nested dictionary of tensors [batch_size * beam_size, ...]
      The function must return logits and new cache.
        logits -> [batch * beam_size, vocab_size]
        new cache -> same shape/structure as inputted cache
    initial_ids: Starting ids for each batch item.
      int32 tensor with shape [batch_size]
    initial_cache: dict containing starting decoder variables information
    vocab_size: int size of tokens
    beam_size: int number of beams
    alpha: float defining the strength of length normalization
    tgt_len: maximum length to decoded sequence
    eos_id: int id of eos token, used to determine when a sequence has finished
  Returns:
    Top decoded sequences [batch_size, beam_size, tgt_len]
    sequence scores [batch_size, beam_size]
  """
    batch_size = tf.shape(initial_ids)[0]
    sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size,
                             beam_size, alpha, tgt_len, eos_id)
    return sbs.search(initial_ids, initial_cache)


def lps_from_logits(ls):
    return ls - Q.reduce_logsumexp(ls, axis=2, keep_dims=True)


def len_penalty(self, n):
    return Q.pow(((5. + Q.cast(n, Q.floatx())) / 6.), self.PS.alpha)


def _get_shape_keep_last_dim(tensor):
    shape_list = Q.int_shape(tensor)

    # Only the last
    for i in range(len(shape_list) - 1):
        shape_list[i] = None

    if isinstance(shape_list[-1], tf.Tensor):
        shape_list[-1] = None
    return tf.TensorShape(shape_list)


def flatten_beam_dim(x):
    sh = list(Q.int_shape(x))
    sh[0] *= sh[1]
    sh.pop(1)
    return Q.reshape(x, sh)


def unflatten_beam_dim(self, x):
    PS = self.PS
    sh = Q.int_shape(x)
    return Q.reshape(x, [PS.batch_size, PS.beam_size] + sh[1:])


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
    """Gather beams from nested structure of tensors.
  Each tensor in nested represents a batch of beams, where beam refers to a
  single search state (beam search involves searching through multiple states
  in parallel).
  This function is used to gather the top beams, specified by
  beam_indices, from the nested tensors.
  Args:
    nested: Nested structure (tensor, list, tuple or dict) containing tensors
      with shape [batch_size, beam_size, ...].
    beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each
     value in beam_indices must be between [0, beam_size), and are not
     necessarily unique.
    batch_size: int size of batch
    new_beam_size: int number of beams to be pulled from the nested tensors.
  Returns:
    Nested structure containing tensors with shape
      [batch_size, new_beam_size, ...]
  """
    # Computes the i'th coodinate that contains the batch index for gather_nd.
    # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..].
    batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])

    # Create coordinates to be passed to tf.gather_nd. Stacking creates a tensor
    # with shape [batch_size, beam_size, 2], where the last dimension contains
    # the (i, j) gathering coordinates.
    coordinates = tf.stack([batch_pos, beam_indices], axis=2)

    return nest.map_structure(lambda state: tf.gather_nd(state, coordinates),
                              nested)


def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
    """Gather top beams from nested structure."""
    _, topk_indexes = tf.nn.top_k(score_or_log_prob, k=beam_size)
    return _gather_beams(nested, topk_indexes, batch_size, beam_size)


# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7
"""
    toks = Q.identity(tgt)
    initial_ids = sos_id * Q.ones([PS.batch_size], dtype=Q.int32)
    decoded_ids, scores, cache = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        PS.beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(PS.top_beams == 1))
    if PS.top_beams == 1:
        decoded_ids = decoded_ids[:, 0, 1:]
        scores = scores[:, 0]
    else:
        decoded_ids = decoded_ids[:, :PS.top_beams, 1:]
        scores = scores[:, :PS.top_beams]
        return
"""


def get_state_shape_invariants(tensor):
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return Q.TensorShape(shape)


def compute_batch_indices(batch_size, beam_size):
    """Computes the i'th coordinate that contains the batch index for gathers.

  Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  batch the beam item is in. This will create the i of the i,j coordinate
  needed for the gather.

  Args:
    batch_size: Batch size
    beam_size: Size of the beam.
  Returns:
    batch_pos: [batch_size, beam_size] tensor of ids
  """
    batch_pos = Q.range(batch_size * beam_size) // beam_size
    batch_pos = Q.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos


def _create_make_unique(inputs):
    """Replaces the lower bits of each element with iota.

  The iota is used to derive the index, and also serves the purpose to
  make each element unique to break ties.

  Args:
    inputs: A tensor with rank of 2 and dtype of Q.float32.
      [batch_size, original_size].

  Returns:
    A tensor after element wise transformation, with dtype the same as inputs.
    [batch_size, original_size].

  Raises:
    ValueError: If the rank of the input tensor does not equal 2.
  """
    if inputs.shape.ndims != 2:
        raise ValueError("Input of top_k_with_unique must be rank-2 "
                         "but got: %s" % inputs.shape)

    height = inputs.shape[0]
    width = inputs.shape[1]
    zeros = Q.zeros([height, width], dtype=Q.int32)

    # Count_mask is used to mask away the low order bits to ensure that every
    # element is distinct.
    log2_ceiling = int(math.ceil(math.log(int(width), 2)))
    next_power_of_two = 1 << log2_ceiling
    count_mask = ~(next_power_of_two - 1)
    count_mask_r0 = Q.constant(count_mask)
    count_mask_r2 = Q.fill([height, width], count_mask_r0)

    # Smallest_normal is the bit representation of the smallest positive normal
    # floating point number. The sign is zero, exponent is one, and the fraction
    # is zero.
    smallest_normal = 1 << 23
    smallest_normal_r0 = Q.constant(smallest_normal, dtype=Q.int32)
    smallest_normal_r2 = Q.fill([height, width], smallest_normal_r0)

    # Low_bit_mask is used to mask away the sign bit when computing the absolute
    # value.
    low_bit_mask = ~(1 << 31)
    low_bit_mask_r0 = Q.constant(low_bit_mask, dtype=Q.int32)
    low_bit_mask_r2 = Q.fill([height, width], low_bit_mask_r0)

    iota = Q.tile(Q.expand_dims(Q.range(width, dtype=Q.int32), 0), [height, 1])

    # Compare the absolute value with positive zero to handle negative zero.
    input_r2 = Q.bitcast(inputs, Q.int32)
    abs_r2 = Q.bitwise_and(input_r2, low_bit_mask_r2)
    if_zero_r2 = Q.equal(abs_r2, zeros)
    smallest_normal_preserving_sign_r2 = Q.bitwise_or(input_r2,
                                                      smallest_normal_r2)
    input_no_zeros_r2 = Q.where(if_zero_r2, smallest_normal_preserving_sign_r2,
                                input_r2)

    # Discard the low-order bits and replace with iota.
    and_r2 = Q.bitwise_and(input_no_zeros_r2, count_mask_r2)
    or_r2 = Q.bitwise_or(and_r2, iota)
    return Q.bitcast(or_r2, Q.float32)


def _create_topk_unique(inputs, k):
    """Creates the top k values in sorted order with indices.

  Args:
    inputs: A tensor with rank of 2. [batch_size, original_size].
    k: An integer, number of top elements to select.

  Returns:
    topk_r2: A tensor, the k largest elements. [batch_size, k].
    topk_indices_r2: A tensor, indices of the top k values. [batch_size, k].
  """
    height = inputs.shape[0]
    width = inputs.shape[1]
    neg_inf_r0 = Q.constant(-np.inf, dtype=Q.float32)
    ones = Q.ones([height, width], dtype=Q.float32)
    neg_inf_r2 = ones * neg_inf_r0
    inputs = Q.where(Q.is_nan(inputs), neg_inf_r2, inputs)

    # Select the current largest value k times and keep them in topk_r2. The
    # selected largest values are marked as the smallest value to avoid being
    # selected again.
    tmp = inputs
    topk_r2 = Q.zeros([height, k], dtype=Q.float32)
    for i in range(k):
        kth_order_statistic = Q.reduce_max(tmp, axis=1, keepdims=True)
        k_mask = Q.tile(Q.expand_dims(Q.equal(Q.range(k), Q.fill([k], i)), 0),
                        [height, 1])
        topk_r2 = Q.where(k_mask, Q.tile(kth_order_statistic, [1, k]), topk_r2)
        ge_r2 = Q.greater_equal(inputs, Q.tile(kth_order_statistic,
                                               [1, width]))
        tmp = Q.where(ge_r2, neg_inf_r2, inputs)

    log2_ceiling = int(math.ceil(math.log(float(int(width)), 2)))
    next_power_of_two = 1 << log2_ceiling
    count_mask = next_power_of_two - 1
    mask_r0 = Q.constant(count_mask)
    mask_r2 = Q.fill([height, k], mask_r0)
    topk_r2_s32 = Q.bitcast(topk_r2, Q.int32)
    topk_indices_r2 = Q.bitwise_and(topk_r2_s32, mask_r2)
    return topk_r2, topk_indices_r2


def top_k_with_unique(inputs, k):
    """Finds the values and indices of the k largests entries.

  Instead of doing sort like tf.nn.top_k, this function finds the max value
  k times. The running time is proportional to k, which is be faster when k
  is small. The current implementation supports only inputs of rank 2.
  In addition, iota is used to replace the lower bits of each element, this
  makes the selection more stable when there are equal elements. The
  overhead is that output values are approximated.

  Args:
    inputs: A tensor with rank of 2. [batch_size, original_size].
    k: An integer, number of top elements to select.

  Returns:
    top_values: A tensor, the k largest elements in sorted order.
      [batch_size, k].
    indices: A tensor, indices of the top_values. [batch_size, k].
  """
    unique_inputs = _create_make_unique(Q.cast(inputs, Q.float32))
    top_values, indices = _create_topk_unique(unique_inputs, k)
    top_values = Q.cast(top_values, inputs.dtype)
    return top_values, indices


def compute_topk_scores_and_seq(sequences,
                                scores,
                                scores_to_gather,
                                flags,
                                beam_size,
                                batch_size,
                                prefix="default",
                                states_to_gather=None,
                                use_tpu=False,
                                use_top_k_with_unique=True):
    """Given sequences and scores, will gather the top k=beam size sequences.

  This function is used to grow alive, and finished. It takes sequences,
  scores, and flags, and returns the top k from sequences, scores_to_gather,
  and flags based on the values in scores.

  This method permits easy introspection using tfdbg.  It adds three named ops
  that are prefixed by `prefix`:
    - _topk_seq: the tensor for topk_seq returned by this method.
    - _topk_flags: the tensor for topk_flags returned by this method.
    - _topk_scores: the tensor for tokp_gathered_scores returned by this method.

  Args:
    sequences: Tensor of sequences that we need to gather from.
      [batch_size, beam_size, seq_length]
    scores: Tensor of scores for each sequence in sequences.
      [batch_size, beam_size]. We will use these to compute the topk.
    scores_to_gather: Tensor of scores for each sequence in sequences.
      [batch_size, beam_size]. We will return the gathered scores from here.
      Scores to gather is different from scores because for grow_alive, we will
      need to return log_probs, while for grow_finished, we will need to return
      the length penalized scores.
    flags: Tensor of bools for sequences that say whether a sequence has reached
      EOS or not
    beam_size: int
    batch_size: int
    prefix: string that will prefix unique names for the ops run.
    states_to_gather: dict (possibly nested) of decoding states.
    use_tpu: A bool, whether to compute topk scores and sequences on TPU.
    use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
      top_k during TPU beam search.

  Returns:
    Tuple of
    (topk_seq [batch_size, beam_size, decode_length],
     topk_gathered_scores [batch_size, beam_size],
     topk_flags[batch_size, beam_size])
  """
    _, topk_indexes = Q.top_k(scores, k=beam_size)
    # The next three steps are to create coordinates for tf.gather_nd to pull
    # out the topk sequences from sequences based on scores.
    # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    # batch the beam item is in. This will create the i of the i,j coordinate
    # needed for the gather
    batch_pos = compute_batch_indices(batch_size, beam_size)

    # top coordinates will give us the actual coordinates to do the gather.
    # stacking will create a tensor of dimension batch * beam * 2, where the
    # last dimension contains the i,j gathering coordinates.
    top_coordinates = Q.stack([batch_pos, topk_indexes], axis=2)

    # Gather up the highest scoring sequences.  For each operation added, give
    # it a concrete name to simplify observing these operations with tfdbg.
    # Clients can capture these tensors by watching these node names.
    def gather(tensor, name):
        return Q.gather_nd(tensor, top_coordinates, name=(prefix + name))

    topk_seq = gather(sequences, "_topk_seq")
    topk_flags = gather(flags, "_topk_flags")
    topk_gathered_scores = gather(scores_to_gather, "_topk_scores")
    if states_to_gather:
        topk_gathered_states = nest.map_structure(
            lambda state: gather(state, "_topk_states"), states_to_gather)
    else:
        topk_gathered_states = states_to_gather
    return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


def beam_search(symbols_to_logits_fn,
                initial_ids,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                states=None,
                eos_id=EOS_ID,
                stop_early=True,
                use_tpu=False,
                use_top_k_with_unique=True):
    """Beam search with length penalties.

  Requires a function that can take the currently decoded symbols and return
  the logits for the next symbol. The implementation is inspired by
  https://arxiv.org/abs/1609.08144.

  When running, the beam search steps can be visualized by using tfdbg to watch
  the operations generating the output ids for each beam step.  These operations
  have the pattern:
    (alive|finished)_topk_(seq,scores)

  Operations marked `alive` represent the new beam sequences that will be
  processed in the next step.  Operations marked `finished` represent the
  completed beam sequences, which may be padded with 0s if no beams finished.

  Operations marked `seq` store the full beam sequence for the time step.
  Operations marked `scores` store the sequence's final log scores.

  The beam search steps will be processed sequentially in order, so when
  capturing observed from these operations, tensors, clients can make
  assumptions about which step is being recorded.

  WARNING: Assumes 2nd dimension of tensors in `states` and not invariant, this
  means that the shape of the 2nd dimension of these tensors will not be
  available (i.e. set to None) inside symbols_to_logits_fn.

  Args:
    symbols_to_logits_fn: Interface to the model, to provide logits.
        Shoud take [batch_size, decoded_ids] and return [batch_size, vocab_size]
    initial_ids: Ids to start off the decoding, this will be the first thing
        handed to symbols_to_logits_fn (after expanding to beam size)
        [batch_size]
    beam_size: Size of the beam.
    decode_length: Number of steps to decode for.
    vocab_size: Size of the vocab, must equal the size of the logits returned by
        symbols_to_logits_fn
    alpha: alpha for length penalty.
    states: dict (possibly nested) of decoding states.
    eos_id: ID for end of sentence.
    stop_early: a boolean - stop once best sequence is provably determined.
    use_tpu: A bool, whether to do beam search on TPU.
    use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
      top_k during TPU beam search.

  Returns:
    Tuple of
    (decoded beams [batch_size, beam_size, decode_length]
     decoding probabilities [batch_size, beam_size])
  """
    batch_size = Q.int_shape(initial_ids)[0]

    # Assume initial_ids are prob 1.0
    initial_log_probs = Q.constant([[0.] + [-INF] * (beam_size - 1)])
    # Expand to beam_size (batch_size, beam_size)
    logps = Q.tile(initial_log_probs, [batch_size, 1])

    # Expand each batch and state to beam_size
    alive = inflate_beam(initial_ids, beam_size)
    alive = Q.expand_dims(alive, axis=2)  # (batch_size, beam_size, 1)
    if states:
        states = nest.map_structure(
            lambda state: inflate_beam(state, beam_size), states)
    else:
        states = {}

    # Finished will keep track of all the sequences that have finished so far
    # Finished log probs will be negative infinity in the beginning
    # flags will keep track of booleans
    done = Q.zeros(Q.int_shape(alive), Q.int32)
    # Setting the scores of the initial to negative infinity.
    scores = Q.ones([batch_size, beam_size]) * -INF
    flags = Q.zeros([batch_size, beam_size], Q.bool)

    def grow_finished(done, scores, flags, curr_seq, curr_scores,
                      curr_finished):
        """Given sequences and scores, will gather the top k=beam size sequences.

    Args:
      done: Current finished sequences.
        [batch_size, beam_size, current_decoded_length]
      scores: scores for each of these sequences.
        [batch_size, beam_size]
      flags: finished bools for each of these sequences.
        [batch_size, beam_size]
      curr_seq: current topk sequence that has been grown by one position.
        [batch_size, beam_size, current_decoded_length]
      curr_scores: scores for each of these sequences. [batch_size, beam_size]
      curr_finished: Finished flags for each of these sequences.
        [batch_size, beam_size]
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences)
    """
        # First append a column of 0'ids to finished to make the same length with
        # finished scores
        done = Q.concat(
            [done, Q.zeros([batch_size, beam_size, 1], Q.int32)], axis=2)

        # Set the scores of the unfinished seq in curr_seq to large negative
        # values
        curr_scores += (1. - Q.cast(curr_finished, Q.floatx())) * -INF
        # concatenating the sequences and scores along beam axis
        curr_done = Q.concat([done, curr_seq], axis=1)
        curr_scores = Q.concat([scores, curr_scores], axis=1)
        curr_flags = Q.concat([flags, curr_finished], axis=1)
        return compute_topk_scores_and_seq(
            curr_done,
            curr_scores,
            curr_scores,
            curr_flags,
            beam_size,
            batch_size,
            "grow_finished",
            use_tpu=use_tpu,
            use_top_k_with_unique=use_top_k_with_unique)

    def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished,
                   states):
        """Given sequences and scores, will gather the top k=beam size sequences.

    Args:
      curr_seq: current topk sequence that has been grown by one position.
        [batch_size, beam_size, i+1]
      curr_scores: scores for each of these sequences. [batch_size, beam_size]
      curr_log_probs: log probs for each of these sequences.
        [batch_size, beam_size]
      curr_finished: Finished flags for each of these sequences.
        [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences)
    """
        # Set the scores of the finished seq in curr_seq to large negative
        # values
        curr_scores += Q.cast(curr_finished, Q.floatx()) * -INF
        return compute_topk_scores_and_seq(curr_seq,
                                           curr_scores,
                                           curr_log_probs,
                                           curr_finished,
                                           beam_size,
                                           batch_size,
                                           "grow_alive",
                                           states,
                                           use_tpu=use_tpu)

    def grow_topk(i, alive, logps, states):
        r"""Inner beam search loop.

    This function takes the current alive sequences, and grows them to topk
    sequences where k = 2*beam. We use 2*beam because, we could have beam_size
    number of sequences that might hit <EOS> and there will be no alive
    sequences to continue. With 2*beam_size, this will not happen. This relies
    on the assumption the vocab size is > beam size. If this is true, we'll
    have at least beam_size non <EOS> extensions if we extract the next top
    2*beam words.
    Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
    https://arxiv.org/abs/1609.08144.

    Args:
      i: loop index
      alive: Topk sequences decoded so far [batch_size, beam_size, i+1]
      logps: probabilities of these sequences. [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.
    Returns:
      Tuple of
        (Topk sequences extended by the next word,
         The log probs of these sequences,
         The scores with length penalty of these sequences,
         Flags indicating which of these sequences have finished decoding,
         dict of transformed decoding states)
    """
        # Get the logits for all the possible next symbols
        flat_ids = Q.reshape(alive, [batch_size * beam_size, -1])

        # (batch_size * beam_size, decoded_length)
        if states:
            flat_states = nest.map_structure(_flatten_beam_dim, states)
            flat_logits, flat_states = symbols_to_logits_fn(
                flat_ids, i, flat_states)
            states = nest.map_structure(
                lambda t: unflatten_beam_dim(t, batch_size, beam_size),
                flat_states)
        else:
            flat_logits = symbols_to_logits_fn(flat_ids)

        logits = Q.reshape(flat_logits, [batch_size, beam_size, -1])

        # Convert logits to normalized log probs
        candidate_log_probs = common_layers.log_prob_from_logits(logits)

        # Multiply the probabilities by the current probabilities of the beam.
        # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
        log_probs = candidate_log_probs + Q.expand_dims(logps, axis=2)

        length_penalty = self.len_penalty(i + 1)

        curr_scores = log_probs / length_penalty
        # Flatten out (beam_size, vocab_size) probs in to a list of possibilities
        flat_curr_scores = Q.reshape(curr_scores, [-1, beam_size * vocab_size])

        topk_scores, topk_ids = Q.top_k(flat_curr_scores, k=beam_size * 2)

        # Recovering the log probs because we will need to send them back
        topk_log_probs = topk_scores * length_penalty

        # Work out what beam the top probs are in.
        topk_beam_index = topk_ids // vocab_size
        topk_ids %= vocab_size  # Unflatten the ids

        # The next three steps are to create coordinates for tf.gather_nd to pull
        # out the correct sequences from id's that we need to grow.
        # We will also use the coordinates to gather the booleans of the beam
        # items that survived.
        batch_pos = compute_batch_indices(batch_size, beam_size * 2)

        # top beams will give us the actual coordinates to do the gather.
        # stacking will create a tensor of dimension batch * beam * 2, where the
        # last dimension contains the i,j gathering coordinates.
        topk_coordinates = Q.stack([batch_pos, topk_beam_index], axis=2)

        # Gather up the most probable 2*beams both for the ids and
        # finished_in_alive bools
        topk_seq = Q.gather_nd(alive, topk_coordinates)
        if states:
            states = nest.map_structure(
                lambda state: Q.gather_nd(state, topk_coordinates), states)

        # Append the most probable alive
        topk_seq = Q.concat(
            [topk_seq, Q.expand_dims(topk_ids, axis=2)], axis=2)

        topk_finished = Q.equal(topk_ids, eos_id)

        return topk_seq, topk_log_probs, topk_scores, topk_finished, states

    def inner_loop(i, alive, logps, done, scores, flags, states):
        """Inner beam search loop.

    There are three groups of tensors, alive, finished, and topk.
    The alive group contains information about the current alive sequences
    The topk group contains information about alive + topk current decoded words
    the finished group contains information about finished sentences, that is,
    the ones that have decoded to <EOS>. These are what we return.
    The general beam search algorithm is as follows:
    While we haven't terminated (pls look at termination condition)
      1. Grow the current alive to get beam*2 topk sequences
      2. Among the topk, keep the top beam_size ones that haven't reached EOS
      into alive
      3. Among the topk, keep the top beam_size ones have reached EOS into
      finished
    Repeat
    To make things simple with using fixed size tensors, we will end
    up inserting unfinished sequences into finished in the beginning. To stop
    that we add -ve INF to the score of the unfinished sequence so that when a
    true finished sequence does appear, it will have a higher score than all the
    unfinished ones.

    Args:
      i: loop index
      alive: Topk sequences decoded so far [batch_size, beam_size, i+1]
      logps: probabilities of the beams. [batch_size, beam_size]
      done: Current finished sequences.
        [batch_size, beam_size, i+1]
      scores: scores for each of these sequences.
        [batch_size, beam_size]
      flags: finished bools for each of these sequences.
        [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.

    Returns:
      Tuple of
        (Incremented loop index
         New alive sequences,
         Log probs of the alive sequences,
         New finished sequences,
         Scores of the new finished sequences,
         Flags indicating which sequence in finished as reached EOS,
         dict of final decoding states)
    """

        # Each inner loop, we carry out three steps:
        # 1. Get the current topk items.
        # 2. Extract the ones that have finished and haven't finished
        # 3. Recompute the contents of finished based on scores.
        topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
            i, alive, logps, states)
        alive, logps, _, states = grow_alive(topk_seq, topk_scores,
                                             topk_log_probs, topk_finished,
                                             states)
        done, scores, flags, _ = grow_finished(done, scores, flags, topk_seq,
                                               topk_scores, topk_finished)

        return (i + 1, alive, logps, done, scores, flags, states)


    inner_shape = Q.TensorShape([None, None, None])
    state_struc = nest.map_structure(get_state_shape_invariants, states)
    (_, alive, logps, done, scores, flags, states) = Q.while_loop(
        _is_finished,
        inner_loop, [Q.constant(0), alive, logps, done, scores, flags, states],
        shape_invariants=[
            Q.TensorShape([]), inner_shape,
            logps.get_shape(), inner_shape,
            scores.get_shape(),
            flags.get_shape(), state_struc
        ],
        parallel_iterations=1,
        back_prop=False)

    alive.set_shape((None, beam_size, None))
    done.set_shape((None, beam_size, None))

    # Accounting for corner case: It's possible that no sequence in alive for a
    # particular batch item ever reached EOS. In that case, we should just copy
    # the contents of alive for that batch item. Q.reduce_any(flags, 1)
    # if 0, means that no sequence for that batch index had reached EOS. We need
    # to do the same for the scores as well.
    done = Q.where(Q.reduce_any(flags, 1), done, alive)
    scores = Q.where(Q.reduce_any(flags, 1), scores, logps)
    return done, scores, states
