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
# https://arxiv.org/pdf/1701.06548.pdf
# https://arxiv.org/pdf/1607.06450.pdf
# https://arxiv.org/pdf/1606.08415.pdf

import qnarre.neura as Q
import qnarre.neura.utils as U
import qnarre.neura.layers as L

from qnarre.feeds.dset.trafo import dset as dset


def dset_for(PS, kind):
    ds = dset(PS, kind)
    n = 1000
    ds = ds.take(n)
    if kind == 'train':
        ds = ds.shuffle(n)
    ds = ds.batch(PS.batch_size)
    return ds


def model_for(PS, full=False):
    ctx = Q.Input(shape=(PS.ctx_len, ), dtype='int32')
    typ = Q.Input(shape=(PS.ctx_len, ), dtype='int32')
    tgt = Q.Input(shape=(PS.tgt_len, ), dtype='int32')
    ins = [ctx, typ, tgt]
    y = L.Trafo(PS)(ins)
    m = Q.Model(inputs=ins, outputs=[y])
    if full:
        m.compile(optimizer=PS.optimizer, loss=PS.losses, metrics=[PS.metrics])
    return m


params = dict(
    batch_size=4,
    ctx_len=16,
    ffn_act='gelu',
    hidden_act='gelu',
    hidden_size=8,
    max_pos=0,
    tgt_len=0,
    token_types=8,
    vocab_size=20,
)

params.update(
    dir_data='.data/trafo',
    log_dir='.model/trafo/logs',
    dir_model='.model/trafo',
    dir_save='.model/trafo/save',
)


def main(_):
    PS = U.Params(params).init_comps()
    model = model_for(PS)

    @Q.function
    def train_step(x, y):
        with Q.GradientTape() as tape:
            logits = model(x)
            loss = PS.losses(y, logits)
            acc = PS.metrics(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        PS.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, acc

    def train():
        step, loss, acc = 0, 0.0, 0.0
        for x, y in dset_for(PS, 'train'):
            step += 1
            loss, acc = train_step(x, y)
            if Q.equal(step % 10, 0):
                m = PS.metrics.result()
                Q.print('Step:', step, ', loss:', loss, ', acc:', m)
        return step, loss, acc

    step, loss, acc = train()
    print('Final step:', step, ', loss:', loss, ', acc:', PS.metrics.result())


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    from absl import flags as F
    F.DEFINE_integer('src_len', None, '')
    from absl import app
    app.run(main)

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models.research import universal_transformer_util
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class UniversalTransformer(transformer.Transformer):
    """Universal Transformer: Depth-wise recurrent transformer model."""

    def encode(self, inputs, target_space, hparams, features=None,
               losses=None):
        """Encode Universal Transformer inputs.

    It is similar to "transformer.encode", but it uses
    "universal_transformer_util.universal_transformer_encoder" instead of
    "transformer.transformer_encoder".

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: Unused.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
        del losses

        inputs = common_layers.flatten4d3d(inputs)

        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            transformer.transformer_prepare_encoder(inputs,
                                                    target_space,
                                                    hparams,
                                                    features=features))

        encoder_input = tf.nn.dropout(
            encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)

        (encoder_output, encoder_extra_output) = (
            universal_transformer_util.universal_transformer_encoder(
                encoder_input,
                self_attention_bias,
                hparams,
                nonpadding=transformer.features_to_nonpadding(
                    features, "inputs"),
                save_weights_to=self.attention_weights))

        return encoder_output, encoder_decoder_attention_bias, encoder_extra_output

    def decode(self,
               decoder_input,
               encoder_output,
               encoder_decoder_attention_bias,
               decoder_self_attention_bias,
               hparams,
               cache=None,
               decode_loop_step=None,
               nonpadding=None,
               losses=None):
        """Decode Universal Transformer outputs from encoder representation.

    It is similar to "transformer.decode", but it uses
    "universal_transformer_util.universal_transformer_decoder" instead of
    "transformer.transformer_decoder".

    Args:
      decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
        hidden_dim]
      encoder_output: Encoder representation. [batch_size, input_length,
        hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
        attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
        self-attention. [batch_size, decoder_length]
      hparams: hyperparmeters for model.
      cache: Unimplemented.
      decode_loop_step: Unused.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]
      losses: Unused.

    Returns:
       Tuple of:
         Final decoder representation. [batch_size, decoder_length,
            hidden_dim]
         encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)

    """
        del decode_loop_step
        del losses
        # TODO(dehghani): enable caching.
        del cache

        decoder_input = tf.nn.dropout(
            decoder_input, 1.0 - hparams.layer_prepostprocess_dropout)

        # No caching in Universal Transformers!
        (decoder_output, dec_extra_output) = (
            universal_transformer_util.universal_transformer_decoder(
                decoder_input,
                encoder_output,
                decoder_self_attention_bias,
                encoder_decoder_attention_bias,
                hparams,
                nonpadding=nonpadding,
                save_weights_to=self.attention_weights))

        # Expand since t2t expects 4d tensors.
        return tf.expand_dims(decoder_output, axis=2), dec_extra_output

    def body(self, features):
        """Universal Transformer main model_fn.


    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
        hparams = self._hparams
        if hparams.add_position_timing_signal:
            # Turning off addition of positional embedding in the encoder/decoder
            # preparation as we do it in the beginning of each step.
            hparams.pos = None

        if self.has_input:
            inputs = features["inputs"]
            target_space = features["target_space_id"]
            (encoder_output, encoder_decoder_attention_bias,
             enc_extra_output) = self.encode(inputs,
                                             target_space,
                                             hparams,
                                             features=features)
        else:
            (encoder_output, encoder_decoder_attention_bias,
             enc_extra_output) = (None, None, (None, None))

        targets = features["targets"]
        targets = common_layers.flatten4d3d(targets)

        (decoder_input, decoder_self_attention_bias
         ) = transformer.transformer_prepare_decoder(targets,
                                                     hparams,
                                                     features=features)

        decoder_output, dec_extra_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=transformer.features_to_nonpadding(features, "targets"))

        expected_attentions = features.get("expected_attentions")
        if expected_attentions is not None:
            attention_loss = common_attention.encoder_decoder_attention_loss(
                expected_attentions, self.attention_weights,
                hparams.expected_attention_loss_type,
                hparams.expected_attention_loss_multiplier)
            return decoder_output, {"attention_loss": attention_loss}

        if hparams.recurrence_type == "act" and hparams.act_loss_weight != 0:
            if self.has_input:
                enc_ponder_times, enc_remainders = enc_extra_output
                enc_act_loss = (
                    hparams.act_loss_weight *
                    tf.reduce_mean(enc_ponder_times + enc_remainders))
            else:
                enc_act_loss = 0.0

            (dec_ponder_times, dec_remainders) = dec_extra_output
            dec_act_loss = (hparams.act_loss_weight *
                            tf.reduce_mean(dec_ponder_times + dec_remainders))
            act_loss = enc_act_loss + dec_act_loss
            tf.contrib.summary.scalar("act_loss", act_loss)
            return decoder_output, {"act_loss": act_loss}

        return decoder_output

    def _greedy_infer(self, features, decode_length, use_tpu=False):
        """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: bool, whether to use the TPU codepath.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
        if use_tpu:
            return self._slow_greedy_infer_tpu(features, decode_length)
        return self._slow_greedy_infer(features, decode_length)

    def _beam_decode(self,
                     features,
                     decode_length,
                     beam_size,
                     top_beams,
                     alpha,
                     use_tpu=False):
        """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: Whether we should use TPU or not.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
        # Caching is not ebabled in Universal Transformer
        # TODO(dehghani): Support fast decoding for Universal Transformer
        return self._beam_decode_slow(features, decode_length, beam_size,
                                      top_beams, alpha, use_tpu)


@registry.register_model
class UniversalTransformerEncoder(transformer.Transformer):
    """Universal Transformer Encoder: Has no decoder (e.g.for classification)."""

    def encode(self, inputs, target_space, hparams, features=None,
               losses=None):
        """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: Unused.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
        del losses
        inputs = common_layers.flatten4d3d(inputs)

        (encoder_input, self_attention_bias,
         _) = (transformer.transformer_prepare_encoder(inputs, target_space,
                                                       hparams))

        encoder_input = tf.nn.dropout(
            encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)

        (encoder_output, encoder_extra_output) = (
            universal_transformer_util.universal_transformer_encoder(
                encoder_input,
                self_attention_bias,
                hparams,
                nonpadding=transformer.features_to_nonpadding(
                    features, "inputs"),
                save_weights_to=self.attention_weights))

        return encoder_output, encoder_extra_output

    def body(self, features):
        """Universal Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
        hparams = self._hparams

        assert self.has_input, (
            "universal_transformer_encoder is applicable on "
            "problems with inputs")

        inputs = features["inputs"]
        target_space = features["target_space_id"]
        encoder_output, enc_extra_output = self.encode(inputs,
                                                       target_space,
                                                       hparams,
                                                       features=features)

        encoder_output = tf.expand_dims(encoder_output, 2)

        if hparams.recurrence_type == "act" and hparams.act_loss_weight != 0:
            ponder_times, remainders = enc_extra_output
            act_loss = hparams.act_loss_weight * tf.reduce_mean(ponder_times +
                                                                remainders)
            tf.contrib.summary.scalar("act_loss", act_loss)

            return encoder_output, {"act_loss": act_loss}
        return encoder_output
