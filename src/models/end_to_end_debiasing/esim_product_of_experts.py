from typing import Dict, Optional, Any, List

from allennlp.models import BasicClassifier, ESIM
from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, MatrixAttention
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, masked_max, get_final_encoder_states
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("esim_classifier_poe")
class ShallowProductOfExpertsClassifier(ESIM):
    """
    This `Model` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    a linear classification layer, which projects into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.

    Registered as a `Model` with name "basic_classifier".

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = None).
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace : `str`, optional (default = "labels")
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(self, vocab: Vocabulary,
                 beta: float,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 matrix_attention: MatrixAttention,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 hypothesis_only_feedforward: FeedForward,
                 output_logit: FeedForward,
                 dropout: float = 0.5,
                 evaluation_mode: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:

        super().__init__(vocab, text_field_embedder, encoder,
                         matrix_attention, projection_feedforward, inference_encoder,
                         output_feedforward, output_logit, dropout, initializer)

        self._classification_layer_hyp_only = hypothesis_only_feedforward
        self._beta = beta

        self._accuracy = CategoricalAccuracy()
        self._hyp_only_accuracy = CategoricalAccuracy()
        self._nll_loss = torch.nn.NLLLoss()
        self._cross_ent_loss = torch.nn.CrossEntropyLoss()
        self.evaluation_mode = evaluation_mode
        initializer(self)

    def finetune(self):
        self.evaluation_mode = True

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise)
        hypothesis_mask = get_text_field_mask(hypothesis)

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_premise = self.rnn_input_dropout(embedded_premise)
            embedded_hypothesis = self.rnn_input_dropout(embedded_hypothesis)

        # encode premise and hypothesis
        encoded_premise = self._encoder(embedded_premise, premise_mask)
        encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        output_dict = self.esim_forward(encoded_premise, encoded_hypothesis, premise_mask, hypothesis_mask, label=label)

        # If we're training, also compute loss and accuracy for the bias-only model
        if not self.evaluation_mode:
            hyp_only_logits = self._classification_layer_hyp_only(get_final_encoder_states(encoded_hypothesis, hypothesis_mask,self._encoder.is_bidirectional()))

            log_probs_pair = torch.log_softmax(output_dict["label_logits"], dim=1)
            log_probs_hyp = torch.log_softmax(hyp_only_logits, dim=1)

            # Combine with product of experts (normalized log space sum)
            # Do not require gradients from hyp-only classifier
            combined = log_probs_pair + log_probs_hyp.detach()

            # NLL loss over combined labels
            loss = self._nll_loss(combined, label.long().view(-1))
            hyp_loss = self._nll_loss(log_probs_hyp, label.long().view(-1))
            self._accuracy(combined, label)
            self._hyp_only_accuracy(hyp_only_logits, label)

            output_dict = {
                "loss": loss + self._beta * hyp_loss
            }
            return output_dict
        else:
            loss = self._cross_ent_loss(output_dict["label_logits"],label)
            output_dict["loss"] = loss
            self._accuracy(output_dict["label_logits"], label)

            return output_dict


    def esim_forward(  # type: ignore
        self,
        encoded_premise, encoded_hypothesis, premise_mask, hypothesis_mask,
        label: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(encoded_premise, h2p_attention)

        # the "enhancement" layer
        premise_enhanced = torch.cat(
            [
                encoded_premise,
                attended_hypothesis,
                encoded_premise - attended_hypothesis,
                encoded_premise * attended_hypothesis,
            ],
            dim=-1,
        )
        hypothesis_enhanced = torch.cat(
            [
                encoded_hypothesis,
                attended_premise,
                encoded_hypothesis - attended_premise,
                encoded_hypothesis * attended_premise,
            ],
            dim=-1,
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
        projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
        v_ai = self._inference_encoder(projected_enhanced_premise, premise_mask)
        v_bi = self._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max = masked_max(v_ai, premise_mask.unsqueeze(-1), dim=1)
        v_b_max = masked_max(v_bi, hypothesis_mask.unsqueeze(-1), dim=1)

        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1) / torch.sum(
            premise_mask, 1, keepdim=True
        )
        v_b_avg = torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1) / torch.sum(
            hypothesis_mask, 1, keepdim=True
        )

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        output_hidden = self._output_feedforward(v_all)
        label_logits = self._output_logit(output_hidden)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        return output_dict


    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
                   "hyp_only_accuracy": self._hyp_only_accuracy.get_metric(reset),
                   "accuracy": self._accuracy.get_metric(reset)
                   }
        return metrics