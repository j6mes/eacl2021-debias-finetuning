from typing import Dict, Optional

from allennlp.models import BasicClassifier
from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("bert_classifier_dfl")
class ShallowDebiasedFocalLossClassifer(Model):
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

    def __init__(
        self,
        vocab: Vocabulary,
        beta: float,
        gamma: float,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        feedforward_hyp_only: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        evaluation_mode: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self.evaluation_mode = evaluation_mode
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        self._feedforward_hyp_only = feedforward_hyp_only

        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if feedforward_hyp_only is not None:
            self._classifier_hyp_only_input_dim = self._feedforward_hyp_only.get_output_dim()
        else:
            self._classifier_hyp_only_input_dim = self._seq2vec_encoder.get_output_dim()


        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._classification_layer_hyp_only = torch.nn.Linear(self._classifier_hyp_only_input_dim, self._num_labels)

        self._beta = beta
        self._gamma = gamma

        self._accuracy = CategoricalAccuracy()
        self._hyp_only_accuracy = CategoricalAccuracy()
        self._element_cross_ent_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self._cross_ent_loss = torch.nn.CrossEntropyLoss()
        initializer(self)


    def finetune(self):
        self.evaluation_mode = True

    def forward(  # type: ignore
        self,
            tokens: TextFieldTensors,
            bias_tokens: TextFieldTensors = None,
            label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        sentence_pair_logits = self._classification_layer(embedded_text)


        # If we're training, also compute loss and accuracy for the bias-only model
        if not self.evaluation_mode and bias_tokens is not None:
            # Make predictions with hypothesis only
            embedded_text = self._text_field_embedder(bias_tokens)
            mask = get_text_field_mask(bias_tokens)

            if self._seq2seq_encoder:
                embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)
            embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

            if self._dropout:
                embedded_text = self._dropout(embedded_text)

            if self._feedforward_hyp_only is not None:
                embedded_text = self._feedforward_hyp_only(embedded_text)
            hyp_only_logits = self._classification_layer_hyp_only(embedded_text)
            hyp_only_probs = torch.softmax(hyp_only_logits,dim=1)

            scaled = (1.0 - hyp_only_probs).pow(self._gamma).detach()

            weighting = torch.cat([scaled[idx, val].unsqueeze(0) for idx, val in enumerate([l.item() for l in label])])
            instance_losses = self._element_cross_ent_loss(sentence_pair_logits,label)

            hyp_loss = self._cross_ent_loss(hyp_only_logits, label.long().view(-1))

            self._accuracy(sentence_pair_logits, label)
            self._hyp_only_accuracy(hyp_only_logits, label)
            output_dict = {
                "loss": (instance_losses * weighting).mean() + self._beta*hyp_loss,
                "logits": sentence_pair_logits,

            }

            return output_dict

        else:
            loss = self._cross_ent_loss(sentence_pair_logits,label)
            self._accuracy(sentence_pair_logits, label)
            return {"loss": loss, "logits":sentence_pair_logits, "probs": torch.softmax(sentence_pair_logits,dim=1) }


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