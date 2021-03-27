import json
import logging
from typing import Dict, Any
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer

from debias_finetuning.readers.frontend.frontend_reader import FrontEndReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("debias_finetuning_classic")
class FineTuningNLIReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 frontend_reader:str = None,
                 frontend_args: Dict[str, Any] = {},
                 lazy: bool = False,
                 concatenate_instances: str = None,
                 concatenate_frontend_reader: str = None,
                 concatenate_frontend_args: Dict[str, Any] = None,
                 sentence1_name: str = "hypothesis",
                 sentence2_name: str = "premise",
                 **kwargs) -> None:
        super().__init__(lazy, **kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}
        self._frontend = FrontEndReader.by_name(frontend_reader)(self, **frontend_args)
        self._concatenate_instances = concatenate_instances

        if self._concatenate_instances is not None and concatenate_frontend_reader is not None:
            self._concatenate_frontend = FrontEndReader.by_name(concatenate_frontend_reader)(self, **concatenate_frontend_args)

        self._sentence1_name = sentence1_name
        self._sentence2_name = sentence2_name

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as f:
            for instance in self._frontend.read(f):
                yield self.text_to_instance(**instance)

        if self._concatenate_instances is not None:
            with open(cached_path(self._concatenate_instances), "r") as f:
                for instance in self._concatenate_frontend.read(f):
                    yield self.text_to_instance(**instance)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence1: str,
                         sentence2: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ

        sentence1_toks = self._tokenizer.tokenize(sentence1)
        sentence2_toks = self._tokenizer.tokenize(sentence2)

        fields: Dict[str, Field] = {
            self._sentence1_name: TextField(sentence1_toks, self._token_indexers),
            self._sentence2_name: TextField(sentence2_toks, self._token_indexers)
        }

        if label:
            fields['label'] = LabelField(label)

        metadata = {f"{self._sentence1_name}_tokens": [x.text for x in sentence1_toks],
                    f"{self._sentence2_name}_tokens": [x.text for x in sentence2_toks]}

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
