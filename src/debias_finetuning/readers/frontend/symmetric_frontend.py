import json

from debias_finetuning.readers.frontend.frontend_reader import FrontEndReader


@FrontEndReader.register("symmetric")
class SymmetricFrontEndReader(FrontEndReader):

    def __init__(self, reader):
        self._reader = reader

    def read(self, file):
        for line in file:
            example = json.loads(line)
            label = example["label"] if "label" in example else example["gold_label"]
            evidence = example["evidence"]
            claim = example["claim"]

            yield self.text_to_instance(unpreprocess(evidence), claim, label)
