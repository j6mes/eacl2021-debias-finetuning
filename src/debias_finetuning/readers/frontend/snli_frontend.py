import json

from debias_finetuning.readers.frontend.frontend_reader import FrontEndReader


@FrontEndReader.register("snli")
class SNLIFrontEndReader(FrontEndReader):

    def __init__(self, reader):
        self._reader = reader

    def read(self, file):
        for line in file:
            example = json.loads(line)
            label = example["label"] if "label" in example else example["gold_label"]
            premise = example["sentence1"]
            hypothesis = example["sentence2"]

            if label == "-":
                continue

            yield {
                "sentence1": hypothesis,
                "sentence2": premise,
                "label": label
            }
