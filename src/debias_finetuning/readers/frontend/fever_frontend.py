import json
import random
from typing import List, Tuple

from drqa.retriever.utils import normalize

from debias_finetuning.readers.fever_database import FEVERDocumentDatabase
from debias_finetuning.readers.frontend.frontend_reader import FrontEndReader
from debias_finetuning.readers.preprocessing import flatten, format_line, unpreprocess, get_non_empty_lines


@FrontEndReader.register("fever")
class FEVERFrontEndReader(FrontEndReader):

    def __init__(self, reader, database, format_evidence=True, ignore_nei=False):
        self._reader = reader
        self._database = FEVERDocumentDatabase(database)
        self._ignore_nei = ignore_nei
        self._format_evidence = format_evidence

    def get_doc_lines(self, page_title: str) -> List[str]:
        return [line.split("\t")[1].strip() if len(line.split("\t"))>1 else "" for line in
                self._database.get_doc_lines(page_title).split("\n")]

    def get_doc_line(self, page_title: str, line_number: int) -> str:
        if line_number is None:
            raise Exception("It looks like an NEI page is being loaded, but no evidence is present\nIt looks like you need to sample some negative evidence for NEI claims")
        doc_lines = self.get_doc_lines(page_title)
        if line_number >= 0:
            return doc_lines[line_number]
        else:
            return random.sample(get_non_empty_lines(doc_lines),1)[0]

    def read(self, file):
        for line in file:
            instance = json.loads(line)

            claim_id: int = instance['id']
            claim: str = instance['claim']
            evidence: List[List[Tuple[int, int, str, int]]] = instance['evidence']
            evidence: List[List[Tuple[str, int]]] = [[(item[2], item[3]) for item in group] for group in evidence]

            label: str = instance['label'] if 'label' in instance else None

            yield from self.read_instances(claim_id, evidence, claim, label)

    def read_instances(self, claim_id, evidence, claim, label):

        if self._ignore_nei and label == "NOT ENOUGH INFO":
            return []

        gold = set()

        for item in set(flatten(evidence)):
            gold.add((item[0], item[1]))

        for item in gold:
            line = self.get_doc_line(item[0], item[1])

            if len(line.split(" ")) > 120:
                continue

            yield {
                "sentence1": claim,
                "sentence2": self.maybe_format_line(item[0], line),
                "label": label
            }

    def maybe_format_line(self, page, evidence_text):
        if self._format_evidence:
            return format_line(page, evidence_text)
        else:
            return normalize(unpreprocess(evidence_text))