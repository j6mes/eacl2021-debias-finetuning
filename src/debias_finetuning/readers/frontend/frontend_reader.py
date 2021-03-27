from abc import ABC

from allennlp.common import Registrable


class FrontEndReader(Registrable, ABC):

    def __init__(self, reader):
        self._reader = reader

    def read(self, file):
        raise NotImplementedError("Not implemented here")
