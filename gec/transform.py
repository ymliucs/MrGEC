# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import Levenshtein
from typing import Iterable, List, Optional, Union

from supar.utils import Field
from supar.utils.transform import Sentence, Transform

from .field import MrField


class Text(Transform):
    fields = ['SRC', 'TGT']

    def __init__(
        self,
        SRC: Optional[Union[Field, Iterable[Field]]] = None,
        TGT=None,
    ) -> Text:
        super().__init__()

        self.SRC = SRC
        self.TGT = TGT

    @property
    def src(self):
        return (self.SRC,)

    @property
    def tgt(self):
        return (self.TGT,)

    def load(self, data: Union[str, Iterable], ref: str = 'cs', mode: str = 'train', **kwargs) -> Iterable[TextSentence]:
        r"""
        Loads the data in Text-X format.
        Also supports for loading data from Text-U file with comments and non-integer IDs.

        Args:
            data (str or Iterable): the path to the data file.

            ref (str): choices=['cs', 'mld', 'mr']
                the way to process Multi-reference data, copy sample num(refs) times,
                select the ref with the minimum levenshtein distance from src, keep Multi-refs for one src

            mode (str): train or predict

        Returns:
            A list of :class:`TextSentence` instances.
        """

        f = open(data)
        index, sentence = 0, []
        if mode == 'train':
            if ref == 'ss':
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        src = sentence[0]
                        for tgt in sentence[1:]:
                            sentence = TextSentence(self, [src, tgt], index)
                            yield sentence
                            index += 1
                        sentence = []
                    else:
                        sentence.append(line)
            elif ref == 'mlr':
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        src = sentence[0]
                        max_r = -math.inf
                        for tgt in sentence[1:]:
                            d = Levenshtein.distance(src.split("\t")[1], tgt.split("\t")[1])
                            r = (len(src.split("\t")[1]) + len(tgt.split("\t")[1]) - d) / (len(src.split("\t")[1]) + len(tgt.split("\t")[1]))
                            if r > max_r:
                                best_tgt = tgt
                                max_r = r
                        sentence = TextSentence(self, [src, best_tgt], index)
                        yield sentence
                        index += 1
                        sentence = []
                    else:
                        sentence.append(line)
            elif ref == 'mr':
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sentence = TextSentence(self, sentence, index)
                        yield sentence
                        index += 1
                        sentence = []
                    else:
                        sentence.append(line)
            else:
                assert False, "ref must in ['ss', 'mlr', 'mr']"
        else:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    # sentence = TextSentence(self, sentence[:1] + ['T'], index)
                    sentence = TextSentence(self, sentence, index)
                    yield sentence
                    index += 1
                    sentence = []
                else:
                    sentence.append(line)


class TextSentence(Sentence):
    def __init__(self, transform: Text, lines: List[str], index: Optional[int] = None) -> TextSentence:
        super().__init__(transform, index)

        self.cands = [(line + '\t').split('\t')[1] for line in lines[1:]]
        self.values = [lines[0].split('\t')[1]] + self.cands
        if len(self.values) > 2:
            self.maps['tgt'] = slice(1, len(self.values), 1)

    def __repr__(self):
        self.cands = self.values[1] if isinstance(self.values[1], list) else self.values[1:]
        lines = ['S\t' + self.values[0]]
        lines.extend(['T\t' + i for i in self.cands])
        return '\n'.join(lines) + '\n'

    def numericalize(self, fields):
        for f in fields:
            sequences = getattr(self, f.name)
            if not isinstance(sequences, list):
                sequences = [sequences]
            self.fields[f.name] = next(f.transform(sequences))
        self.pad_index = fields[0].pad_index
        return self
