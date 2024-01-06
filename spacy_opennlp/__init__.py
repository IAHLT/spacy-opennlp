#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from spacy_opennlp.version import __version__  # noqa: F401
import os
import re
from typing import Optional

import spacy
from spacy import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import registry
from spacy.vocab import Vocab

from spacy_opennlp.lemmatizer import OpenNLPLemmatizer
from spacy_opennlp.pos_tagger import OpenNLPTagger
from spacy_opennlp.sentenizer import OpenNLPSentenizer
from spacy_opennlp.tokenizer import OpenNLPTokenizer

Doc.set_extension("opennlp_tag", default=None, force=True)
Token.set_extension("opennlp_tag", default=None, force=True)
Span.set_extension("opennlp_tag", default=None, force=True)


def load(lang: str, path: Optional[str] = None, **kwargs) -> Language:
    config = {"nlp": {"tokenizer": {}}}
    name = lang
    config["nlp"]["tokenizer"]["@tokenizers"] = "spacy_opennlp.PipelineAsTokenizer.v1"  # noqa: E501
    config["nlp"]["tokenizer"]["lang"] = lang
    config["nlp"]["tokenizer"]["path"] = path
    for key, value in kwargs.items():
        config["nlp"]["tokenizer"][key] = value
    return spacy.blank(name, config=config)


@registry.tokenizers("spacy_opennlp.PipelineAsTokenizer.v1")
def create_tokenizer(lang: str, path: Optional[str] = None):
    def tokenizer_factory(nlp, lang=lang, path=path, **kwargs) -> "OpenNLPPipeline":
        return OpenNLPPipeline(lang=lang, path=path, vocab=nlp.vocab)

    return tokenizer_factory


class OpenNLPPipeline:
    def __init__(self, path: str, vocab: Vocab, lang: str):
        self.lang = lang
        self.pos_tagger = OpenNLPTagger(model_path=os.path.join(path, f"{lang}-pos.bin"))
        self.lemmatizer = OpenNLPLemmatizer(
            model_path=os.path.join(path, f"{lang}-lemmatizer.bin")
        )
        self.tokenizer = OpenNLPTokenizer(model_path=os.path.join(path, f"{lang}-token.bin"))
        self.sentenizer = OpenNLPSentenizer(
            model_path=os.path.join(path, f"{lang}-sent.bin")
        )
        self.vocab = vocab
        self._ws_pattern = re.compile(r"\s+")

    def __call__(self, text):
        if not text:
            return Doc(self.vocab)
        elif text.isspace():
            return Doc(self.vocab, words=[text], spaces=[False])

        sentences = self.sentenizer(text)

        pos = []
        lemmas = []
        spaces = []
        words = []
        sent_starts = []

        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            tokens_with_pos = self.pos_tagger(tokens)
            pos += [t[-1] for t in tokens_with_pos]
            lemmas += self.lemmatizer(tokens_with_pos)
            spaces += [True] * len(tokens)
            words += tokens
            sent_starts += [True] + [False] * (len(tokens) - 1)

        doc = Doc(
            self.vocab,
            words=words,
            spaces=spaces,
            lemmas=lemmas,
            sent_starts=sent_starts,
        )

        for token, pos in zip(doc, pos):
            token._.opennlp_tag = pos
        return doc

    def pipe(self, texts):
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        YIELDS (Doc): A sequence of Doc objects, in order.
        """
        for text in texts:
            yield self(text)

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes_data, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self

