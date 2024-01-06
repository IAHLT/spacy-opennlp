#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import logging
import re
from pathlib import Path
from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

__OPENNLP_BIN__ = "opennlp"


class OpenNLPLemmatizer:
    def __init__(self, model_path):
        if not Path(model_path).exists():
            raise LookupError("OpenNLP model file is not set!")
        self.model_path = model_path

    def lemmatize(self, tagged_tokens):
        _input = ""
        for token_tag in tagged_tokens:
            _input += token_tag[0] + "_" + token_tag[1] + " "

        gc.collect()
        p = Popen(
            [__OPENNLP_BIN__, "LemmatizerME", self.model_path],
            shell=False,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )
        (stdout, stderr) = p.communicate(_input)

        if p.returncode != 0:
            raise OSError("OpenNLP command failed!")

        output = re.sub(r"\nExecution time:(.*)$", "", stdout)

        lemmas = []
        for line in output.strip().split("\n"):
            words = line.split("\t")
            lemmas.append(words[-1])
        return lemmas

    def __call__(self, *args, **kwargs):
        return self.lemmatize(*args, **kwargs)
