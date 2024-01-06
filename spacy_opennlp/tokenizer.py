#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import logging
import re
from pathlib import Path
from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

__OPENNLP_BIN__ = "opennlp"


class OpenNLPTokenizer:
    def __init__(self, model_path):
        if not Path(model_path).exists():
            raise LookupError("OpenNLP model file is not set!")
        self._model_path = model_path

    def tokenize(self, sentences):
        if isinstance(sentences, list):
            _input = ""
            for sent in sentences:
                if isinstance(sent, list):
                    _input += " ".join((x for x in sent))
                else:
                    _input += " " + sent
            _input = _input.lstrip()
            _input += "\n"
        else:
            _input = sentences

        gc.collect()
        p = Popen(
            [__OPENNLP_BIN__, "TokenizerME", self._model_path],
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

        tokens = output.strip().split(" ")
        return tokens

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
