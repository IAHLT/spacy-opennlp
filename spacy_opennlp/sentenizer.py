#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import logging
from pathlib import Path
from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

__OPENNLP_BIN__ = "opennlp"


class OpenNLPSentenizer:
    def __init__(self, model_path):
        if not Path(model_path).exists():
            raise LookupError("OpenNLP model file is not set!")
        self._model_path = model_path

    def _sentenize(self, sentences):
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
        line_buffered = 1
        p = Popen(
            [__OPENNLP_BIN__, "SentenceDetector", self._model_path],
            shell=False,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
            bufsize=line_buffered,
        )
        (stdout, stderr) = p.communicate(_input)

        if p.returncode != 0:
            raise OSError("OpenNLP command failed!")

        sentences = [l for l in stdout.splitlines() if l.strip()]
        return sentences

    def __call__(self, *args, **kwargs):
        return self._sentenize(*args, **kwargs)
