

## Spacy wrapper for OpenNLP models


This package provides a very simple wrapper for the OpenNLP models for [spaCy](https://spacy.io/) which is working through command line calls to the OpenNLP tools. It is not very efficient, so the main purpose check and test existing OpenNLP models.

### Installation

Install opennlp with apt-get or brew:

```bash
brew install opennlp # on Mac
apt-get install opennlp # on Linux
```
For Windows, you can download the binaries from [here](https://opennlp.apache.org/download.html).
OpenNLP requires Java, so make sure you have Java installed. Then add the path to the OpenNLP binaries to your PATH environment variable.
Check if OpenNLP is installed correctly by running the following command:

```bash
opennlp
```



Install the package with pip:

```bash
pip install spacy-opennlp
```

### Usage

The usage is very simple. You can load the models with the `load` function,
where `path` is the path to the OpenNLP models (expected 4 models, e.g. for Hebrew: `he-lemmatizer.bin`, `he-pos.bin`, `he-sent.bin`, `he-token.bin`),
and `lang` is the language code (e.g. `he` for Hebrew, so package will look for `{lang}-lemmatizer.bin`, `{lang}-pos.bin`, `{lang}-sent.bin`, `{lang}-token.bin` in the `path` directory).

```python
import spacy_opennlp


nlp = spacy_opennlp.load(
    "he",
    path="<PATH TO MODELS THERE>",
)
text = "בתל אביב יש כמה רכבת קלה."
doc = nlp(text)
for sentence in doc.sents:
    print(sentence.text)
    
    for token in sentence:
        print(token.text, token.lemma_, token._.opennlp_tag)
    print()
```

