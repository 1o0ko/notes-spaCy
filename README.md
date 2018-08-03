#  :fire: [spaCy](https://spacy.io/) :fire:

What is spaCy? it's a blazingly-fast, free, open-source library for NLP in python :snake:. It was born out of frustration with **NLTK** and **CoreNLP**. Read [Dead Code Should Be Buried](https://explosion.ai/blog/dead-code-should-be-buried) if you want to read more.

> Almost all examples are either from [spaCy's documentation](https://spacy.io/usage/) of from [ml reference](http://mlreference.com/spacy)



## Installation

Use pip to install the package and then download appropriate language model and word vectors

```Bash
$ pip install -U spacy
$ python -m spacy download en
$ python -m spacy download en_core_web_md
```

There is a [variety](https://spacy.io/usage/models) of different models in differnt languages such as English, German, Spanish or French. 



## Tokenization

```python
document_string = "Miss Phd has a really nice title, doesn't she?"
[token for token in nlp(document_string)]
```

```bash
# [Miss, Phd, has, a, really, nice, title, ,, does, n't, she, ?]
```



## Part-of-speech tagging and lemmas

``` python
import spacy

# Load the language model and parse your document.
nlp = spacy.load('en')
doc = nlp("I won't use this tired example with fox and lazy dog. Will I?")

# Print out all parts of speech and lemmas.
for token in doc:
    print('%-14s' * 5 % (token, token.norm_, token.pos_, token.tag_, token.lemma_))
```

```bash
# I             i             PRON          PRP           -PRON-
# wo            will          VERB          MD            will
# n't           not           ADV           RB            not
# use           use           VERB          VB            use
# this          this          DET           DT            this
# tired         tired         ADJ           JJ            tire
# example       example       NOUN          NN            example
# with          with          ADP           IN            with
# fox           fox           NOUN          NN            fox
# and           and           CCONJ         CC            and
# lazy          lazy          ADJ           JJ            lazy
# dog           dog           NOUN          NN            dog
# .             .             PUNCT         .             .
# Will          will          VERB          MD            will
# I             i             PRON          PRP           -PRON-
# ?             ?             PUNCT         .             ?	
```



## Noun chunks

```python
document = """\
New York is a main location of a best selling novel: "I <3 NYC".
"""
nlp = spacy.load('en')
doc = nlp(document)

# Print all noun chunks.
# These are contiguous noun phrases.
for i, chunk in enumerate(doc.noun_chunks, 1):
    print(f"{i}) {chunk}")
```

```bash
# 1) New York
# 2) a main location
# 3) a best selling novel
# 4) "I <3 NYC
```





## Sentence boundaries

```Python
document = """\
This is a document with many sentences. To make it more iteresting I will throw some puncation here, like Mr. Stokowiec or M.D.! But this isn't my last word!
"""

nlp = spacy.load('en')
doc = nlp(document)

for i, sentence in enumerate(doc.sents, 1):
    print(f"A sentence {i}: {sentence}")
```

```bash
# A sentence 1: This is a document with many sentences.
# A sentence 2: To make it more iteresting I will throw some puncation here, like Mr. Stokowiec or M.D.!
# A sentence 3: But this isn't my last word!
```



## Named Entities 

```python
document = "I like to visit Element AI in London."

nlp = spacy.load('en')
doc = nlp(document)

for ent in doc.ents:
     print('%-14s' * 4 % (ent.text, ent.start_char, ent.end_char, ent.label_))
```

```bash
# Example with start and end positions of the NE from the original substring

# Element AI    16            26            ORG
# London        30            36            GPE
```



## Character and token categories

```python
document = """\
I visit beaglepie.com about 1,000 times a day.
I often email myself at hi@elementai.com to remember things.
At 4:00pm I paid $8.75 for an indulgently fancy coffee.
"""

nlp = spacy.load('en')
doc = nlp(document)

# Extract all emails, URLs, and numbers from document_string.
emails  = [token for token in doc if token.like_email]
urls    = [token for token in doc if token.like_url]
numbers = [token for token in doc if token.like_num]

print(f"E-mails: {emails}")
print(f"URLs: {urls}")
print(f"numbers: {numbers}")
```

```bash
# E-mails: [hi@elementai.com]
# URLs: [beaglepie.com, hi@elementai.com]
# numbers: [1,000, 8.75]
```

A more comprehensive example

```python	
# spacy can recognize these kinds of tokens:
categories = [
    'alpha',     # Letters (in any language).
    'digit',     # Digits like 0-9 or ১২৩ (Bengali digits).
    'lower',     # lower case like this.
    'upper',     # UPPER CASE LIKE THIS.
    'title',     # Title Case Like This.
    'punct',     # Punctuation marks.
    'space',     # All-whitespace tokens.
    'bracket',   # Brackets like [ or ].
    'quote',     # Quotation marks.
    'stop'       # Stop words.
]

document = """A word in Russian, such as «Привет» is still
understood in terms of character classes. Numbers [like £300]
can be recognized as well."""

doc = nlp(document)

skip_and_print('Categories from this string:\n%s' % sample2)

for category in categories:
    print(f"{category}:")
    print([
        token.text
        for token in doc2
        if getattr(token, 'is_' + category)
    ])
```



```bash
# Categories from this string:
# A word in Russian, such as «Привет» is still
# understood in terms of character classes. Numbers [like £300]
# can be recognized as well.
# alpha:
# ['A', 'word', 'in', 'Russian', 'such', 'as', 'Привет', 'is', 'still', 'understood', 'in', 'terms', 'of', 'character', 'classes', 'Numbers', 'like', 'can', 'be', 'recognized', 'as', 'well']

# digit:
# ['300']

# lower:
# ['word', 'in', 'such', 'as', 'is', 'still', 'understood', 'in', 'terms', 'of', 'character', 'classes', 'like', 'can', 'be', 'recognized', 'as', 'well']

# upper:
# ['A']

# title:
# ['A', 'Russian', 'Привет', 'Numbers']

# punct:
# [',', '«', '»', '.', '[', ']', '.']

# space:
# ['\n', '\n']

# bracket:
# ['[', ']']

# quote:
# ['«', '»']

# stop:
# ['in', 'such', 'as', 'is', 'still', 'in', 'of', 'can', 'be', 'as', 'well']
```





## Word vectors

```python
from itertools import product

tokens = nlp(u'dog cat banana')

for token1, token2 in product(tokens, tokens):
    print('%-14s' * 3 % (token1.text, token2.text, token1.similarity(token2)))
```

```bash
# dog           dog           1.00000
# dog           cat           0.53907
# dog           banana        0.28761
# cat           dog           0.53907
# cat           cat           1.00000
# cat           banana        0.48752
# banana        dog           0.28761
# banana        cat           0.48752
# banana        banana        1.00000
```

```python
nlp = spacy.load('en_core_web_md')
tokens = nlp(u'dog cat banana Stokowiec')

for token in tokens:
    print('%-14s' * 4 % (
        token.text, token.has_vector, token.vector_norm, token.is_oov
    ))
```

```Bash
# dog           True          7.03367       False
# cat           True          6.68082       False
# banana        True          6.70001       False
# Stokowiec     False         0.0           True
```





# Real-life example

``` python
import spacy
import numpy as np

from collections import defaultdict

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def tokenize(text, nlp):
    return [token.lemma_ for token in nlp(text) if not token.is_stop]


class MeanEmbeddingVectorizer(object):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = nlp.vocab.vectors.shape[1]

    def fit(self, X, y):
        return self

    def transform(self, X):
        ''' X: list of texts
        '''
        # spacy will default to vector of zeros if word has no embedding
        return np.array([
            np.mean([token.vector 
                     for token in self.nlp(text) 
                     if not token.is_stop], axis=0)
            for text in X
        ])
    
class TfidfEmbeddingVectorizer(object):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = nlp.vocab.vectors.shape[1]
        self.word2weight = None

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda text: tokenize(text, self.nlp))
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
            return np.array([
                np.mean([token.vector * self.word2weight[token.lemma_]
                         for token in self.nlp(text)
                         if not token.is_stop], axis=0)
            for text in X
        ]) 
        
if __name__ == '__main__':
	# load language model
    nlp = spacy.load('en_core_web_md')

    # build two pipelines
    trees_on_w2v = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(nlp)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    
    trees_on_w2v_tfidf = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(nlp)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    
    X = ["This is awsome", "This sucks", "I don't know how to feel about this"]
    y = ["positive", "negative" , "neutral"]
    
    # test on un-seen data: super
    trees_on_w2v.fit(X, y);
    print(trees_on_w2v.predict(["This is super"]))
    
    trees_on_w2v_tfidf.fit(X, y);
    print(trees_on_w2v_tfidf.predict(["This is super"]))
```

```bash
> ['positive']
> ['positive']
```



## Dependency parsing

```python
document = """\
Autonomous cars shift insurance liability toward manufacturers
"""

nlp = spacy.load('en')
doc = nlp(document)

for chunk in doc.noun_chunks:
    print('%-25s' * 4 % (
        chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text))
```

```bash
# Autonomous cars          cars                     nsubj                    shift                    
# insurance liability      liability                dobj                     shift                    
# manufacturers            manufacturers            pobj                     toward     
```



````python
document = """\
Autonomous cars shift insurance liability toward manufacturers
"""

nlp = spacy.load('en')
doc = nlp(document)

for token in doc:
    print('%-15s' * 5 % (
        token.text, token.dep_, token.head.text, token.head.pos_,
        [child for child in token.children]))
````

```bash
# Autonomous     amod           cars           NOUN           []             
# cars           nsubj          shift          VERB           [Autonomous]   
# shift          ROOT           shift          VERB           [cars, liability, toward]
# insurance      compound       liability      NOUN           []             
# liability      dobj           shift          VERB           [insurance]    
# toward         prep           shift          VERB           [manufacturers]
# manufacturers  pobj           toward         ADP            []      
```

## Cool stuff

```Python
''' Build-in dependency visualizer'''
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers")
displacy.render(doc, style='dep', jupyter=True)
```

##  More Examples

* Visit [the official documentaiton](https://spacy.io/usage/examples) for cool examples!
  * Custom tagging
  * Text classification 
  * … and more! :bomb::boom:
* Be sure to checkout [thinc](https://github.com/explosion/thinc).
