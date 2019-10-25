"""main_3a.py

An instance of the Text class should be initialized with a file path (a file or
directory). The example here uses object as the super class, but you may use
nltk.text.Text as the super class.

An instance of the Vocabulary class should be initialized with an instance of
Text (not nltk.text.Text).

"""
import os
import nltk
import re
import wn as wn
from nltk import word_tokenize
from nltk.corpus import PlaintextCorpusReader

from main import ENGLISH_VOCABULARY


class Text(object):
    STOPLIST = set(nltk.corpus.stopwords.words())
    def __init__(self, path):

        if os.path.isfile(path):
            fh = open(path, 'r');
            self.rawText = Text(word_tokenize(fh))

        elif os.path.isdir(path):
            corpus = PlaintextCorpusReader(path, '.*.mrg')
            self.rawText = Text(nltk.word_tokenize(corpus.raw()))


    def token_count(self):
        """Just return all tokens."""
        return len(self.rawText)

    def type_count(self):
        """Returns the type count, with minimal normalization by lower casing."""
        # an alternative would be to use the method nltk.text.Text.vocab()
        return len(set([w.lower() for w in self.rawText]))

    def sentence_count(self):
        """Return number of sentences, using the simplistic measure of counting period,
        exclamation marks and question marks."""
        return len([t for t in self.rawText if t in ('.', '!', '?')])


    def is_content_word(self, word):
        return word.lower() not in self.STOPLIST and word[0].isalpha()

    def most_frequent_content_words(self):
        """Return a list with the 25 most frequent content words and their
        frequencies. The list has (word, frequency) pairs and is ordered on the
        frequency."""
        dist = nltk.FreqDist([w for w in self.rawText if self.is_content_word(w)])
        return dist.most_common(n=25)

    def most_frequent_bigrams(self):
        """Return a list with the 25 most frequent bigrams that only contain
        content words. The list returned should have pairs where the first
        element in the pair is the bigram and the second the frequency, as in
        ((word1, word2), frequency), these should be ordered on frequency."""
        filtered_bigrams = [b for b in list(nltk.bigrams(self.rawText))
        if self.is_content_word(b[0]) and self.is_content_word(b[1])]
        dist = nltk.FreqDist([b for b in filtered_bigrams])
        return dist.most_common(n=25)



    def find_sirs(self):
        p = re.compile(r"Sir [\w|-]+")
        s_list = list(set(p.findall(self.raw))).sort()
        return s_list

    def find_brackets(self):
        list1 = re.findall(r"[(].*?[)]", self.raw)
        list2 = re.findall(r"[\[].*?[\]]", self.raw)
        return list1 + list2

    def find_roles(self):

        roles_list = re.findall(r"^(.*?): ", self.raw, re.MULTILINE)
        temp =[]
        for role in roles_list:
            if len(role) <= 5:
                temp.append(role)
            else:
                if role[:5] != 'SCENE':
                    temp.append(role)

        result = list(temp).sort();

        return result

    def find_repeated_words(self):

        roles_list = self.tokens
        count = 0
        res = []
        for index in range(len(roles_list)):
            if (index == 0) :
                count = count + 1
                continue
            if (roles_list[index] == roles_list[index-1]):
                count = count + 1
            else:
                if (count >= 3):
                    str = roles_list[index-1] + ' ' + roles_list[index-1] + ' ' + roles_list[index-1]
                    res.append(str)
                count = 1
        res = list(set(res)).sort()
        return res



class Vocabulary(object):
    ENGLISH_VOCABULARY = set(w.lower() for w in nltk.corpus.words.words())

    def __init__(self, text):
        self.text = text
        # keeping the unfiltered list around for statistics
        self.all_items = set([w.lower() for w in text])
        self.items = self.all_items.intersection(ENGLISH_VOCABULARY)
        # restricting the frequency dictionary to vocabulary items
        self.fdist = nltk.FreqDist(t.lower() for t in text if t.lower() in self.items)
        self.text_size = len(self.text)
        self.vocab_size = len(self.items)

    def __str__(self):
        return "<Vocabulary size=%d text_size=%d>" % (self.vocab_size, self.text_size)

    def __len__(self):
        return self.vocab_size

    def frequency(self, word):
        return self.fdist[word]

    def pos(self, word):
        # do not volunteer the pos for words not in the vocabulary
        if word not in self.items:
            return None
        synsets = wn.synsets(word)
        return synsets[0].pos() if synsets else 'n'

    def gloss(self, word):
        # do not volunteer the gloss (definition) for words not in the vocabulary
        if word not in self.items:
            return None
        synsets = wn.synsets(word)
        # make a difference between None for words not in vocabulary and words
        # in the vocabulary that do not have a gloss in WordNet
        return synsets[0].definition() if synsets else 'NO DEFINITION'

    def kwic(self, word):
        self.text.concordance(word)

