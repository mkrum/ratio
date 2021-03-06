import sys
import string
import glob
import os
import gensim

data_path = 'data/'

def load_embedding(tasknumber, language):
    return gensim.models.Word2Vec.load('embeddings/{}/qa{}'.format(language, tasknumber))

def vocab_size(tasknumber, language):
    model = gensim.models.Word2Vec.load('embeddings/{}/qa{}'.format(language, tasknumber))
    return len(model.wv.vocab)

def tokenize(line):
    line = line.split()[1:]

    words = []
    for word in line:
        if len(word) > 0 and word[-1] in string.punctuation:
            words += [word[:-1], word[-1]]
        else:
            words += [word]

    return words


def create_embedding(language, tasknumber):

    files = glob.glob('{}/{}/qa{}_*.txt'.format(data_path, language, tasknumber))
    lines = sum(list(map(lambda x: open(x, 'r').read().splitlines(), files)), [])

    sentences = []
    for line in lines:
        sentences.append(tokenize(line))

    model = gensim.models.Word2Vec(sentences, size=10, window=5, workers=4, min_count=0)
    model.save('embeddings/{}/qa{}'.format(language, tasknumber))



if __name__ == '__main__':

    lang_class = [d for d in os.listdir(data_path) if os.path.isdir(data_path + d)]

    for lang in lang_class:
        os.mkdir('embeddings/' + lang)
        for i in range(1, 21):
            create_embedding(lang, i)
