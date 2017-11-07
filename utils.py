
import string

def remove_punc(raw_string):
    translator = str.maketrans('', '', string.punctuation)
    return raw_string.translate(translator)


