class Onehot(object):

    def __init__(self, tasks_data):
        self.words = set()
        self.word_to_index = {}
        self.index_to_word = {}

        for td in tasks_data:
            for data in [td.train_data, td.valid_data, td.test_data]:
                for story, answers in data:
                    for word in story:
                        self.words.add(word)
                    for ans in answers:
                        self.words.add(ans)

            for index, word in enumerate(self.words):
                self.word_to_index[word] = index
                self.index_to_word[index] = word

        self.num_words = len(self.words)


    def get_encoding(self, word):
        encoding = [0] * self.num_words
        index = self.word_to_index[word]
        encoding[index] = 1
        return encoding


    def get_word(self, encoding):
        max_value = max(encoding)
        index = encoding.index(max_value)
        return self.index_to_word[index]
