
import framework as fw
from collections import Counter
import string
import utils as ut

class Baseline(object):

    def __init__(self):
        self.answer_count = Counter()

    def train(self, question, answers):

        for answer in answers:
            answer = ut.remove_punc(answer)
            self.answer_count[answer] += 1

    def answer(self, questions):
        
        answers = []
        for question in questions:
            question_statement = question[-1]
            
            simmilarity = []
            ans = []
            for line in question[:-1]:

                simmilarity.append(0)

                max_ans = None
                highest_count = -1
                for word in line.split(' '):
                    
                    word = ut.remove_punc(word)
                    if word in question_statement:
                        simmilarity[-1] += 1

                    if self.answer_count[word] >= highest_count:
                        max_ans = word
                        highest_count = self.answer_count[word]

                ans.append(max_ans)

            answer = ans[simmilarity.index(max(simmilarity))]

            if answer is None:
                answer = self.answer_count.most_common(1)[0]

            answers.append(answer)
        return answers

    def print_metrics(self):
        print(self.answer_count.most_common(10))
