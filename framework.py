
import inspect

class Model(object):

    def answer(self, question):
        raise Exception(inspect.stack()[0][3]+" not implemented")

    def train(self, question, answer):
        raise exception(inspect.stack()[0][3]+" not implemented")

    def print_metrics(self):
        pass
