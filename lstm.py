import dynet as dy

class LSTM(object):

    def __init__(self, input_dimension):

        NUM_LAYERS = 2
        HIDDEN_DIM = 100
        #FLAT_HIDDEN = 64

        INPUT_DIM = input_dimension

        self.pc = dy.ParameterCollection()
        self.builder = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, self.pc)
        #self.builder = dy.VanillaLSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, self.pc)


        self.current_state = self.builder.initial_state()
        self.loss_buffer = []

        self.params = {}

        self.params['W_1'] = self.pc.add_parameters((INPUT_DIM, HIDDEN_DIM))
        #self.params['W_1'] = self.pc.add_parameters((FLAT_HIDDEN, HIDDEN_DIM))
        #self.params['W_2'] = self.pc.add_parameters((WORD_VEC_DIM, FLAT_HIDDEN))

        self.params['bias_1'] = self.pc.add_parameters((INPUT_DIM))
        #self.params['bias_1'] = self.pc.add_parameters((FLAT_HIDDEN))
        #self.params['bias_2'] = self.pc.add_parameters((WORD_VEC_DIM))

        self.params['input_dim'] = input_dimension
        self.reset()

        self.trainer = dy.AdamTrainer(self.pc)

    def save(self, path):
        self.pc.save(path)

    def load(self, path):
        self.pc.populate(path)

    def read(self, word_vector):
        self.input_words.append(dy.vecInput(self.params['input_dim']))
        self.input_words[-1].set(word_vector)
        self.current_state = self.current_state.add_input(self.input_words[-1])

    def backprop(self):
        total_loss = dy.esum(self.loss_buffer)

        loss_val = total_loss.value()

        total_loss.backward()
        self.trainer.update()

        self.loss_buffer = []
        return loss_val

    def reset(self):
        dy.renew_cg()
        self.current_state = self.builder.initial_state()
        self.input_words = []
        self.actual_word = dy.vecInput(self.params['input_dim'])

        self.W_1 = dy.parameter(self.params['W_1'])
        self.bias_1 = dy.parameter(self.params['bias_1'])

    def answer(self):
        #return self.W_2 * (self.W_1 * self.current_state.output() + self.bias_1) + self.bias_2
        return (self.W_1 * self.current_state.output()) + self.bias_1

    def softmax_answer(self):
        return dy.softmax(self.answer())

    def train(self, actual):
        prediction = self.answer()

        self.actual_word.set(actual)
        loss = dy.squared_distance(prediction, self.actual_word)

        loss.backward()
        self.trainer.update()

        return loss.value()

    def train_softmax(self, actual):
        prediction = self.softmax_answer()

        #loss = -dy.log(dy.pick(prediction, actual))
        loss = -dy.log(prediction[actual])

        loss.backward()
        self.trainer.update()

        return loss.value()

    def batch_size(self):
        return len(self.loss_buffer)
