import dynet as dy
from lstm import LSTM

class RNN(LSTM):

    def __init__(self, input_dimension):

        NUM_LAYERS = 2
        HIDDEN_DIM = 100
        #FLAT_HIDDEN = 64

        INPUT_DIM = input_dimension

        self.pc = dy.ParameterCollection()
        self.builder = dy.SimpleRNNBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, self.pc)


        self.current_state = self.builder.initial_state()
        self.loss_buffer = []

        self.params = {}

        self.params['W_1'] = self.pc.add_parameters((INPUT_DIM, HIDDEN_DIM))

        self.params['bias_1'] = self.pc.add_parameters((INPUT_DIM))

        self.params['input_dim'] = input_dimension
        self.reset()

        self.trainer = dy.AdamTrainer(self.pc)
