import dynet as dy
import datahandling as dh
import embeddings as emb
import numpy as np

class LSTM(object):

    def __init__(self):

        NUM_LAYERS = 4
        INPUT_DIM = 100
        HIDDEN_DIM = 256

        WORD_VEC_DIM = 100

        self.pc = dy.ParameterCollection()
        self.builder = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, self.pc)

        self.input_word = dy.vecInput(100)

        self.current_state = self.builder.initial_state()
        self.loss_buffer = []

        self.params = {}
        self.params["W"] = self.pc.add_parameters((WORD_VEC_DIM, HIDDEN_DIM))
        self.params["bias"] = self.pc.add_parameters((WORD_VEC_DIM))

        self.W = dy.parameter(self.params["W"])
        self.bias = dy.parameter(self.params["bias"])

        self.input_word = dy.vecInput(100)
        self.actual_word = dy.vecInput(100)

        self.trainer = dy.SimpleSGDTrainer(self.pc)

    def read(self, word_vector):
        self.input_word.set(word_vector)
        self.current_state = self.current_state.add_input(self.input_word)

    def flush(self):

        total_loss = dy.esum(self.loss_buffer)

        total_loss.backward()
        self.trainer.update()
        loss_val = total_loss.value()
        self.loss_buffer = []

        dy.renew_cg()
        self.current_state = self.builder.initial_state()
        self.input_word = dy.vecInput(100)
        self.actual_word = dy.vecInput(100)
        self.W = dy.parameter(self.params["W"])
        self.bias = dy.parameter(self.params["bias"])

        return loss_val


    def answer(self):
        return dy.softmax(self.W*self.current_state.output() + self.bias)

    def train(self, actual):
        prediction = self.answer()
        self.actual_word.set(actual)
        self.loss_buffer.append( dy.huber_distance(prediction, self.actual_word))
    

if __name__ == '__main__':

    model = LSTM()
    data = dh.TaskData(1, 'english')
    embedding = emb.load_embedding(1, 'english')  
    
    train_len = len(data.train_data)

    for epoch in range(10):
        epoch_loss = []
        
        j = 0
        for story, answers in data.train_data:
            j += 1
            print('{}/{}'.format(j, train_len), end='\r')
            
            current_answer = 0
            for word in story:
                model.read(embedding[word])

                if word == '?':
                    ans = answers[current_answer]
                    current_answer += 1
                    model.train(embedding[ans])

            epoch_loss.append(model.flush())

        loss = sum(epoch_loss) / len(epoch_loss)
        print('Epoch: {} Loss: {}'.format(epoch, loss))

    
    test_len = len(data.test_data)
    total = 0.0
    correct = 0.0
    j = 0
    for story, answers in data.test_data:
        print('{}/{}'.format(j, test_len), end='\r')
        j += 1
        current_answer = 0
        for word in story:
            model.read(embedding[word])

            if word == '?':
                prediction_vector = np.array(model.answer().value())
                prediction = embedding.wv.most_similar(positive=[prediction_vector], negative=[])[0][0]

                ans = answers[current_answer]
                current_answer += 1

                total += 1.0
                print('{} {}'.format(ans, prediction))
                if ans == prediction:
                    correct += 1.0

    print(correct/total)


