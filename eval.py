import random
from collections import Counter
import numpy as np
from lstm import LSTM
import datahandling as dh
import embeddings as emb

def evaluate(model, data, embedding):

    predictions = Counter()
    occurances = Counter()
    correct_predicitons = Counter()

    test_len = len(data)
    j = 0
    total = 0.0
    correct = 0.0
    for story, answers in data:
        j += 1
        print('{}/{}'.format(j, test_len), end='\r')

        current_answer = 0
        for word in story:
            model.read(embedding[word])
            if word == '?':
                ans = answers[current_answer]

                occurances[ans] += 1
                current_answer += 1

                prediction_vector = np.array(model.answer().value())
                prediction = embedding.wv.most_similar(positive=[prediction_vector],
                                                       negative=[])[0][0]

                predictions[prediction] += 1

                total += 1.0
                if ans == prediction:
                    correct_predicitons[prediction] += 1
                    correct += 1.0

        model.reset()

    print('{: >15} {: >15} {: >15} {: >15}'.format('Prediction', 'Count', 'Actual',
                                                   'Accuracy'))
    total = sum(occurances.values())
    for word, n in predictions.most_common(10):

        accuracy = round(100 * correct_predicitons[word] / n, 2)
        print('{: >15} {: >15} {: >15} {: >15}'.format(word, n, occurances[word], accuracy))

    succes_rate = correct/total
    return succes_rate

def main():
    model = LSTM()
    data = dh.TaskData(1, 'english')
    embedding = emb.load_embedding(1, 'english')

    #set train length to 1000
    train_len = 1500

    for epoch in range(1000):
        epoch_loss = []

        j = 0
        total = 0.0
        correct = 0.0

        #randomly shuffle before each epoch
        random.shuffle(data.train_data)

        for story, answers in data.train_data[:train_len]:
            j += 1
            print('{}/{}'.format(j, train_len), end='\r')

            current_answer = 0
            for word in story:
                model.read(embedding[word])

                if word == '?':
                    ans = answers[current_answer]
                    current_answer += 1

                    prediction_vector = np.array(model.answer().value())
                    prediction = embedding.wv.most_similar(positive=[prediction_vector],
                                                           negative=[])[0][0]

                    epoch_loss.append(model.train(embedding[ans]))

                    total += 1.0
                    if ans == prediction:
                        correct += 1.0

            #story_loss = model.backprop()
            #epoch_loss.append(story_loss)

            model.reset()

        train_acc = correct/total

        loss = sum(epoch_loss) / len(epoch_loss)
        print('Epoch: {} Loss: {} Train Accuracy: {}'.format(epoch, loss, train_acc))

        validation_rate = evaluate(model, data.valid_data, embedding)
        print('Validation Success Rate: {}'.format(validation_rate))
        model.save('saved_models/rnn/epoch-{}'.format(epoch))

    test_rate = evaluate(model, data.test_data, embedding)
    print('Test Success Rate: {}'.format(test_rate))


if __name__ == '__main__':
    main()
