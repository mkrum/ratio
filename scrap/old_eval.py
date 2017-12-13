import random
from collections import Counter
from lstm import LSTM
import old_dh as dh
import old_onehot as oh

def evaluate(model, data, one):

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
            model.read(one.get_encoding(word))
            if word == '?':
                ans = answers[current_answer]

                occurances[ans] += 1
                current_answer += 1

                prediction_vector = model.softmax_answer().value()
                prediction = one.get_word(prediction_vector)

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
    data = dh.TaskData(1, 'english')
    one = oh.Onehot([data])
    model = LSTM(one.num_words)

    epochs = 1000
    stories = 3
    train_data = data.train_data[:stories]
    count = 0


    for epoch in range(epochs):
        epoch_loss = []

        j = 0
        total = 0.0
        correct = 0.0

        #randomly shuffle before each epoch
        random.shuffle(train_data)

        for story, answers in train_data:
            j += 1
            print('{}/{}'.format(j, stories), end='\r')

            current_answer = 0
            for word in story:
                model.read(one.get_encoding(word))
                count += 1

                if word == '?':
                    ans = answers[current_answer]
                    current_answer += 1

                    prediction_vector = model.softmax_answer().value()
                    prediction = one.get_word(prediction_vector)

                    epoch_loss.append(model.train_softmax(one.word_to_index[ans]))

                    total += 1.0
                    if ans == prediction:
                        correct += 1.0

            #story_loss = model.backprop()
            #epoch_loss.append(story_loss)

            model.reset()
        train_acc = correct/total

        loss = sum(epoch_loss) / len(epoch_loss)
        print('Epoch: {} Loss: {} Train Accuracy: {}'.format(epoch, loss, train_acc))

        if epoch % 100 == 0 and False:
            validation_rate = evaluate(model, data.valid_data, one)
            print('Validation Success Rate: {}'.format(validation_rate))
            model.save('saved_models/lstm/epoch-{}'.format(epoch))
    print(count)

    test_rate = evaluate(model, data.test_data, one)
    print('Test Success Rate: {}'.format(test_rate))


if __name__ == '__main__':
    main()
