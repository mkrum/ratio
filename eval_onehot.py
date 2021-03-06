import os
import sys
import random
from collections import Counter
from lstm import LSTM
from rnn import RNN
import datahandling as dh
import onehot as oh


def evaluate(model, data, one):
    print('evaluating')
    predictions = Counter()
    occurances = Counter()
    correct_predicitons = Counter()

    test_len = len(data) if len(data) > 20 else len(data[0])
    j = 0
    total = 0.0
    correct = 0.0
    task_accuracies = []
    for task in data:
        task_total = 0.0
        task_correct = 0.0
        for story, answers in task:
            j += 1
            print('{}/{}'.format(j, test_len), end='\r')

            current_answer = 0
            for line in story:
                for word in line:
                    model.read(one.get_encoding(word))
                    if word == '?':
                        ans = answers[current_answer]

                        occurances[ans] += 1
                        current_answer += 1

                        prediction_vector = model.softmax_answer().value()
                        prediction = one.get_word(prediction_vector)

                        predictions[prediction] += 1

                        total += 1.0
                        task_total += 1.0
                        if ans == prediction:
                            correct_predicitons[prediction] += 1
                            correct += 1.0
                            task_correct += 1.0
            model.reset()
        task_accuracies.append(task_correct/task_total)

    print('{: >15} {: >15} {: >15} {: >15}'.format('Prediction', 'Count', 'Actual',
                                                   'Accuracy'))
    total = sum(occurances.values())
    for word, n in predictions.most_common(10):

        accuracy = round(100 * correct_predicitons[word] / n, 2)
        print('{: >15} {: >15} {: >15} {: >15}'.format(word, n, occurances[word], accuracy))

    succes_rate = correct/total
    return succes_rate, task_accuracies

def main():
    task, language, model_type = sys.argv[1:]

    data = dh.TaskData(task, language)
    one = oh.Onehot([data])

    if model_type == 'lstm':
        model = LSTM(one.num_words)
    elif model_type == 'rnn':
        model = RNN(one.num_words)

    save_path = 'saved_models/{}-{}-{}'.format(model_type, language, task)
    results_path = 'results/{}-{}-{}.txt'.format(model_type, language, task)

    epochs = 50
    stories = 1500
    state = None

    best_val = 0

    for epoch in range(epochs):
        epoch_loss = []

        j = 0
        total = 0.0
        correct = 0.0

        #randomly shuffle before each epoch
        random.shuffle(data.train_data)

        for story, answers in data.train_data[:stories]:
            current_answer = 0
            j += 1
            print('{}/{}'.format(j, stories), end='\r')
            for line in story:

                if '?' in line:
                    state = model.current_state

                for word in line:
                    model.read(one.get_encoding(word))
                    if word == '?':
                        ans = answers[current_answer]
                        current_answer += 1

                        prediction_vector = model.softmax_answer().value()
                        prediction = one.get_word(prediction_vector)

                        epoch_loss.append(model.train_softmax(one.word_to_index[ans]))

                        model.current_state = state

                        total += 1.0
                        if ans == prediction:
                            correct += 1.0

                #story_loss = model.backprop()
                #epoch_loss.append(story_loss)

            model.reset()

        train_acc = correct/total

        loss = sum(epoch_loss) / len(epoch_loss)
        print('Epoch: {} Loss: {} Train Accuracy: {}'.format(epoch, loss, train_acc))

        validation_rate, _ = evaluate(model, [data.valid_data], one)

        if validation_rate > best_val:
            model.save(save_path)
            best_val = validation_rate

        print('Validation Success Rate: {}'.format(validation_rate))

    model.load(save_path)
    test_rate, task_accuracies = evaluate(model, data.test_data_tasks, one)
    print('Test Success Rate Combined: {}'.format(test_rate))
    for i, acc in enumerate(task_accuracies):
        print('Test Success Rate Task {}: {}'.format(i+1, acc))

    with open(results_path, 'w') as res_file:
        res_file.write('{}\n{}'.format(best_val, test_rate))

if __name__ == '__main__':
    main()
