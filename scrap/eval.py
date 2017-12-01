
import sys
from datahandling import TaskData
from models.baseline import Baseline

model = Baseline()

task = sys.argv[1]
data = TaskData(task)
train = data.get_train()

n_epochs = 1

for _ in range(n_epochs):
    for story, answers in data.get_train():
        model.train(story, answers)

model.print_metrics()

total = 0.0
n_correct = 0.0
for story, answers in data.get_test():
    predictions = model.answer(story)
    for prediction, answer in zip(predictions, answers):
        total += 1.0
        print('{} {}'.format(prediction, answer))

        if prediction == answer:
            n_correct += 1.0

print(n_correct/total)
