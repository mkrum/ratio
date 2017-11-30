import sys
import os
from collections import Counter

def main():
    ''' do the training and testing '''
    if len(sys.argv) != 2 or sys.argv[1] not in {'english', 'hindi', 'shuffled'}:
        print('usage: python {} (english|hindi|shuffled)'.format(sys.argv[0]))
        sys.exit(1)

    data_dir = os.path.join('data', sys.argv[1])

    tasks = [1, 2]

    train_files = [os.path.join(data_dir, 'qa{}_train.txt'.format(i)) for i in tasks]
    # test_files = [os.path.join(data_dir, 'qa{}_valid.txt'.format(i)) for i in tasks]
    test_files = [os.path.join(data_dir, 'qa{}_test.txt'.format(i)) for i in tasks]

    counts = train(train_files)
    evaluate(counts, test_files)

def train(train_files):
    ''' count the number of times each answer occurs in the training data '''
    counts = Counter()

    for train_file in train_files:
        with open(train_file) as f:
            for line in f:
                if '?' in line:
                    _, _, answer, _ = parse_question(line)
                    counts[answer] += 1

    return counts


def evaluate(counts, test_files):
    ''' run the model on the test data '''
    correct = 0
    total = 0
    for test_file in test_files:
        # list of sentences in the current story
        story = []
        with open(test_file) as f:
            for line in f:
                # question
                if '?' in line:
                    num, question, answer, _ = parse_question(line)
                    question = set(question)

                    # get the statement that has the most words in common with the question.
                    # the goal is to get the most recent statement that contains the subject
                    # of the question
                    max_common = 0
                    best_statement = ''
                    for statement in story:
                        stmt_set = set(statement)
                        # greater-than-equal: take the most recent sentence if there are ties
                        if len(stmt_set & question) >= max_common:
                            max_common = len(stmt_set & question)
                            best_statement = statement

                    # pick the most common answer that is in the chosen statement.
                    # if the statement does not contain any answers, then the default
                    # is the most common answer from the training data.
                    # note than some of the hindi answers are two words
                    max_count = 0
                    best_answer = counts.most_common()[0][0]
                    best_statement = ' '.join(best_statement)
                    for ans, count in counts.items():
                        if ans in best_statement:
                            if count > max_count:
                                max_count = count
                                best_answer = ans

                    if best_answer == answer:
                        correct += 1
                    total += 1
                # statement
                else:
                    # if the id number is 1, start a new story
                    num, statement = parse_statement(line)
                    if num == 1:
                        story = []
                    story.append(statement)

    print(correct/total)


def parse_statement(line):
    ''' split a statement into id number and sentence '''
    line = line.rstrip('\n.').split('\t')[0]
    num, *statement = line.split()
    statement.append('.')
    num = int(num)
    return num, statement

def parse_question(line):
    ''' split a question into id number, question, answer, and hints '''
    question, answer, hints = line.rstrip('\n').split('\t')
    question = question.rstrip(' ?')
    num, *question = question.split()
    question.append('?')
    num = int(num)
    hints = hints.split()
    hints = [int(x) for x in hints]
    return num, question, answer, hints

if __name__ == '__main__':
    main()
