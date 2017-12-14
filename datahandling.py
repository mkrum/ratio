class TaskData(object):

    def __init__(self, task_str, language):

        self.train_data = []
        self.test_data = []
        self.valid_data = []

        for task in task_str.split('-'):
            self.load_data(task, language)


    def load_data(self, task, language):
        ''' adds the data to train, validation, test '''
        train_filename = 'data/{}/qa{}_train.txt'.format(language, task)
        valid_filename = 'data/{}/qa{}_valid.txt'.format(language, task)
        test_filename = 'data/{}/qa{}_test.txt'.format(language, task)

        self.train_data += self.parse_data(train_filename)
        self.valid_data += self.parse_data(valid_filename)
        self.test_data += self.parse_data(test_filename)


    @staticmethod
    def parse_data(filename):
        with open(filename) as f:
            data = []
            story = []
            answers = []

            for line in f:
                if '?' in line:
                    _, question, answer, _ = TaskData.parse_question(line)
                    answers.append(answer)
                    story.append(question)
                else:
                    number, line = TaskData.parse_statement(line)
                    if number == 1 and answers:
                        data.append((story, answers))
                        story = []
                        answers = []
                    story.append(line)

            return data


    @staticmethod
    def parse_statement(line):
        ''' split a statement into id number and sentence '''
        line = line.rstrip('\n.').split('\t')[0]
        num, *statement = line.split()
        statement.append('.')
        num = int(num)
        return num, statement


    @staticmethod
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
    my_data = TaskData(1, 'english')
    my_story, my_answer = my_data.train_data[0]
    print(my_story)
    print()
    print(my_answer)
