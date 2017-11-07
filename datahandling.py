
import os
import glob

data_path = 'tasks_1-20_v1-2'

class TaskData(object):
    
    def __init__(self, taskname, language_class='en-10k'):
        path = '{}/{}/'.format(data_path, language_class)

        train_file = glob.glob('{}*_{}_train.txt'.format(path, taskname))[0]
        test_file = glob.glob('{}*_{}_test.txt'.format(path, taskname))[0]

        self.train_stories, self.train_answers = self.parse_data(train_file)
        self.test_stories,  self.test_answers = self.parse_data(test_file)

        self.current_train_sample = 0

    def get_train(self):
        return self.train_stories, self.train_answers

    def get_test(self):
        return self.test_stories, self.test_answers

    def parse_data(self, filename): 
        lines = open(filename, 'r').read().splitlines()

        stories = []
        answers = []
        story_buffer = []
        question_buffer = []
        answer_buffer = []
        
        last_number = 0
        for line in lines:

            start = line.find(' ')
            number = int(line[:start])
            line = line[start + 1:]

            if number == 1:
                stories.append(question_buffer)
                answers.append(answer_buffer)
                story_buffer = []
                question_buffer = []
                answer_buffer = []

            #question line
            if '\t' in line:
                line = line.split('\t')
                answer_buffer.append(line[1])
                story_buffer.append(line[0])

                question_buffer.append(story_buffer)
                story_buffer = []

            else:
                story_buffer.append(line)

        #remove first element in the list, since it is empty
        return stories[1:], answers[1:]

                


if __name__ == '__main__':
    data = TaskData('single-supporting-fact')
    stories, answers = data.get_train() 
    print(stories[0])
    print(answers[0])
