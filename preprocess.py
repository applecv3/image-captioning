from nltk.tokenize import word_tokenize
import pickle, re
import config


class Processor:

    def __init__(self, data, path, is_training):

        self.config = config.Config()
        self.data = data
        self.path_pickle = path
        self.is_training = is_training

        self.word2idx = None
        self.img_idx2file = {}

        self.selected_caption = []
        self.selected_image_id = []

        self.process()

    def remove(self, sentence):

        striped_sentence = sentence.strip().lower()
        processed_sentence = re.sub("[(,),;]", '', striped_sentence)

        while not processed_sentence[-1].isalpha() and not processed_sentence[-1].isdigit():
            processed_sentence = processed_sentence[:-1]

        return processed_sentence

    def converter(self, words):

        converted_words = [self.word2idx.get(word, 2) for word in words]  # convert each word into idx

        return converted_words

    def padding(self, indice):  # pad with zero from behind if it's shorter than the "max_length"

        length = len(indice)

        if length < self.config.max_length:

            length_to_pad = self.config.max_length - length - 1
            indice += [3]+[0] * length_to_pad  # add an 'end token' at the end of each caption and pad

        return indice

    def select_by_length(self, annotations):  # only to get captions with length <= max_length

        for ann in annotations:

            completed_caption = self.remove(ann['caption'])
            words = word_tokenize(completed_caption)

            if len(words) < self.config.max_length:

                self.selected_caption.append(words)
                self.selected_image_id.append(ann['image_id'])

    def process(self):

        annotations = self.data['annotations']
        self.select_by_length(annotations)

        if not self.is_training:
            print('Loading the word2idx dict created on the train data')
            with open(self.path_pickle, 'rb') as f:
                self.word2idx = pickle.load(f)

        else:
            print('Creating a new word2idx dict')
            self.word2idx = {'<pad>': 0, '<start>': 1, '<unknown>': 2, '<end>': 3}
            word_frequency = {}

            for words in self.selected_caption:

                for word in words:

                    word_frequency[word] = word_frequency.get(word, 0) + 1

            sorted_words = sorted(word_frequency.items(), key=lambda x: -x[1])[
                           :self.config.vocab_size - len(self.word2idx)]  # only to get the top 5000 most frequent ones

            for idx, items in enumerate(sorted_words):
                self.word2idx[items[0]] = idx + 4  # starts from 4

            assert len(self.word2idx) == self.config.vocab_size

            with open(self.path_pickle, 'wb') as f:
                pickle.dump(self.word2idx, f)

        img_info = self.data['images']

        self.img_idx2file = {data['id']: data['file_name'] for data in img_info}







