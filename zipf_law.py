import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
import pandas as pd
from queue import Queue
from collections import Counter


class ZipfLaw:
    def __init__(self):
        self.ngram_size = 2
        self.forbidden_signs = [' ', '?', '.', '!', ":", ]
        self.words = re.findall(
            r'\w+', open('lab2/potop.txt').read().lower())

        if not os.path.isfile(f'vectorized2_{self.ngram_size}.txt'):
            if not os.path.isfile('root_dict.json'):
                self.root_dict = self.extract_root_dict()
                json.dump(self.root_dict, open('root_dict.json', 'w'))
            else:
                print(f"Loading precalculated dictionary...")
                self.root_dict = json.load(open('root_dict.json', 'r'))
            self.histogram = self.vectorize_text()
            # json.dump(self.histogram, open(
            #     f'vectorized_{self.ngram_size}.txt', 'w'))
        else:
            print("Loading precalculated vectorized text...")
            self.histogram = json.load(
                open(f'vectorized_{self.ngram_size}.txt', 'r'))
        self.plot_law()

    def plot_law(self):
        mst_comon = self.histogram.most_common(30)
        xs = [x[0] for x in mst_comon]
        ys = [x[1] for x in mst_comon]
        print(xs, ys)
        df = pd.DataFrame(data={'words': xs, 'occurences': ys})
        sns.catplot(x='words', y='occurences', kind='bar', data=df)
        plt.title(f"Ngrams {self.ngram_size}")
        plt.show()

    def extract_root_dict(self, loc='lab2/odm.txt'):
        root_dict = {}
        with open(loc, 'r') as f:
            for line in f:
                words = line.replace(',', '').replace('\n', '').split(' ')
                for word in words:
                    root_dict[word.lower()] = words[0].lower()
        return root_dict

    def vectorize_text(self, text_loc='lab2/potop.txt'):
        wgrams = {}
        # with open(text_loc, 'r') as f:
        #     content = f.read().replace('\n', ' ')
        #     for char in [',', ':', '!', '.', '?', ';', '"', '-', ')', '(']:
        #         content = content.replace(char, '')
        #     content = content.split(" ")
        content = self.words
        if self.ngram_size == 1:
            return Counter(self.words)
        gram_count = 0
        ngram = content[0:self.ngram_size]
        unknown_count = 0
        queue = Queue(maxsize=self.ngram_size)
        for i in range(self.ngram_size, len(content)):
            if queue.not_full():
                queue.put(self.root_dict[content[i].lower()])
            if tuple(ngram) in wgram:
                wgrams[tuple(ngram)] += 1
            else:
                wgrams[tuple(ngram)] = 1
            try:
                ngram[gram_count] = self.root_dict[content[i].lower()]
            except KeyError:
                unknown_count += 1
                print(f"Unknown:{content[i].lower()}|")
                self.root_dict[content[i].lower()] = content[i].lower()
                ngram[gram_count] = content[i].lower()

            gram_count += 1
            if gram_count == self.ngram_size:
                gram_count = 0
        print(f"Unknowns: {unknown_count}")
        return Counter(wgrams)


if __name__ == "__main__":
    zl = ZipfLaw()
