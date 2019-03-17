import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
import numpy as np
from scipy.optimize import curve_fit
from queue import Queue
from collections import Counter


def zipf_function(x, k):
    return k/x


def mandelbrot_function(x, P, d, B):
    return P/((x + d)**B)


class ZipfLaw:
    def __init__(self):
        self.ngram_size = 1
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
        mst_comon = self.histogram.most_common(50)
        xs = [x[0] for x in mst_comon]
        ys = [x[1] for x in mst_comon]
        rank = [j+1 for j in range(len(xs))]
        ys, xs = zip(*sorted(zip(ys, xs), reverse=True))
        zpopt, _ = curve_fit(zipf_function, rank, ys)
        mpopt, _ = curve_fit(mandelbrot_function, rank,
                             ys, bounds=(0.001, None))
        plt.plot(rank, ys, '*r-', label='Potop')
        plt.plot(rank, zipf_function(rank, *zpopt), 'g-', label='Zipf law')
        plt.plot(rank, mandelbrot_function(rank, *mpopt),
                 'b-', label='Mandelbrot law')
        plt.xticks(rank, tuple(xs))
        plt.title(f"ngrams of size: {self.ngram_size}")
        plt.legend()
        plt.show()

    def extract_root_dict(self, loc='lab2/odm.txt'):
        root_dict = {}
        with open(loc, 'r') as f:
            for line in f:
                words = line.replace('\n', '').split(',')
                for word in words:
                    root_dict[word.lower().strip()] = words[0].lower()
        return root_dict

    def vectorize_text(self, text_loc='lab2/potop.txt'):
        wgrams = Counter()
        content = self.words
        if self.ngram_size == 1:
            return Counter(self.words)
        print("Counting grams...")
        unknown_count = 0
        queue = Queue(maxsize=self.ngram_size)
        for i in range(self.ngram_size, len(content)):
            if queue.full():
                current_ngram = ' '.join(queue.queue)
                wgrams[current_ngram] += 1
                queue.get()
            try:
                if self.root_dict[content[i].lower()] == 'w':
                    print(content[i])
                queue.put(self.root_dict[content[i].lower()])
            except KeyError:
                print(f"Unknown:{content[i].lower()}|")
                self.root_dict[content[i].lower()] = content[i].lower()
                queue.put(content[i].lower())
        print(f"Unknowns: {unknown_count}")
        return wgrams


if __name__ == "__main__":
    zl = ZipfLaw()
