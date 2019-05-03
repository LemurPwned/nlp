import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
import numpy as np
from scipy.optimize import curve_fit
from queue import Queue
from collections import Counter
from matplotlib.pyplot import figure
from colorama import Fore, Back, Style

figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')


def zipf_function(x, k):
    return k/x


def mandelbrot_function(x, P, d, B):
    return P/((x + d)**B)


class ZipfLaw:
    def __init__(self):
        self.ngram_size = 3
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
        # hapax_legomena x: occ(x) = 1
        hapax_legomena = [
            key for key in self.histogram if self.histogram[key] == 1]
        perc = self.calculate_percentile()
        print(f"{Fore.CYAN}Hapax legomena:{Fore.RESET} {len(hapax_legomena)}")
        if self.ngram_size == 1:
            print(f"{Fore.GREEN}Percentile 50:{Fore.RESET} {len(perc)}: {perc}")
        else:
            print(
                f"{Fore.GREEN}Percentile 50:{Fore.RESET} {len(perc)}: {perc[:5]}")
        self.plot_law()

    def calculate_percentile(self, percentile=0.5):
        counter = 0
        i = 0
        text_len = len(self.words)
        perc_words = []
        m = self.histogram.most_common()
        while counter <= (text_len*percentile):
            i += 1
            counter += m[i][1]
            perc_words.append(m[i][0])
        return perc_words

    def plot_law(self):
        mst_common = self.histogram.most_common(20)
        xs = [x[0] for x in mst_common]
        ys = [x[1] for x in mst_common]
        rank = [j+1 for j in range(len(xs))]
        ys, xs = zip(*sorted(zip(ys, xs), reverse=True))
        zpopt, _ = curve_fit(zipf_function, rank, ys)
        mpopt, _ = curve_fit(mandelbrot_function, rank,
                             ys, p0=[18286, 0.0, 1.0])
        plt.plot(rank, ys, '*r-', label='Potop')
        plt.plot(rank, zipf_function(rank, *zpopt), 'g-', label='Zipf law')
        plt.plot(rank, mandelbrot_function(rank, *mpopt),
                 'b-', label='Mandelbrot law')
        plt.xticks(rank, tuple(xs))
        plt.title(f"ngrams of size: {self.ngram_size}")
        plt.legend()
        plt.xticks(rotation=15)
        plt.show()

    def extract_root_dict(self, loc='lab2/odm.txt'):
        root_dict = {}
        with open(loc, 'r') as f:
            for line in f:
                words = line.replace('\n', ' ').split(',')
                for word in words:
                    root_dict[word.lower().strip()] = words[0].lower()
        return root_dict

    def vectorize_text(self, text_loc='lab2/potop.txt'):
        wgrams = Counter()
        content = self.words
        unknown_count = 0
        if self.ngram_size == 1:
            for word in self.words:
                try:
                    wgrams[self.root_dict[word]] += 1
                except KeyError:
                    print(f"Unknown:{word}|")
                    unknown_count += 1
                    self.root_dict[word] = word
                    wgrams[word] += 1
            print(f"{Fore.RED}Unknowns:{Fore.RESET} {unknown_count}")
            return wgrams
        print("Counting grams...")
        queue = Queue(maxsize=self.ngram_size)
        for i in range(self.ngram_size, len(content)):
            if queue.full():
                current_ngram = ' '.join(queue.queue)
                wgrams[current_ngram] += 1
                queue.get()
            try:
                queue.put(self.root_dict[content[i]])
            except KeyError:
                print(f"Unknown:{content[i]}|")
                unknown_count += 1
                self.root_dict[content[i]] = content[i]
                queue.put(content[i])
        print(f"{Fore.RED}Unknowns:{Fore.RESET} {unknown_count}")
        return wgrams


if __name__ == "__main__":
    zl = ZipfLaw()
