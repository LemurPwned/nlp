import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import json
import pandas as pd
from colorama import Fore, Back, Style
from collections import Counter


def euclid_metric(v1, v2, common_vals):
    return 1.0/np.sqrt(np.sum([np.power(v1[key] - v2[key], 2) for key in common_vals]))


def manhattan_metric(v1, v2, common_vals):
    return 1.0/np.sum([np.abs(v1[key] - v2[key]) for key in common_vals])


def max_metric(v1, v2, common_vals):
    return 1.0/np.max([np.abs(v1[key] - v2[key]) for key in common_vals])


def cosine_metric(v1, v2, common_vals):
    cval = np.sum([v1[key]*v2[key] for key in common_vals]) / \
                  (len(v1.keys())*len(v2.keys()))
    return 1.0/(1.0-cval)


class NgramClassifier:
    def __init__(self, ngram_size=2):
        self.metrics = [euclid_metric, cosine_metric,
                        manhattan_metric, max_metric]
        self.encodings = ['utf-8', 'windows-1250', 'windows-1252']
        self.set_ngram_size(ngram_size)

    def set_ngram_size(self, ngram_size):
        self.ngram_size = ngram_size
        json_filename = f'corpus_{self.ngram_size}.json'
        self.corpus = {}
        if os.path.isfile(json_filename):
            self.corpus = json.load(open(json_filename, 'r'))
        else:
            for lang in glob.iglob('./lab1/*'):
                corpus_lang = os.path.basename(lang)[:-5]
                if corpus_lang not in self.corpus:
                    self.corpus[corpus_lang] = {}
                hash_gram = self.build_ngram_statistics(lang)
                self.corpus[corpus_lang] = {
                    **hash_gram, **self.corpus[corpus_lang]}
            json.dump(self.corpus, open(json_filename, 'w'))

    def detect_language(self, sample_text, true_label=None, file_type=True):
        sample_ngram = self.build_ngram_statistics(sample_text, file_type)
        verdict = []
        langs = list(self.corpus.keys())
        for lang in langs:
            """
            only common sets are computed to reduce complexity
            """
            common_vals = set(
                [*sample_ngram.keys(), *self.corpus[lang].keys()])
            metrics_vals = [metrics(Counter(sample_ngram), Counter(self.corpus[lang]), common_vals)
                            for metrics in self.metrics]
            verdict.append(metrics_vals)
            # verbose here
            # print('\n'.join([f"{m.__name__}: {val}" for m,
                            #  val in zip(self.metrics, metrics_vals)]))
        verdict = np.array(verdict)
        print(f"\nFinal verdict for {Fore.GREEN}{sample_text}{Fore.RESET}")
        metric_array = np.zeros(shape=(len(self.metrics), 4))
        for i in range(len(self.metrics)):
            # for a given metric which was the largest
            lang_ind = np.argmax(verdict[:, i])
            print(
                f"\t{Fore.RED}{self.metrics[i].__name__}{Fore.RESET}: {Fore.GREEN}{langs[lang_ind]}{Fore.RESET}, {np.max(verdict[:,i])}")
            if true_label is not None:
                if true_label == langs[lang_ind]:
                    # tp , fp , tn, fn
                    metric_array[i, :] += np.array([1, 0, len(langs) - 1, 0])
                else:
                    metric_array[i, :] += np.array([0, 1, len(langs) - 2, 1])
        return metric_array

    def build_ngram_statistics(self, lang, file_type=True):
        hash_ngram = {}
        for enc in self.encodings:
            try:
                if file_type:
                    with open(lang, mode='r', encoding=enc) as f:
                        content = f.read().strip()
                    print(f"{lang}: detected encoding {enc}")
                else:
                    content = lang
            except UnicodeDecodeError:
                pass
            else:
                text_len = len(content)
                content = ''.join(
                    filter(lambda x: not x.isdigit(), content))
                for i in range(0, len(content)):
                    ngram = content[i:i+self.ngram_size]
                    if ngram in hash_ngram:
                        hash_ngram[ngram] += 1/text_len
                    else:
                        hash_ngram[ngram] = 1/text_len
                return hash_ngram

    def plot_histogram(self, lang):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(list(self.corpus[lang].keys()), self.corpus[lang].values(), color='b')
        plt.show()

    def launch_tests(self):
        # metrics x 4 = 4 = tp, fp, tn ,fn
        ngram_res = pd.DataFrame(
            columns=['ngrams', 'metrics', 'recall', 'precision', 'F1', 'acc'])
        for s_ngram in range(2, 5):
            self.set_ngram_size(s_ngram)
            print(f"{Fore.YELLOW}Results for ngram size of {s_ngram}{Fore.RESET}")
            metric_array = np.zeros(shape=(len(self.metrics), 4))
            c = 0
            with open('test_sentences.txt', 'r') as f:
                for line in f:
                    c += 1
                    label, text = line.split(';')
                    metric_array += self.detect_language(text,
                                                        true_label=label, file_type=False)
            recall = metric_array[:, 0] / \
                (metric_array[:, 0] + metric_array[:, 3])
            precision = metric_array[:, 0] / \
                (metric_array[:, 0] + metric_array[:, 1])
            acc = (metric_array[:, 0] + metric_array[:, 2]) / \
                   np.sum(metric_array[:, :], axis=1)
            print(f"\n{Fore.MAGENTA}SCORES FOR EACH METRIC{Fore.RESET}")
            for i in range(len(self.metrics)):
                print(
                    f"{self.metrics[i].__name__}, recall {recall[i]}, prec. {precision[i]}, F1 {2*recall[i]*precision[i]/(precision[i]+recall[i])}")
                print(
                    f"\tTP, FP, TN, FN {metric_array[i, :]}")
            tmp_series = pd.DataFrame.from_dict(data={'recall': recall.tolist(),
                                                       'precision': precision.tolist(),
                                                       'acc': acc.tolist(),
                                                       'metrics': [m.__name__ for m in self.metrics]})
            tmp_series['F1'] = 2*tmp_series['recall']*tmp_series['precision'] / \
                (tmp_series['precision'] + tmp_series['recall'])
            tmp_series['ngrams'] = s_ngram
            ngram_res = pd.concat([ngram_res, tmp_series], axis=0, join='outer', join_axes=None, ignore_index=False,
                        keys=None, levels=None, names=None, verify_integrity=False, sort=False,
                        copy=True)
        print(ngram_res)
        ngram_res.to_csv('FINAL_RESULTS.csv')

    def visualize_ngrams(self, metric='euclid_metric'):
        df = pd.read_csv('FINAL_RESULTS.csv')
        sns.catplot(x="ngrams", y='F1', kind="bar", palette="ch:.25", data=df[df['metrics'] == metric])
        plt.title(metric)
        plt.xlabel("Ngram size")
        plt.show()

if __name__ == "__main__":
    nc=NgramClassifier()
    # nc.launch_tests()
    nc.visualize_ngrams('max_metric')
