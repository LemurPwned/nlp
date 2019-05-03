from collections import Counter
import numpy as np
import json
import os
import re
import pickle as pkl
from colorama import Fore, Style
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import random
import networkx as nx


class Notes:
    def __init__(self):
        self.N = 51557  # all notes
        self.word_index = {}
        self.stop_words = open('./lab5/stopwords.txt', 'r').read().split('\n')
        print(f"{Fore.YELLOW}Loading saved root...{Fore.RESET}")
        self.root_dict = json.load(open('./root_dict.json', 'r'))
        # if os.path.isfile('./lab5/notes_idf.json'):
        #     print(f"{Fore.GREEN}Loading saved statistics...{Fore.RESET}")
        #     self.idf = json.load(open('./lab5/notes_idf.json', 'r'))
        #     self.tf = json.load(open('./lab5/notes_tf.json', 'r'))
        # else:
        # self.tf = Counter({})
        # self.idf = Counter({})
        # self.load_notes()

        if os.path.isfile('./lab5/notes2.pkl'):
            print(f"{Fore.GREEN}Loading saved notes...{Fore.RESET}")
            self.notes = pkl.load(open('./lab5/notes2.pkl', 'rb'))

    def load_notes(self, loc='./lab5/pap.txt'):
        self.notes = {}
        with open(loc, 'r') as f:
            lines = f.read().split('#')
            for note in lines:
                words = re.findall(r'\w+', note.lower())
                if words == []:
                    continue
                self.tf += Counter(words)
                self.idf += Counter({w: 1 for w in words})
                self.notes[words[0]] = words[1:]
        print(len(self.notes))
        self.stoplist = self.idf.most_common(60)
        json.dump(self.stoplist, open('./lab5/stoplist.json', 'w'))
        json.dump(self.idf, open('./lab5/notes_idf.json', 'w'))
        json.dump(self.tf,  open('./lab5/notes_tf.json', 'w'))
        pkl.dump(self.notes, open('./lab5/notes2.pkl', 'wb'))

    def graph_representation(self, window_size=3):
        test_size = 50
        filename = f'./lab5/graph_{window_size}_{test_size}.pkl'
        print(f"{Fore.GREEN}Calculating graphs...{Fore.RESET}")
        document_vectors = []
        # create word index
        if not os.path.isfile('test_note_dmp.pkl'):
            test_set = np.random.choice(list(self.notes.values()), (100, ))
            pkl.dump(test_set, open('test_note_dmp.pkl', 'wb'))
            print(test_set[0])
        else:
            test_set = pkl.load(open('test_note_dmp.pkl', 'rb'))
        if not os.path.isfile():
            for note in test_set:
                for w in note:
                    if w not in self.stop_words:
                        try:
                            wp = self.root_dict[w]
                        except KeyError:
                            wp = w
                            self.root_dict[wp] = wp
                        if wp not in self.word_index.keys():
                            self.word_index[wp] = len(self.word_index)
            print(len(self.word_index))
            for note in test_set:
                # G = nx.DiGraph()
                ilen = len(self.word_index)
                vsm = np.zeros(shape=(ilen**2,), dtype=int)
                nlen = len(note)
                for i, w in enumerate(note[1:]):  # first is ID
                    if w in self.stop_words:
                        continue
                    for j in range(i, i+window_size):
                        if j >= nlen:
                            break
                        if note[j] in self.stop_words:
                            continue
                        vsm[self.word_index[self.root_dict[note[j]]] *
                            ilen + self.word_index[self.root_dict[w]]] += 1
                        # weight = 1.0
                        # if edge in G:
                        #     weight += G[note[j]][w]['weight']
                        #     print(weight)
                        # G.add_edge(*edge, weight=weight)
                document_vectors.append(vsm)
            pkl.dump(document_vectors, open(filename, 'rb'))
        else:
            print(f'Loading saved graphs...')
            document_vectors = pkl.load(
                open(filename, 'wb'))
        del self.root_dict
        # clusterize
        print(len(document_vectors))
        classes = 10
        pca = TruncatedSVD(n_components=5)
        features_t = pca.fit(np.array(document_vectors))
        # del document_vectors
        print(f"{Fore.MAGENTA}Projecting with TSNE...{Fore.RESET}")
        tsne = TSNE(n_components=2, perplexity=5)
        Y = tsne.fit_transform(features_t)

        color_list = [['#'+''.join([random.choice('0123456789ABCDEF')
                                    for j in range(6)])] for i in range(classes)]
        mb = MiniBatchKMeans(n_clusters=classes, batch_size=100)
        # Y = np.array(document_vectors)
        mb.fit_transform(Y)
        labels = mb.labels_
        for i, label in enumerate(np.unique(labels)):
            plt.scatter(Y[labels == label, 0],
                        Y[labels == label, 1], c=color_list[i], label=str(i))
        plt.title("Klasyfikacja PAP")
        plt.legend()
        plt.show()

    def classify(self):
        classes = 10
        print(f"{Fore.CYAN}Tokenizing...{Fore.RESET}")
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', analyzer='word',
                                ngram_range=(1, 2), stop_words=self.stop_words)
        features = tfidf.fit_transform(
            np.random.choice(self.notes, size=(150, )))
        print(f"{Fore.YELLOW}Reducing with Truncated SVD...{Fore.RESET}")
        pca = TruncatedSVD(n_components=5)
        features_t = pca.fit_transform(features)
        print(f"{Fore.MAGENTA}Projecting with TSNE...{Fore.RESET}")
        tsne = TSNE(n_components=2, perplexity=5)
        Y = tsne.fit_transform(features_t)

        color_list = [['#'+''.join([random.choice('0123456789ABCDEF')
                                    for j in range(6)])] for i in range(classes)]
        mb = MiniBatchKMeans(n_clusters=classes)
        mb.fit_transform(Y)
        labels = mb.labels_
        feature_names = np.array(tfidf.get_feature_names())
        for i, label in enumerate(np.unique(labels)):
            plt.scatter(Y[labels == label, 0],
                        Y[labels == label, 1], c=color_list[i], label=feature_names[label])
        plt.title("Klasyfikacja PAP")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    n = Notes()
    # n.load_notes()
    # n.classify()
    n.graph_representation(3)
