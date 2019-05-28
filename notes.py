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

    def graph(self, test_set, window_size=2):
        filename = f'./lab5/graph_{window_size}_{len(test_set)}.pkl'
        print(f"{Fore.GREEN}Calculating graphs...{Fore.RESET}")
        document_vectors = []
        # create word index
        if not os.path.isfile(filename):
            print(f'Words in index: {len(self.word_index)}')
            for note in test_set:
                ilen = len(self.word_index)
                vsm = np.zeros(shape=(ilen**2,), dtype=np.uint8)
                nlen = len(note)
                for i, w in enumerate(note[1:]):  # first is ID
                    if w in self.stop_words:
                        continue
                    for j in range(i, i+window_size):
                        if j >= nlen:
                            break
                        if note[j] in self.stop_words:
                            continue
                        vsm[self.word_index[note[j]] *
                            ilen + self.word_index[w]] += 1
                document_vectors.append(vsm)
            pkl.dump(document_vectors, open(filename, 'wb'))
        else:
            print(f'Loading saved graphs...')
            document_vectors = pkl.load(
                open(filename, 'rb'))
        return np.array(document_vectors)

    def classify_and_evaluate(self, n_classes=5, test_size=50):
        dmp_filename = f'./lab5/test_{test_size}_dmp_note.pkl'
        if not os.path.isfile(dmp_filename):
            print(f'Reducing to stems...')
            test_set = [self.notes[str(i+1).zfill(6)]
                        for i in range(test_size)]
            self.root_dict = json.load(open('./root_dict.json', 'r'))
            new_test_set = []
            for note in test_set:
                new_note = []
                for w in note:
                    if w not in self.stop_words:
                        try:
                            wp = self.root_dict[w]
                        except KeyError:
                            wp = w
                            self.root_dict[wp] = wp
                        if wp not in self.word_index.keys():
                            self.word_index[wp] = len(self.word_index)
                        new_note.append(wp)
                new_test_set.append(new_note)
            test_set = new_test_set
            del self.root_dict
            pkl.dump(test_set, open(dmp_filename, 'wb'))
            pkl.dump(self.word_index, open('./lab5/word_index.pkl', 'wb'))
        else:
            test_set = pkl.load(open(dmp_filename, 'rb'))
            self.word_index = pkl.load(open('./lab5/word_index.pkl', 'rb'))
        print(
            f'{Fore.BLUE}Read test set of size {len(test_set)}, {len(test_set[0])}, {len(test_set[1])}{Fore.RESET}')
        for i, method in enumerate([self.tfidf, self.graph, self.graph, self.graph, self.graph]):
            if i == 0:
                documents = method(test_set)
            else:
                documents = method(test_set, i+1)
            pca = TruncatedSVD(n_components=10)
            print(f"\t{Fore.MAGENTA}Reducing with SVD...{Fore.RESET}")
            print(documents.shape)
            features_reduced = pca.fit_transform(documents)
            train_notes = features_reduced[:-1, :]
            test_note = features_reduced[-1,
                                         :].reshape(1, -1)  # last is test one
            mb = MiniBatchKMeans(n_clusters=n_classes, batch_size=100)
            mb.fit(train_notes)
            test_label = mb.predict(test_note)
            train_labels = mb.labels_
            similar_notes_indices = np.nonzero(
                train_labels == test_label)[0].tolist()
            similar_notes_indices = [str(i+1).zfill(6)
                                     for i in similar_notes_indices]
            print(f'\t{Fore.CYAN}Method {method.__name__}, arg {i}{Fore.RESET}')
            print(
                f'\t{Fore.MAGENTA}Found {len(similar_notes_indices)} similar notes in train set...{Fore.RESET}')

    def tfidf(self, documents):
        documents = [" ".join(doc) for doc in documents]
        print(f"{Fore.CYAN}Tokenizing... {len(documents)}{Fore.RESET}")
        print(documents[0])
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', analyzer='word', lowercase=False,
                                ngram_range=(1, 2), stop_words=self.stop_words)
        features = tfidf.fit_transform(documents)
        return features


if __name__ == "__main__":
    n = Notes()
    n.classify_and_evaluate(5, 100)
