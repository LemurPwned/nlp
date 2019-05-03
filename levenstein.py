import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import davies_bouldin_score
from collections import Counter
from colorama import Fore, Back, Style
from nltk.cluster.kmeans import KMeansClusterer
import nltk
import os
from difflib import SequenceMatcher


def longestSubstring(str1, str2):
    seqMatch = SequenceMatcher(None, str1, str2)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))
    if (match.size != 0):
        return 1 - len(str1[match.a: match.a + match.size])/np.max([len(str1), len(str2)])
    else:
        return 1


def dunn_index(cluster_centers, cluster_sizes):
    max_size = np.max(cluster_sizes)
    min_distance = 99999
    for i in range(0, len(cluster_centers)):
        for j in range(0, i):
            dist = np.sqrt(
                np.sum(np.power(cluster_centers[i] - cluster_centers[j], 2)))
            if dist < min_distance:
                min_distance = dist
    return min_distance/max_size


def dice_metric(ngram1, ngram2, ngram_size=2):
    common = len(set(ngram1) & set(ngram2))
    ngramsum = len(ngram1) + len(ngram2)
    return (2*common/ngramsum)


def cosine_metric(ngram1, ngram2, ngram_size=2):
    v1 = Counter(ngram1)
    v2 = Counter(ngram2)
    common_vals = set(v1.keys()) & set(v2.keys())
    cval = np.sum([v1[key]*v2[key] for key in common_vals]) / \
        (len(v1.keys())*len(v2.keys()))
    return cval


def cluster_perf(cluster_centers_indices_, matrix, labels):
    cluster_center_labels = [labels[index]
                             for index in cluster_centers_indices_]
    cluster_sizes = [len(labels[labels == cluster_label])
                     for cluster_label in cluster_center_labels]
    min_dist = 9999
    for cluster_index_1 in cluster_centers_indices_:
        for cluster_index_2 in cluster_centers_indices_:
            if cluster_index_1 == cluster_index_2:
                continue
        dist = matrix[cluster_index_1, cluster_index_2]
        if dist == 0.0:
            continue
        if dist < min_dist:
            min_dist = dist
    print(min_dist)
    return min_dist/np.max(cluster_sizes)


class Clustering:
    def __init__(self):
        self.word_counter = Counter()
        self.cluster_number = 20
        # self.cluster_number = 7
        self.clf = KMeans(n_clusters=self.cluster_number,
                          precompute_distances=False)
        self.clf = AffinityPropagation(affinity='precomputed')

    def form_similarity_matrix(self, tokens, metrics):
        similarity_matrix = np.empty(shape=(len(tokens), len(tokens)))
        for i in range(len(tokens)):
            if i % 500 == 0:
                print(f"{Fore.RED}Processing...{i}{Fore.RESET}")
            for j in range(len(tokens)):
                if i == j:
                    similarity_matrix[i, j] = 0
                    similarity_matrix[j, i] = 0
                    continue
                similarity_matrix[i, j] = metrics(tokens[i], tokens[j])
                similarity_matrix[j, i] = similarity_matrix[i, j]
        return similarity_matrix

    def purify_names(self, company_names):
        new_company_names = []
        for name in company_names:
            new_name = []
            for word in name.split(' '):
                if word not in self.stoplist:
                    new_name.append(word)
            new_company_names.append(" ".join(new_name))
        return new_company_names

    def perform_clustering(self, force=False):
        self.samples = 3000
        self.metric = dice_metric
        company_names = self.extract_names()
        self.form_stoplist()
        company_names = company_names[:self.samples]
        print(f"{Fore.MAGENTA}Tokenizing...{Fore.RESET}")
        if self.metric.__name__ != 'longestSubstring':
            company_ngrams = [self.ngramize_words(
                name) for name in company_names]
        else:
            company_ngrams = company_names
        company_ngrams = list(filter(lambda x: x != [], company_ngrams))
        # tokenize names !
        # do string similarity not word similarity
        clustering = 'other'
        if clustering == 'kmeans':
            self.cluster_number = int(np.sqrt(self.samples))
            self.clf = KMeans(
                n_clusters=self.cluster_number, n_jobs=-1, max_iter=1200)
        else:
            self.clf = AffinityPropagation(affinity='precomputed')
        print(f"{Fore.YELLOW}Tokens: {len(company_ngrams)}")

        if os.path.isfile(f'{self.metric.__name__}_{self.samples}1.npy') and (not force):
            print(
                f"{Fore.GREEN}Loading a precomputed similarity matrix...{Fore.RESET}")
            similarity_matrix = np.load(
                f'{self.metric.__name__}_{self.samples}1.npy')
        else:
            print(f"{Fore.GREEN}Forming similarity matrix...{Fore.RESET}")
            similarity_matrix = self.form_similarity_matrix(
                company_ngrams, self.metric)
            np.save(f'{self.metric.__name__}_{self.samples}1.npy',
                    similarity_matrix)
        print(f"{Fore.CYAN}Calculating clusters...{Fore.RESET}")
        self.clf.fit(similarity_matrix)
        labels = self.clf.labels_
        if clustering == 'kmeans':
            cluster_centers = self.clf.cluster_centers_
            cluster_sizes = [len(labels[labels == labels])
                             for label in np.unique(labels)]
            perf = dunn_index(cluster_centers, cluster_sizes)
        else:
            cluster_centers = self.clf.cluster_centers_indices_
            perf = cluster_perf(cluster_centers, similarity_matrix, labels)
        bould = davies_bouldin_score(similarity_matrix, labels)

        print(len(cluster_centers))
        print(
            f"{Fore.MAGENTA}Perfomance: Dunn: {perf}, Davies-Bouldin {bould}...{Fore.RESET}")
        print(f"{Fore.BLUE}Writing clusters...{Fore.RESET}")
        with open(f'1_{self.samples}_{self.clf.__class__}_{self.metric.__name__}_clusterfile.txt', 'w') as f:
            for lab in np.unique(labels):
                for index in np.nonzero(labels == lab)[0]:
                    f.write(f'{company_names[index]}\n')
                f.write("###############\n\n")

    def extract_names(self, loc='lab3/lines.txt'):
        company_names = []
        self.line_counter = 0
        with open(loc, 'r') as f:
            for line in f:
                self.line_counter += 1
                fixed_line = line.replace('\n', '')
                company_names.append(fixed_line)
                for sgn in [',', ';', '?', ':']:
                    fixed_line = fixed_line.replace(sgn, ' ')
                split = fixed_line.split(' ')
                # remove numbers?
                for word in split:
                    self.word_counter[word.upper()] += 1
        return company_names

    def form_stoplist(self, plot=False):
        min_occ = np.sqrt(self.line_counter)/2
        mst_common = self.word_counter.most_common(50)
        stoplist = list(filter(lambda x: x[1] > min_occ, mst_common))
        # remainder = list(filter(lambda x: x[1] not in stoplist, mst_common))
        xs = [x[0] for x in mst_common]
        ys = [x[1] for x in mst_common]
        self.stoplist = [x[0] for x in stoplist]
        print(len(self.stoplist))
        print(self.stoplist)
        # self.remainder = [x[0] for x in remainder]
        if plot:
            plt.plot(xs, ys)
            plt.xticks(rotation=35)
            plt.show()

    def ngramize(self, words, ngram_size=1):
        ngrams = []
        pure_words = []
        for word in words.split(' '):
            if word not in self.stoplist:
                pure_words.append(word)
        for i in range(0, len(pure_words)):
            ngrams.append(' '.join(pure_words[i:i+ngram_size]))
        if len(ngrams) == 0:
            print(words, pure_words)
        return ngrams

    def ngramize_words(self, words, ngram_size=2):
        ngrams = []
        for i in range(len(words)):
            ngram = words[i:i+ngram_size]
            ngrams.append(ngram)
        return ngrams

    def threshold_cluster(self, matrix, company_names, thres=0.8):
        overlooked_indices = []
        clusters = []
        for i, row in enumerate(matrix):
            if i not in overlooked_indices:
                is_threshold = np.nonzero(row[row < thres])
                cluster = []
                for i in is_threshold:
                    cluster.append(company_names[i])
                    overlooked_indices.append(i)
                clusters.append(cluster)
        with open("test_clusters.txt", 'r') as f:
            for cluster in clusters:
                for comp in cluster:
                    f.write(f'{comp}')


if __name__ == "__main__":
    # print(longestSubstring())
    clust = Clustering()
    # clust.no_gram_clustering()
    clust.perform_clustering()
