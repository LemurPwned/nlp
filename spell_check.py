from Levenshtein import editops, distance
from collections import Counter
from colorama import Fore
import os
import re
import json
import numpy as np
import random


class BayesSpellCheck:
    def __init__(self):
        self.forms = []
        self.lex = {}
        self.t_lex = Counter()
        self.total_words = 0
        self.alpha = 0.5
        self.mistakes_loc = './lab4/popular_mistakes2.json'
        # if os.path.isfile(self.mistakes_loc) and:
        #     self.popular_mistakes = json.load(open(self.mistakes_loc, 'r'))
        # else:
        print(f"{Fore.YELLOW}Calculating common mistakes...{Fore.RESET}")
        self.popular_mistakes = Counter()
        self.account_popular_mistakes()

        self.beta = 1
        self.gamma = 0.8

    def read_forms(self, loc='./lab4/formy.txt'):
        with open(loc, 'r') as f:
            for line in f:
                self.forms.append(line.strip().replace('\n', ''))

    def read_corpus(self, loc='./lab4'):
        locs = ['dramat.txt', 'proza.txt', 'popul.txt', 'publ.txt', 'wp.txt']
        for c in locs:
            c_loc = os.path.join(loc, c)
            classname = c.replace('.txt', '')
            with open(c_loc, 'r') as f:
                words = re.findall(r'\w+', f.read().lower())
                self.lex[classname] = Counter(words)
                self.t_lex += Counter(words)
            self.total_words += sum(self.lex[classname].values())

    def get_most_probable_candidates(self, mistake):
        candidates = []
        for clas in self.lex.keys():
            for word in self.lex[clas].keys():
                d = distance(word, mistake)
                if d <= 5:
                    candidates.append((word, d))
        return candidates

    def probabilities(self, word):
        """
        count number of occurences of word in a document class
        normalize by the all the words
        """
        try:
            N = 0  # all possible candidates - histogram count
            candidates = self.get_most_probable_candidates(word)
            for candidate in candidates:
                for c in self.lex.keys():
                    N += self.lex[c][candidate[0]]
            probabilities = []
            for candidate in candidates:
                Nc = 0
                for c in self.lex.keys():
                    Nc += self.lex[c][candidate[0]]
                P_c = (Nc + self.alpha) / \
                    (N + self.alpha*self.total_words)
                """
                L -> 0 -> p(c) = 1
                L -> len(c) -> p(c) = 0
                p(c) = 1-L/len(c)
                """
                lev = candidate[1]
                edits = editops(candidate[0], word)
                mistake_mod = 0
                for edit_type, pos1, pos2 in edits:
                    try:
                        mistake = (edit_type, candidate[0][pos1], word[pos2])
                        if mistake in self.popular_mistakes.keys():
                            mistake_mod += self.beta * \
                                self.popular_mistakes[mistake]
                        else:
                            mistake = (
                                edit_type, word[pos2], candidate[0][pos1])
                            if mistake in self.popular_mistakes.keys():
                                mistake_mod += self.beta * \
                                    self.popular_mistakes[mistake]
                    except IndexError:
                        continue
                lev -= mistake_mod
                P_w_c = 1-(lev/len(word))
                prob = P_w_c*(-1*np.log(P_c))
                probabilities.append((candidate, prob))
            probabilities.sort(key=lambda x: x[1], reverse=True)
            return probabilities[:np.min([5, len(probabilities)-1])]
        except KeyError:
            pass

    def mistakes(self, loc='./lab4/bledy.txt'):
        with open(loc, 'r') as f:
            lines = f.readlines()
        test_suite = np.random.choice(lines, 30)
        score = 0
        max_in_dict = 0
        for line in test_suite:
            incorrect, target = line.replace('\n', '').strip().split(';')
            print(f'{incorrect} ==>')
            lst = self.probabilities(incorrect)
            for cand in lst:
                print(f'\t{cand[0]}: {cand[1]}')
                if cand[0][0] == target:
                    score += 1
                    break
            if target in self.t_lex.keys():
                print(f"{Fore.GREEN}Target in dictionary: {target}{Fore.RESET}")
                max_in_dict += 1
            else:
                print(f"{Fore.RED}Correct not in dictionary: {target}{Fore.RESET}")
        print(f"{Fore.CYAN}Total score: {score*100/max_in_dict}%{Fore.RESET}")

    def detect_mistakes(self, s1, s2):
        edits = editops(s1, s2)
        for edit_type, pos1, pos2 in edits:
            try:
                mistake = (edit_type, s1[pos1], s2[pos2])
                if mistake not in self.popular_mistakes:
                    mistake = (edit_type, s2[pos2], s1[pos1])
                self.popular_mistakes[mistake] += 1
            except IndexError:
                continue

    def account_popular_mistakes(self, loc='./lab4/bledy.txt'):
        with open(loc, 'r') as f:
            for line in f:
                lines = line.replace('\n', '').strip().split(';')
                self.detect_mistakes(lines[0], lines[1])
        self.popular_mistakes = self.popular_mistakes.most_common(50)
        mst_com = self.popular_mistakes[0][1]
        self.popular_mistakes = {mistake:  val /
                                 mst_com for mistake, val in self.popular_mistakes}
        print(self.popular_mistakes)
        # json.dump(self.popular_mistakes, open(self.mistakes_loc, 'w'))


if __name__ == "__main__":
    bs = BayesSpellCheck()
    bs.read_corpus()
    bs.mistakes()
