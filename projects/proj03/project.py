# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


_CRAWL_DELAY = None
_LAST_REQUEST_TIME = 0.0


def _get_crawl_delay():
    global _CRAWL_DELAY
    if _CRAWL_DELAY is None:
        try:
            resp = requests.get('https://gutenberg.org/robots.txt', timeout=10)
            resp.raise_for_status()
            match = re.search(r'(?i)crawl-delay:\\s*([0-9.]+)', resp.text)
            if match:
                _CRAWL_DELAY = float(match.group(1))
            else:
                _CRAWL_DELAY = 0.5
        except Exception:
            _CRAWL_DELAY = 0.5
    return _CRAWL_DELAY


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    global _LAST_REQUEST_TIME
    delay = _get_crawl_delay()
    wait = delay - (time.time() - _LAST_REQUEST_TIME)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    _LAST_REQUEST_TIME = time.time()
    text = resp.text.replace('\r\n', '\n').replace('\r', '\n')
    start_match = re.search(r'\*\*\*\s*START OF[^\*]*\*\*\*', text)
    end_match = re.search(r'\*\*\*\s*END OF[^\*]*\*\*\*', text)
    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)
    return text[start_idx:end_idx]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    text = book_string.replace('\r\n', '\n').replace('\r', '\n')
    paragraphs = [p for p in re.split(r'(?:\n\s*){2,}', text) if p.strip()]
    tokens = ['\x02']
    if not paragraphs:
        tokens.append('\x03')
        return tokens
    pattern = re.compile(r'\w+|[^\w\s]', re.UNICODE)
    for i, paragraph in enumerate(paragraphs):
        tokens.extend(pattern.findall(paragraph))
        tokens.append('\x03')
        if i != len(paragraphs) - 1:
            tokens.append('\x02')
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        unique_tokens = pd.Index(tokens).drop_duplicates()
        if len(unique_tokens) == 0:
            return pd.Series(dtype=float)
        probs = np.repeat(1 / len(unique_tokens), len(unique_tokens))
        return pd.Series(probs, index=unique_tokens)
    
    def probability(self, words):
        if len(words) == 0:
            return 1.0
        probabilities = self.mdl.reindex(words)
        if probabilities.isna().any():
            return 0.0
        return float(probabilities.prod())
        
    def sample(self, M):
        choices = np.random.choice(self.mdl.index.to_numpy(), size=M, p=self.mdl.values)
        return ' '.join(choices)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        if len(tokens) == 0:
            return pd.Series(dtype=float)
        counts = pd.Series(tokens).value_counts(sort=False)
        return counts / counts.sum()
    
    def probability(self, words):
        if len(words) == 0:
            return 1.0
        probabilities = self.mdl.reindex(words)
        if probabilities.isna().any():
            return 0.0
        return float(probabilities.prod())
        
    def sample(self, M):
        choices = np.random.choice(self.mdl.index.to_numpy(), size=M, p=self.mdl.values)
        return ' '.join(choices)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        tokens = list(tokens)
        if len(tokens) < self.N:
            return []
        return [tuple(tokens[i:i + self.N]) for i in range(len(tokens) - self.N + 1)]
        
    def train(self, ngrams):
        columns = ['ngram', 'n1gram', 'prob']
        if not ngrams:
            self._ngram_probs = {}
            self._prefix_map = {}
            return pd.DataFrame(columns=columns)
        ngram_counts = {}
        prefix_counts = {}
        for ng in ngrams:
            ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
            prefix = ng[:-1]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        rows = []
        prefix_map = {}
        prob_lookup = {}
        for ng, count in ngram_counts.items():
            prefix = ng[:-1]
            prob = count / prefix_counts[prefix]
            rows.append((ng, prefix, prob))
            prob_lookup[ng] = prob
            prefix_map.setdefault(prefix, []).append((ng[-1], prob))
        formatted = {}
        for prefix, opts in prefix_map.items():
            tokens = np.array([tok for tok, _ in opts], dtype=object)
            probs = np.array([val for _, val in opts], dtype=float)
            formatted[prefix] = (tokens, probs)
        self._ngram_probs = prob_lookup
        self._prefix_map = formatted
        return pd.DataFrame(rows, columns=columns)
    
    def probability(self, words):
        words = tuple(words)
        if len(words) == 0:
            return 1.0
        if len(words) < self.N:
            return self.prev_mdl.probability(words)
        total = self.prev_mdl.probability(words[:self.N - 1])
        if total == 0:
            return 0.0
        for i in range(self.N - 1, len(words)):
            ngram = words[i - self.N + 1:i + 1]
            prob = self._ngram_probs.get(ngram)
            if prob is None:
                return 0.0
            total *= prob
        return float(total)
    

    def sample(self, M):
        tokens = ['\x02']
        if M <= 0:
            tokens.append('\x03')
            return ' '.join(tokens)
        steps = max(M - 1, 0)
        for _ in range(steps):
            nxt = self._sample_from_history(tokens)
            if nxt is None:
                nxt = '\x03'
            tokens.append(nxt)
        tokens.append('\x03')
        return ' '.join(tokens)

    def _sample_from_history(self, history):
        needed = self.N - 1
        if needed <= len(history):
            prefix = tuple(history[-needed:]) if needed > 0 else tuple()
            options = self._prefix_map.get(prefix)
            if options is None:
                return None
            choices, probs = options
            return np.random.choice(choices, p=probs)
        if isinstance(self.prev_mdl, NGramLM):
            return self.prev_mdl._sample_from_history(history)
        if len(self.prev_mdl.mdl) == 0:
            return '\x03'
        choices = self.prev_mdl.mdl.index.to_numpy()
        return np.random.choice(choices, p=self.prev_mdl.mdl.values)
