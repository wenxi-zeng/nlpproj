import os.path
import sys
import nltk
import numpy as np
import re
from nltk.tokenize import sent_tokenize
from math import log, exp
import random

random.seed(1)

def load_text(filename):
	f = open(filename, 'r')
	text = f.read()
	ret = str(text).replace("\\r\\n", " ").replace("\\", "").replace("\n\n", ".")
	return ret

def split_sentence(text):
	res = re.split(r'[.?!;:,]+', text)
	res = [i for i in res if len(i) > 0]
	ret = []
	for r in res:
		ret.append(r.strip())
	return ret

def get_ngrams(n, text):
	sentences = split_sentence(text)
	tot = len(sentences)
	i = 0
	while i < tot:
		sen = re.split(r'[^\w]', sentences[i])
		prefix = ['<s>' for i in range(n - 1)]
		postfix = ['</s>']
		sen = [i for i in sen if len(i) > 0]
		sen = prefix + sen + postfix

		slen = len(sen)
		if slen < n + 1:
			i += 1
			continue

		context = sen[: n - 1]
		for j in range(n - 1, slen):
			context_str = ""
			for c in context:
				context_str += c + " "
			yield (sen[j].strip(), context_str.strip())
			context.append(sen[j])
			context.pop(0)
		i += 1


class NGramLM:
	def __init__(self, n):
		self.n = n    				 # save n
		self.ngrams_counts = dict()  # n-grams seen in the training data
		self.context_counts = dict() # contexts seen in the training data
		self.vocabulary = []	     # keeping track of words seen in the training data

	def update(self, text):
		# update internal counts and vocabulary for n-grams in text, which is a list of words / strings
		n = self.n
		for ng in get_ngrams(1, text):
			if ng[0] not in self.vocabulary:
				self.vocabulary.append(ng[0])

		for ng in get_ngrams(n, text):
			word, context = ng[0], ng[1]
			if context not in self.context_counts:
				self.context_counts[context] = 0
			self.context_counts[context] += 1
			gram = context + " " + word
			if gram not in self.ngrams_counts:
				self.ngrams_counts[gram] = 0
			self.ngrams_counts[gram] += 1

	def word_prob(self, word, context, delta = 0):
		# returns the probability of n-gram (word, context) using the model's internal counters
		# if context is previous unseen, 1 / |V|
		if context not in self.context_counts:
			return 1 / len(self.vocabulary)
		if word not in self.vocabulary:
			word = "unk"
		gram = context + " " + word
		if gram not in self.ngrams_counts:
			# return 0
			self.ngrams_counts[gram] = 0
		# return float(self.ngrams_counts[gram] + delta) / (self.context_counts[context] + delta * len(self.context_counts))
		return float(self.ngrams_counts[gram] + delta) / (self.context_counts[context] + delta * len(self.vocabulary))

	def random_word(self, context, delta = 0):
		self.vocabulary.sort()
		tot = 0
		words, counts = [], []
		for w in self.vocabulary:
			if w == 'unk':
				continue
			gram = context + ' ' + w
			if gram not in self.ngrams_counts:
				continue
			ng_count = self.ngrams_counts[gram]
			if ng_count > 0:
				words.append(w)
				counts.append(ng_count)
				tot += ng_count

		counts = [c / tot for c in counts]

		num = len(words)
		for i in range(1, num):
			counts[i] += counts[i - 1]

		t = random.random()
		l, r = 0, num
		while l < r:
			m = int((l + r) / 2)
			if counts[m] < t:
				l = m + 1
			elif counts[m] > t:
				r = m
		ind = l
		return words[ind]

	def likeliest_word(self, context, delta = 0):
		self.vocabulary.sort()
		tot = 0
		words, counts = [], []
		for w in self.vocabulary:
			if w == 'unk':
				continue
			gram = context + ' ' + w
			if gram not in self.ngrams_counts:
				continue
			ng_count = self.ngrams_counts[gram]
			if ng_count > 0:
				words.append(w)
				counts.append(ng_count)
				tot += ng_count

		ind, max_count = -1, 0
		num = len(words)
		for i in range(num):
			if max_count < counts[i]:
				max_count = counts[i]
				ind = i

		return words[ind]

	def candidate_words(self, context, k, delta = 0):
		self.vocabulary.sort()
		tot = 0
		words, counts = [], []
		for w in self.vocabulary:
			if w == 'unk':
				continue
			gram = context + ' ' + w
			if gram not in self.ngrams_counts:
				continue
			ng_count = self.ngrams_counts[gram]
			if ng_count > 0:
				words.append(w)
				counts.append(ng_count)
				tot += ng_count
		k = min(k, len(words))
		indices = sorted(range(len(counts)), key = lambda i : counts[i], reverse = True)[: k]
		ret = []
		for ind in indices:
			ret.append(words[ind])
		return ret

def create_ngramlm(n, corpus_path):
	# returns an NGramLM trained on corpus_path
	# corpus_path: the path of corpus file
	text = load_text(corpus_path)
	text = mask_rare(text)
	ret = NGramLM(n)
	ret.update(text)
	return ret

def text_prob(model, text, delta = 0):
	#  returns the log probability of text, which is again a list of words/strings, under model, which is a trained NGramLM
	n = model.n
	ret = 0
	for t in get_ngrams(n, text):
		word, context = t[0], t[1]
		# print(word, ", ", context, model.word_prob(word, context, delta))
		ret += log(model.word_prob(word, context, delta))
	return ret #exp(ret)

def mask_rare(corpus):
	vocabulary, tmp_set, rare_words = set(), set(), set()
	for ng in get_ngrams(1, corpus):
		if ng[0] in tmp_set:
			vocabulary.add(ng[0])
		tmp_set.add(ng[0])
	for w in tmp_set:
		if w not in vocabulary:
			rare_words.add(w)
	sentences = split_sentence(corpus)
	ret = ''
	for sen in corpus.split():
		for s in re.split(r'[^\w]', sen):
			if s in rare_words:
				sen = sen.replace(s, '<unk>')
		ret += ' ' + sen
	return ret


class NGramInterpolator:
	def __init__(self, n, lambdas):
		# n is the size of the largest n-gram considered by the model and
		# lambda is a list of length n containing the interpolation factors (ﬂoats) in descending order of n-gram size.
		self.models = []
		self.length = len(lambdas)
		self.lambdas = lambdas
		self.n = n
		for i in range(self.length):
			self.models.append(NGramLM(n - i))

	def update(self, text):
		# update all of the internal NGramLMs
		for m in self.models:
			m.update(text)

	def word_prob(self, word, context, delta = 0):
		# return the linearly interpolated probability using lambdas and
		# the probabilities given by the internal NGramLMs
		# need to be fixed because the first part define the prob is a log value.
		ret = 0
		for i in range(self.length):
			ret += self.models[i].word_prob(word, context, delta) * self.lambdas[i]
		return ret

def create_ngramip(n, corpus_path, lambdas):
	# returns an NGramLM trained on corpus_path
	# corpus_path: the path of corpus file
	text = load_text(corpus_path)
	text = mask_rare(text)
	ret = NGramInterpolator(n, lambdas)
	ret.update(text)
	return ret

def perplexity(model, corpus_path, delta = 0):
	# returns the perplexity of a trained model on the test data in the ﬁle corpus path
	test_text = load_text(corpus_path)
	sentences = split_sentence(test_text)
	res, N = 0, 0
	for s in sentences:
		res += text_prob(model, s, delta)
		tmp = [t for t in re.split(r'[^\w]', s) if len(t) > 0]
		N += len(tmp)
	I = res / N
	return 2 ** (-I)

def random_text(model, max_length, delta = 0):
	n = model.n
	words = []
	for i in range(n - 1):
		words.append('<s>')
	for i in range(n, n + max_length):
		context = ''
		for j in range(i - n, i - 1):
			context += words[j] + ' '
		context = context.strip()
		rand_word = model.random_word(context)
		words.append(rand_word)
		if rand_word == '</s>':
			break
	ret = ''
	for i in range(n - 1, len(words)):
		if words[i] == '</s>':
			 break
		ret += words[i] + ' '
	return ret.strip()

def likeliest_text(model, max_length, delta = 0):
	n = model.n
	words = []
	for i in range(n - 1):
		words.append('<s>')
	for i in range(n, n + max_length):
		context = ''
		for j in range(i - n, i - 1):
			context += words[j] + ' '
		context = context.strip()
		rand_word = model.likeliest_word(context)
		words.append(rand_word)
		if rand_word == '</s>':
			break
	ret = ''
	for i in range(n - 1, len(words)):
		if words[i] == '</s>':
			 break
		ret += words[i] + ' '
	return ret.strip()

def last_k_words(k, text):
	# return a string consists of the last k words of a given sentence.
	words = text.split()
	ret = ''
	for w in words[-k:]:
		ret += w + ' '
	return ret.strip()

def beam_search_text(model, max_length, k, delta = 0):
	n = model.n
	words = ''
	for i in range(n - 1):
		words += ' <s>'
	words = words.strip()

	cand_words = model.candidate_words(words, k)

	cands = []
	for w in cand_words:
		cands.append(words + ' ' + w)

	for i in range(max_length - 1):
		tmp = []
		for cand in cands:
			if last_k_words(1, cand) == '</s>':
				tmp.append(cand)
				continue
			cand_words = model.candidate_words(last_k_words(n - 1, cand), k)
			# get the K candidate words for the given context
			for w in cand_words:
				tmp.append(cand + ' ' + w)

		indices = sorted(range(len(tmp)), key = lambda i : text_prob(model, tmp[i], delta), reverse = True)[: k]
		cands = [tmp[i] for i in indices]

	ret = []
	for cand in cands:
		cand = cand.replace('<s>', '').replace('</s>', '')
		ret.append(cand.strip())
	return ret
