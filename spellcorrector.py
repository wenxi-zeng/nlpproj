import re
import pickle
import sys

import ngram
from queue import Queue

letters = 'abcdefghijklmnopqrstuvwxyz'


def edit_n(word, n=1):
    q = Queue()
    q.put(word)
    result = set()

    for i in range(0, n):
        while not q.empty():
            for w in edit(q.get()):
                result.add(w)
        for w in result:
            q.put(w)
        result.clear()

    while not q.empty():
        result.add(q.get())

    return result


def edit(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def candidates(model, word, context):
    result = list()
    for c in edit_n(word):
        if c in model.vocabulary:
            result.append((c, model.word_prob(c, context)))
    return result


def correct(model, word, context):
    c = candidates(model, word, context)
    if len(c) <= 0:
        return None
    else:
        return max(c, key=lambda item: item[1])[0]


def check(model, sentences):
    result = dict()
    n = model.n
    for line in sentences:
        line = line.lower()
        for word, context in ngram.get_ngrams(n, line):
            if word not in model.vocabulary:
                corrected = correct(model, word, context)
                if corrected is None:
                    continue
                if corrected not in result:
                    result[corrected] = set()
                result[corrected].add(word)

    return result


def load_test(filename):
    result = dict()
    with open(filename) as f:
        for line in f:
            s = line.split(':')
            result[s[0]] = list(re.findall(r'\w+', s[1].lower()))

    return result


def evaluate(corrected, gold):
    tp = 0
    fp = 0
    fn = 0
    for word, err_spells in corrected.items():
        gold_err_spells_set = gold.get(word, None)
        if gold_err_spells_set is None:
            fp += len(err_spells)
        else:
            for err in err_spells:
                if err in gold_err_spells_set:
                    tp += 1
                else:
                    fp += 1

    for word, gold_err_spells in gold.items():
        corrected_err_spells_set = corrected.get(word, None)
        if corrected_err_spells_set is None:
            fn += len(gold_err_spells)
        else:
            for gold_err in gold_err_spells:
                if gold_err not in corrected_err_spells_set:
                    fn += 1

    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    f_measure = 2 * precision * recall / (precision + recall)
    return precision, recall, f_measure


def save_model(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_model(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


def create_and_save_model(src_file, model_file, n):
    model = ngram.create_ngramlm(n, src_file)
    save_model(model, model_file)


def run(model, datafn, labelfn):
    test_text = ngram.load_text(datafn)
    sentences = ngram.split_sentence(test_text)
    corrected = check(model, sentences)

    print(corrected)
    gold = load_test(labelfn)
    print(evaluate(corrected, gold))


def main(args):
    if len(args) != 4:
        print("Invalid arguments, try:\n"
              "     spellcorrector.py <ngram> <corpus> <data file> <label file>")
        return

    model = ngram.create_ngramlm(int(args[0]), args[1])
    run(model, args[2], args[3])

    pass


if __name__ == '__main__':
    main(sys.argv[1:])