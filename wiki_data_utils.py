from __future__ import absolute_import
from __future__ import print_function

import os
import re
import numpy as np
import csv
import itertools
from collections import Counter
import unicodedata
import io

def load_wiki(data_dir='movieqa/'):
    """Load wiki text (titles & text) and questions.
    
    Returns a tuple of the wikipedia keys & values along with a list 
    of questions and answers for training, dev, and test. Also has list of all
    entities used for softmax.
    """

    qa_dir = data_dir + 'questions/wiki_entities/'
    ks_dir = data_dir + 'knowledge_source/'

    ks_files = ['wiki-entities_qa_train.txt', 'wiki-entities_qa_dev.txt', 'wiki-entities_qa_test.txt'] 

    print("Parsing Entity File")
    re_list, entities, ent_rev, ent_idx = get_ent_regexs(ks_dir + 'entities.txt')

    questions = []
    for f in ks_files:
        print("Parsing Question File: {}".format(f))
        questions.append(parse_questions(qa_dir + f, re_list))
    
    print("Parsing Wiki File")
    docs = parse_wiki(ks_dir + 'wiki.txt', re_list)

    return docs, questions, (re_list, entities, ent_rev, ent_idx)


def get_ent_regexs(ent_file):
    """
    Returns a list of regex's to find entities, a dict mapping enities to unique
    tokens, and a reverse dictionary
    """
    #Compute Regex list to find entities in sentences
    ent_list = []
    re_list = []
    entities = {}
    ent_rev = {}
    ent_idx = {}

    with io.open(ent_file, 'r',  encoding="utf-8") as read:
        for l in read:
            l = l.strip()
            l = unicodedata.normalize('NFKD', l).encode('ascii', 'ignore')
            if len(l) > 0:
                ent_list.append(l)

    ent_list.sort(key=lambda x: -len(x))
    for i in range(len(ent_list)):
        k = ent_list[i]
        v = '__{}__'.format(i)
        entities[k] = v
        ent_rev[v] = k
        ent_idx[v] = i + 1  # 0 is reserved for nil word
    
    re_list = [
        (
            re.compile('\\b{}\\b'.format(re.escape(e))),
            '{}'.format(entities[e])
        ) for e in ent_list
    ]

    return re_list, entities, ent_rev, ent_idx


def parse_questions(qa_file, re_list):
    splitter = re.compile('\\b.*?\S.*?(?:\\b|$)')

    qas =[]
    q_idx = 0

    with io.open(qa_file, 'r',  encoding="utf-8") as f:
        for ex in f:
            if q_idx % 1000 == 0:
                print("{} questions parsed.".format(q_idx))
            
            ex = unicodedata.normalize('NFKD', ex).encode('ascii', 'ignore')

            # Split questions and answers
            tabs = ex.split('\t')
            q = tabs[0][1:]   # Remove first element since it is a number
            ans = tabs[1]
            # Substitie all entities with single tokens
            for r, v in re_list:
                q = r.sub(v, q)
                ans = r.sub(v, ans)
            # Tokenize
            q_split = [t.strip() for t in splitter.findall(q)]
            ans = ans.replace(',', ' ')
            a_split = [t.strip() for t in splitter.findall(ans)]
            
            a_split = [a for a in a_split if '__' in a]
            
            qas.append([q_split, a_split])
            q_idx += 1

    return qas


def parse_wiki(wiki_file, re_list):
    splitter = re.compile('\\b.*?\S.*?(?:\\b|$)')

    docs = []
    last_line_no = 0

    with io.open(wiki_file, 'r',  encoding="utf-8") as f:
        lines = f.readlines()
        
        # Ignore title for sentence kv-encoding
        # title = ''
        doc_lines = []
        
        # Loop over lines in the documnet
        for i, line in enumerate(lines):
            line = line.strip()
            if line == '':
                continue
            idx = int(line[:line.find(' ')])
            line = line[line.find(' ') + 1:] 
            
            line = unicodedata.normalize('NFKD', line).encode('ascii', 'ignore')
            
            if idx < last_line_no:
                docs.append(doc_lines)
                # title = ''
                doc_lines = []
            
            if idx == '1':
                # Again ignore title
                # title = ' '.join(words[1:])
                # title.replace('\n', '')
                doc_lines = []

            elif int(idx) >= 1:
                for r, v in re_list:
                    line = r.sub(v, line)

                split = [t.strip() for t in splitter.findall(line)]
                doc_lines.append(split)
            
            else:
                print("IDK ", line)

            if i % 10000 == 0:
                print("{} Wiki lines parsed.".format(i))
            
            last_line_no = idx
        
    return docs   


def get_word_idx(docs, questions):
    qa_train, qa_dev, qa_test = questions
    qa = qa_train + qa_dev + qa_test

    # Combine questions an answers
    qa = map(lambda x: x[0] + x[1], qa)
    
    # Flatten questions
    qa_vocab = list(itertools.chain.from_iterable(qa))
    # Flatten each document to single list, and flatten all docs together
    wiki_vocab = list(itertools.chain.from_iterable(
                            map(lambda x: list(itertools.chain.from_iterable(x)), docs)))
    vocab = set(qa_vocab + wiki_vocab)
    
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    return word_idx


def get_max_lens(docs, questions, max_key_len):
    # Use Sentence level knowledge representation, so both key are value are wikipedia sentences
    max_key_len = min(max_key_len, max(map(len, reduce(lambda x, y: x + y, docs))))
    max_value_len = max_key_len

    # Combine question sets and extract questions alone
    qa_train, qa_dev, qa_test = questions
    qa = qa_train + qa_dev + qa_test
    qs = [x[0] for x in qa]

    query_size = max(map(len, qs))

    return max_key_len, max_value_len, query_size


def get_word_freqs(docs):
    # Flatten each document to single list, and flatten all docs together
    wiki_vocab = list(itertools.chain.from_iterable(
                            map(lambda x: list(itertools.chain.from_iterable(x)), docs)))

    freqs = Counter(wiki_vocab)
    return freqs


def build_hash(docs, questions, word_idx, freq_max=1000):
    """
    For each question, find documents which share at least one word
    with Freq < 1,000.

    Return dictionary mapping index of question to indexes of all sentences which
    contain matching word.
    """
    qa_train, qa_dev, qa_test = questions

    freqs = get_word_freqs(docs)

    word_to_sentence_idxs = {}

    # For every word in vocab, if freq < 1000, get list of sentences indexes which
    # contain that word
    for word in word_idx:
        if freqs[word] <= freq_max:
            # Initialize empty list, begin adding sentence indexes
            word_to_sentence_idxs[word] = []

            # Loop over sentences (all documents combined)
            for idx, sentence in enumerate(itertools.chain.from_iterable(docs)):
                if word in sentence:
                    word_to_sentence_idxs[word].append(idx)


    # For each question, combine all indexes of all words in the question
    train_idxs = []
    for q_idx, (q, a) in enumerate(qa_train):
        train_idxs.append([])
        for word in q:
            if freqs[word] <= freq_max:
                train_idxs[q_idx] += word_to_sentence_idxs[word]
        train_idxs[q_idx] = set(train_idxs[q_idx])

    dev_idxs = []
    for q_idx, (q, a) in enumerate(qa_dev):
        dev_idxs.append([])
        for word in q:
            if freqs[word] <= freq_max:
                dev_idxs[q_idx] += word_to_sentence_idxs[word]
        dev_idxs[q_idx] = set(dev_idxs[q_idx])

    test_idxs = []
    for q_idx, (q, a) in enumerate(qa_test):
        test_idxs.append([])
        for word in q:
            if freqs[word] <= freq_max:
                test_idxs[q_idx] += word_to_sentence_idxs[word]
        test_idxs[q_idx] = set(test_idxs[q_idx])

    # Return hashes for all sets of questions
    return train_idxs, dev_idxs, test_idxs


def vectorize_data(docs, questions, ent_idx, word_idx, q_hash, memory_size, key_length, query_length):
    """
    Vectorize stories and queries.

    If a sentence length < key_length, the sentence will be padded with 0's.

    If a set of sentences from the hash is < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length key_length filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    # First flatten the docs to list of sentences
    all_sentences = list(itertools.chain.from_iterable(docs))

    S = []
    Q = []
    A = []
    for q_idx, (query, answer) in enumerate(questions):
        story = [all_sentences[s_idx] for s_idx in q_hash[q_idx]]

        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, key_length - len(sentence))
            ss.append([word_idx[w] for w in sentence[:key_length]] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which 
        # corresponds to vector of lookup table
        # for i in range(len(ss)):
        #     ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * key_length)

        lq = max(0, query_length - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        # y = np.zeros(len(ent_idx) + 1) # 0 is reserved for nil word
        y = []
        for a in answer:
            try:
                y.append(ent_idx[a])
                # y[ent_idx[a]] = 1
            except:
                print("Weird...")
                print(a)
                pass

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)


def multi_acc_score(y_preds, y_true):
    correct_preds = [yp in yt for yp, yt in zip(y_preds, y_true)]
    return float(np.mean(correct_preds))


# def tokenize(sent):
#     '''Return the tokens of a sentence including punctuation.
#     >>> tokenize('Bob dropped the apple. Where is the apple?')
#     ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
#     '''
#     return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

# def parse_entities(ent_file):
#     ents = []
#     with open(ent_file, 'r') as f:
#         ents = f.readlines()
#         ents = map(lambda x: x.replace('\n', ''), ents)
#     return ents


# def recurse_answers(ans, ents, real_ans):
#     if len(ans) == 0:
#         return real_ans

#     if ans[0] in ents:
#         real_ans.append(ans[0])
#         if len(ans) > 1 and ans[1][0] == ' ':
#             ans[1] = ans[1][1:]
#         return recurse_answers(ans[1:], ents, real_ans)
#     elif len(ans) >= 2:
#         ans_0 = [ans[0] + ',' + ans[1]]
#         ans_new = ans_0 + ans[2:]
#         return recurse_answers(ans_new, ents, real_ans)
#     else:        
#         ans = ans[0].split(',')
#         ans_0 = [real_ans[-1] + ', ' + ans[0]]
#         ans_new = ans_0 + ans[1:]
#         return recurse_answers(ans_new, ents, real_ans[:-1])

# def parse_answers(ans, ents):
#     ''' Rediculous parser needed to separate out answer entities which contain a comma

#     NOTE they also include entities with correct names. They however removed duplicate
#     answers with a simple single token set, strings dont match. 
#     '''
#     ans = ans.split(',')
#     real_answers = recurse_answers(ans, ents, [])
#     return real_answers

# def parse_questions(qa_file, ents):
#     qas =[]

#     with open(qa_file, 'r') as f:
#         reader = csv.reader(f, delimiter='\t')
#         for row in reader:
#             question_tok = tokenize(row[0])[1:]   # Remove first element since it is 
#             answers = parse_answers(row[1], ents)
#             qas.append([question_tok, answers])
#     return qas[0], qas[1]


# def parse_wiki(wiki_file):
#     docs = []
#     last_line_no = 0
#     with open(wiki_file, 'r') as f:
#         lines = f.readlines()
        
#         title = ''
#         doc_lines = []
        
#         for i, line in enumerate(lines):
#             words = line.split(' ')    
#             if line == '\n':
#                 continue
                
#             line_no = int(words[0])
            
#             if line_no < last_line_no:
#                 docs.append([title.replace('\n', ''), map(lambda x: x.replace('\n', ''), doc_lines)])
#                 title = ''
#                 doc_lines = []
            
#             if words[0] == '1':
#                 title = ' '.join(words[1:])
#                 title.replace('\n', '')
#                 doc_lines = []

#             elif int(words[0]) >= 1:
#                 line = ' '.join(words[1:])
#                 doc_lines.append(line)
            
#             else:
#                 print "IDK " + line
            
#             if i % 10000 == 0:
#                 print i
            
#             last_line_no = line_no
        
#     return docs   

