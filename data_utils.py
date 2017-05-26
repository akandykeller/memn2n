from __future__ import absolute_import

import os
import re
import numpy as np

def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    S_lens = []
    Q = []
    Q_lens = []
    A = []

    for story, query, answer in data:
        ss = []
        ss_lens = []
        for i, sentence in enumerate(story):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)
            # Add time word directly at the end of the sentence
            ss[i][len(sentence)] = len(word_idx) - memory_size - i + len(story)
            ss_lens.append(len(sentence) + 1)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]
        ss_lens = ss_lens[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which 
        # corresponds to vector of lookup table
        # for i in range(len(ss)):
        #    ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)
            ss_lens.append(0)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        S_lens.append(ss_lens)
        Q.append(q)
        Q_lens.append(len(query))
        A.append(y)
    return np.array(S), np.array(S_lens), np.array(Q), np.array(Q_lens), np.array(A)



def vectorize_data_ordertest(data, word_idx, sentence_size_w_order):
    """
    Vectorize stories and queries.

    S = sentence with two words appended
    A = Whether words are in correct order or flipped order
    """
    S = []
    s_lens = []
    Q = []
    A = []
    for story, query, answer in data:
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size_w_order - len(sentence))
            s = [word_idx[w] for w in sentence]
            s_lens.append(len(s))

            # word1 always comes before word2
            w1_idx = np.random.randint(0, len(s)-1)
            w2_idx = np.random.randint(w1_idx + 1, len(s))

            w1 = s[w1_idx]
            w2 = s[w2_idx]

            # 50% chance of correct order
            if np.random.random() < 0.5:
                Q.append([w1, w2])
                A.append([0, 1])
            else:
                Q.append([w2, w1])
                A.append([1, 0])

            # pad to length
            s += [0] * ls
            S.append(s)

        lq = max(0, sentence_size_w_order - len(query))
        q = [word_idx[w] for w in query]
        s_lens.append(len(q))

        # word1 always comes before word2
        w1_idx = np.random.randint(0, len(q)-1)
        w2_idx = np.random.randint(w1_idx + 1, len(q))

        w1 = q[w1_idx]
        w2 = q[w2_idx]

        # 50% chance of correct order
        if np.random.random() < 0.5:
            Q.append([w1, w2])
            A.append([0, 1])
        else:
            Q.append([w2, w1])
            A.append([1, 0])

        q += [0] * lq
        S.append(q)

    return np.array(S), np.array(s_lens), np.array(Q), np.array(A)

