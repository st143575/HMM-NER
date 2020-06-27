import re
import torch

import time


start = time.time()
### data preprocessing
# open file and store each line in a list
with open("../NER-de-train.tsv") as trainset:
    ts_list = []
    for line in trainset.readlines():
        ts_list.append(line.split())
    #print(ts_list)

# list of word lists for each sentence
word_lists = []
# list of label lists for each sentence
label_lists = []
# list of words for each sentence
w_intern_list = []
# list of labels for each sentence
l_intern_list = []
# remove the first line and the last line of each sentence
# Example:  first line: ['#', "http://de.wikipedia.org/wiki/Fiddlin'_Arthur_Smith", '[2010-01-15]']
#           last line:  []
# and store words of each sentence into an internal word list and
# their corresponding labels into an internal label list
for lst in ts_list:
    if len(lst) == 0:    # a sentence is finished
        word_lists.append(w_intern_list)
        label_lists.append(l_intern_list)
    elif lst[0] == '#':  # start of a sentence
        w_intern_list = []
        l_intern_list = []
    elif len(lst) > 0:   # intern list for a real word, e.g. ['7', 'Songs', 'B-OTH', 'O']
        w_intern_list.append(lst[1])
        l_intern_list.append(lst[2])
"""
for i in range(len(word_lists)):
    print(i, word_lists[i])
    print(i, label_lists[i])
"""

# calculate number of word types and label types
M = 0  # number of word types
N = 0  # number of label types
# append all words in w_list and all labels in l_list(with duplicates)
w_list = []
l_list = []
for lst in ts_list:
    if len(lst)>0 and re.match('\d',lst[0]):
        w_list.append(lst[1])
        l_list.append(lst[2])
#w_list = [lst[1] for lst in ts_list if len(lst)>0 and re.match('\d',lst[0])]
#l_list = [lst[2] for lst in ts_list if ]

# remove duplicated words and labels from w_list and l_list
# make sure the order doesn't change
w_list_nodup = []
l_list_nodup = []
[w_list_nodup.append(w) for w in w_list if w not in w_list_nodup]
[l_list_nodup.append(l) for l in l_list if l not in l_list_nodup]
M = len(w_list_nodup)
N = len(l_list_nodup)
print("M=", M, "\tN=", N)
#print("w_list:", w_list_nodup, len(w_list_nodup))
#print("l_list:", l_list_nodup, len(l_list_nodup))
#end = time.time()
#print(str(end-start))

"""
# initialize the initial probability matrix(vector)
P_init = torch.zeros(N)
# initialize the transition probability matrix
P_trans = torch.zeros(N, N)
# initialize the observation probability matrix
P_observ = torch.zeros(N, M)
"""

### prepare for training(data structures initialization)
# initialize dictionary that mapps words to indices(position in matrix). Key: word, Value: index
word2id = dict()
for i in range(len(w_list_nodup)):
    word2id[w_list_nodup[i]] = i
# initialize dictionary that mapps labels to indices(position in matrix). Key: label, Value: index
label2id = dict()
for j in range(len(l_list_nodup)):
    label2id[l_list_nodup[j]] = j

#print("word2id:", word2id, len(word2id))
#print("label2id:", label2id, len(label2id))
#end = time.time()
#print(str(end-start))

### training
def train(word_lists, label_lists, word2id, label2id):
    """:arg
        :param word_lists: list of word lists for each sentence
        :param label_lists: list of label lists for each word in each sentence
        :param word2id: dictionary, mapps word to index, i.e. the position in matrix
        :param label2id: dictionary, mapps label to index, i.e. the position in matrix
        :return: P_init, P_trans, P_observ
        """
    assert len(word_lists) == len(label_lists)

    # initialize the initial probability matrix(vector)
    P_init = torch.zeros(N)
    # initialize the transition probability matrix
    P_trans = torch.zeros(N, N)
    # initialize the observation probability matrix
    P_observ = torch.zeros(N, M)

    # Format:
    # label_lists = [[l1, l2], [l3, l5, l4], ... , [l5, l2]]
    # word_lists  = [[w1, w2], [w3, w5, w4], ... , [w5, w2]]

    # estimate initial probability matrix
    for l_lst in label_lists:
        # get the index of the first label of each (internal) label list
        init_label_id = label2id[l_lst[0]]
        # add 1 on the position of this index in P_init
        P_init[init_label_id] += 1
    # Laplace-smoothing
    P_init[P_init == 0.] = 1e-10
    # calculate the initial probabilities
    P_init /= P_init.sum()

    # estimate transition probability matrix
    for l_lst in label_lists:
        for i in range(len(l_lst) - 1):
            # get the index of each 2 labels in each (internal) label list
            current_label_id = label2id[l_lst[i]]
            next_label_id = label2id[l_lst[i+1]]
            # use them as row_index and column_index for P_trans
            # add 1 on the position in P_trans
            P_trans[current_label_id][next_label_id] += 1
        # Laplace-smoothing
        P_trans[P_trans == 0.] = 1e-10
        # calculate the transition probabilities
        P_trans /= P_trans.sum(dim=1, keepdim=True)

    # estimate observation probability matrix
    for l_lst, w_lst in zip(label_lists, word_lists):
        for label, word in zip(l_lst, w_lst):
            # get the indices of each label in l_lst and its corresponding word in w_lst
            label_id = label2id[label]
            word_id = word2id[word]
            # use them as row_index and column_index for P_observ
            # add 1 on the position in P_observ
            P_observ[label_id][word_id] += 1
    # Laplace-smoothing
    P_observ[P_observ == 0.] = 1e-10
    # calculate the observation probabilities
    P_observ /= P_observ.sum(dim=1, keepdim=True)

    print("Initial probability matrix \nP_init:", P_init, len(P_init))
    print("\n")
    print("Transition probability matrix \nP_trans:", P_trans, len(P_trans))
    print("\n")
    print("Observation probability matrix \nP_observ:", P_observ, len(P_observ))

    ### print for each label the first 10 words with hightest probabilities
    # format top10_dict: {  label1: [(word1, prob1), (word2, prob2), ... ,(word10, prob10)],
    #                       label2: [(word1, prob1), (word2, prob2), ... ,(word10, prob10)],
    #                       ...
    #                       label24: [(word1, prob1), (word2, prob2), ... ,(word10, prob10)]}
    top10_dict = dict()
    for l_id in range(len(P_observ)):
        # get the first 10 highest probabilities for each label
        # Format: top10_prob_id= (values=tensor([...]), indices=tensor([...])), values=prob
        top10_prob_id = P_observ[l_id].topk(10)
        # word list that corresponds to current label
        w_list_currlab = []
        for prob, w_id in zip(top10_prob_id[0].tolist(), top10_prob_id[1].tolist()):
            w_list_currlab.append((w_list_nodup[w_id], prob))
        top10_dict[l_list_nodup[l_id]] = (w_list_currlab)
    print("\n")
    print("TOP10_DICT:", top10_dict, len(top10_dict))



train(word_lists, label_lists, word2id, label2id)
end = time.time()
print("consumed time:", str(end-start))
