import re
import torch

### data preprocessing
# open file and store each line in a list
with open("../NER-de-train.tsv") as corpus:
    list = []
    for line in corpus.readlines():
        list.append(line.split())
#print(list)

# list of word lists for each sentence
word_lists = []
# list of label lists for each word in each sentence
label_lists = []
# list of words for each sentence
w_intern_list = []
# list of labels for each word in each sentence
l_intern_list = []
for lst in list:
    if len(lst) == 0:
        word_lists.append(w_intern_list)
        label_lists.append(l_intern_list)
    elif lst[0] == '#':
        w_intern_list = []
        l_intern_list = []
    elif len(lst) > 0:
        w_intern_list.append(lst[1])
        l_intern_list.append(lst[2])
"""
for i in range(len(word_lists)):
    print(i, word_lists[i])
    print(i, label_lists[i])
"""

# calculate number of word types and number of label types
M = 0  # number of word types
N = 0  # number of label types
word_list = []
label_list = []
# append all words in a list and all labels in a list
for lst in list:
    if len(lst) > 0 and re.match('\d', lst[0]):
        word_list.append(lst[1])
        label_list.append(lst[2])

# remove duplicated words and labels by converting lists to sets
word_set = set(word_list)
label_set = set(label_list)
M = len(word_set)
N = len(label_set)
print("M=", M, "\tN=", N)
#print("word_set=", word_set, type(word_set))
#print("label_set=", label_set, type(label_set))

# initialize the initial probability matrix
P_init = torch.zeros(N)
# initialize the transition probability matrix
P_trans = torch.zeros(N, N)
# initialize the observation probability matrix
P_observ = torch.zeros(N, M)

# convert sets to lists, so that each word corresponds to each label
w_list = []
l_list = []
for w in word_set:
    w_list.append(w)
for l in label_set:
    l_list.append(l)

# initialize dictionary that mapps word to index(position in matrix). Key: words, Value: indices
word2id = dict()
for i in range(len(w_list)):
    word2id[w_list[i]] = i
# initialize dictionary that mapps label to index(position in matrix). Key: labels, Value: indices
label2id = dict()
for j in range(len(l_list)):
    label2id[l_list[j]] = j

#print("word2id=", word2id)
#print("label2id=", label2id)


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

    # estimate initial probability matrix
    for label_list in label_lists:
        init_label_id = label2id[label_list[0]]
        P_init[init_label_id] += 1
    #Pi_init[Pi_init == 0.] = 1e-10
    P_init / P_init.sum()

    # estimate transition probability matrix
    for label_list in label_lists:
        for i in range(len(label_list)-1):
            current_label_id = label2id[label_list[i]]
            next_label_id = label2id[label_list[i+1]]
            P_trans[current_label_id][next_label_id] += 1
        #P_trans[P_trans == 0.] = 1e-10
        P_trans / P_trans.sum(dim=1, keepdim=True)

    # estimate observation probability matrix
    for label_list, word_list in zip(label_lists, word_lists):
        assert len(label_list) == len(word_list)
        for label, word in zip(label_list, word_list):
            label_id = label2id[label]
            word_id = word2id[word]
            P_observ[label_id][word_id] += 1
        #P_observ[P_observ == 0.] = 1e-10
        P_observ / P_observ.sum(dim=1, keepdim=True)

    print("Initial probability vector \nP_init:", P_init)
    print("\n")
    print("Transition probability matrix \nP_trans:", P_trans)
    print("\n")
    print("Observation probability matrix \nP_observ:", P_observ)
    return P_init, P_trans, P_observ

train(word_lists, label_lists, word2id, label2id)

# print for each label the first 10 words with hightest probabilities
# format top10_dict: {  label1: [(word1, prob1), (word2, prob2), ... ,(word10, prob10)],
#                       label2: [(word1, prob1), (word2, prob2), ... ,(word10, prob10)],
#                       ...
#                       label24: [(word1, prob1), (word2, prob2), ... ,(word10, prob10)]}
top10_dict = dict()
for l_id in range(len(P_observ)):
    top10_prob_id = P_observ[l_id].topk(10)  # format top10_prob_id: (values, indices), values are probabilities
    w_list_currlab = []  # word list corresponds to current label
    for prob, w_id in zip(top10_prob_id[0].tolist(), top10_prob_id[1].tolist()):
        w_list_currlab.append((w_list[w_id], prob))
    top10_dict[l_list[l_id]] = (w_list_currlab)
print("\n")
print("DICTIONARY:", top10_dict, len(top10_dict))


# calculate the probabilities for the first 10 instances from NER-de-dev.tsv
