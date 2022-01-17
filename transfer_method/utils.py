import numpy as np
import math
import random


def maxLength(seqs):
    max_len = 0
    for i in range(len(seqs)):
        st = seqs[i]
        if (len(st) > max_len): max_len = len(st)
    return max_len


def deleteSame(seqs):
    new_seqs = list(set(seqs))
    new_seqs.sort(key=seqs.index)
    return new_seqs


def getData(filePath):
    seqs = []
    seq = ''
    with open(filePath) as f:
        for each in f:
            if each.startswith('>'):
                seqs.append(seq)
                seq = ''
            else:
                seq += each.replace('\n', '')
        seqs.append(seq)
        seqs.pop(0)
    seqs = replace(seqs)
    return deleteSame(seqs)


def encoder(seqs, max_len):
    id = 'ACGT'
    data_e = []
    for i in range(len(seqs)):
        length = len(seqs[i])
        elemt, st = [], seqs[i]
        for j in st:
            try:
                index = id.index(j) + 1
                elemt.append(index)
            except Exception as err:
                print(st)
        if length < max_len:
            elemt += [0] * (max_len - length)
        data_e.append(elemt)
    return np.array(data_e)


def createNeg_zn(posSeq, k_let):
    negSeqs = []
    for seq in posSeq:
        seq_fragment = [seq[i:i + k_let] for i in range(0, len(seq), k_let)]
        fragment_num = len(seq_fragment)
        shuffle_num = int(math.ceil(fragment_num * 1))  # 调节打乱的百分比

        fragment_index = list(range(fragment_num))
        shuffle_index = random.sample(fragment_index, shuffle_num)

        arr = []
        for i in range(fragment_num):
            a = seq_fragment[i]
            if i in shuffle_index:
                l = list(seq_fragment[i])
                np.random.shuffle(l)
                a = ''.join(l)
            arr.append(a)
        neg = ''.join(arr)
        negSeqs.append(neg)
    return negSeqs


def shuffler(sequence, k_let):
    length = [sequence[i:i + k_let] for i in range(0, len(sequence), k_let)]
    np.random.shuffle(length)
    return ''.join(length)


def createNeg(posSeq, k_let):
    negSeq = []
    for pos in posSeq:
        neg = shuffler(pos, k_let)
        negSeq.append(neg)
    return negSeq


def createLabel(poslen, neglen):
    label = []
    label += [1] * poslen
    label += [0] * neglen
    return np.array(label)


def replace(seqs):
    new_seqs = []
    for seq in seqs:
        seq = seq.upper()
        seq = seq.replace('W', random.choice('AT')).replace('S', random.choice('CG')).replace('R', random.choice('AG')) \
            .replace('Y', random.choice('CT')).replace('K', random.choice('GT')).replace('M', random.choice('AC')) \
            .replace('B', random.choice('CGT')).replace('D', random.choice('AGT')) \
            .replace('H', random.choice('ACT')).replace('V', random.choice('ACG')) \
            .replace('N', random.choice('ACGT'))
        new_seqs.append(seq)
    return new_seqs
