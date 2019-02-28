import numpy as np
from joblib import Parallel, delayed
from munkres import Munkres, print_matrix
from tqdm import tqdm
import pandas as pd

def union(li_1, li_2):
    uni = list(set().union(li_1, li_2))
    return len(uni)


def pairwise_scores(li, n_jobs=1):
    """
    :param li: List of integers
    :return:
    """

    nb_vertical = len(li)
    scores = np.zeros((nb_vertical, nb_vertical), dtype=np.int)

    pairs_ids = [(i, j) for i in range(nb_vertical) for j in range(i + 1, nb_vertical)]
    if n_jobs == 1:
        for i, j in tqdm(pairs_ids):
            li1 = li[i]
            li2 = li[j]
            score = union(li_1=li1, li_2=li2)
            scores[i, j] = score
            scores[j, i] = score
    else:
        def my_loop(tup):
            i, j = tup
            li1 = li[i]
            li2 = li[j]
            score = union(li_1=li1, li_2=li2)
            return score

        scores_li = Parallel(n_jobs=n_jobs)(delayed(my_loop)(k) for k in pairs_ids)
        for (i, j), score in zip(pairs_ids, scores_li):
            scores[i, j] = score
            scores[j, i] = score

    return scores


def kuhn_assignment(scores_mat):
    my_mat = - scores_mat.copy()
    my_mat = my_mat + 1e8 * np.eye(len(my_mat))

    m = Munkres()
    indexes = m.compute(my_mat)

    pairs = []
    for row, col in indexes:
        if (col, row) not in pairs:
            pairs.append((row, col))
    return pairs


def get_assignments(df):
    df = pd.read


if __name__ == '__main__':
    data_dct = np.load('./Save/data_dct.npy').item()
    for dataframe_name in data_dct:
        df = data_dct[dataframe_name]

        df_v = df.loc[df['H_V'] == 'V']
        tags_v = df_v['Tags_Int']
        tags_v = tags_v.str.split()
        tags_v = tags_v.apply(lambda x: [int(val) for val in x]).values

        scores = pairwise_scores(tags_v)
        # print(df.head())