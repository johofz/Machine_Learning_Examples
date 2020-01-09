import numpy as np
import math, random
from collections import Counter
import warnings

def euclidean_dist(p1, p2):
    summ = 0
    for i in range(0, len(p1)):
        summ += (p1[i] - p2[i])**2
    dist = math.sqrt(summ)
    return dist

def myKNearestNeighbor(data, predict, k=3):
    if(len(data)) >= k:
        warnings.warn(f'Parameter "k" is set to {k}, which is greater or equal to the data-size of {len(data)}')
    dists = []
    for group in data:
        for feature in data[group]:
            dists.append([euclidean_dist(feature, predict), group])
    votes = [i[1] for i in sorted(dists)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

def convert_data(df, test_size=0.2):
    full_data = df.astype(np.float64).values.tolist()
    random.shuffle(full_data)

    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    [train_set[i[-1]].append(i[:-1]) for i in train_data]
    [test_set[i[-1]].append(i[:-1]) for i in test_data]

    return train_set, test_set


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('fivethirtyeight')

    data_set = {'k':[[1,2], [2,3], [2,4]],
                'r':[[6,8], [5,7], [6,9]]}
    new_data_pt = [4,4]
    res = myKNearestNeighbor(data_set, new_data_pt, k=3)
    print(res)
    [[plt.scatter(j[0], j[1], color=i) for j in data_set[i]] for i in data_set]
    plt.scatter(new_data_pt[0], new_data_pt[1], color='g')
    plt.show()