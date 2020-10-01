import numpy as np


def customPremute(dim, max=5):
    # take a number and put it in the top 5 available spots
    # prem = [-1 for i in range(dim)]
    perm = []

    # numbers from 0 to dim-1
    numbers = [i for i in range(dim)]

    # starts at [0 to max-1]
    currentMax = np.array(numbers[:max])

    for i in range(dim):

        #  shuffle the current top
        asdf = np.array(currentMax)
        np.random.shuffle(asdf)
        currentMax = asdf.tolist()

        perm.append(currentMax.pop(0))

        if(i+max < dim):
            currentMax.append(numbers[i+max])

    return perm


print(customPremute(10, max=2))
