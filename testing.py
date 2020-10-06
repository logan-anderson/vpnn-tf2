import numpy as np


def customPermute(dim: int, width: int, maxRange=4):
    def getNum(currentIndex):
        return (currentIndex % width) * (width) + currentIndex // width
    # take a number and put it in the top 5 available spots
    perm = []

    # numbers from 0 to dim-1
    numbers = [i for i in range(dim)]

    currentMax = np.array([getNum(i) for i in range(maxRange)])

    currentIndex = maxRange

    for i in range(dim):
        print(currentMax)

        #  shuffle the current top
        temp = np.array(currentMax)
        np.random.shuffle(temp)
        currentMax = temp.tolist()

        perm.append(currentMax.pop(0))

        if(i + maxRange < dim):
            currentMax.append(getNum(currentIndex))

        currentIndex = currentIndex + 1

    return perm


print(customPermute(16, 4, maxRange=2))
