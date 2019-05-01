import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

# Label each one either Red of Blue with numbers 0 and 1
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

# Take Blue families and plot them
blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

# plt.show()

# Add a new-comer to the neighbourhood

newcomer = np.random.randint(0, 100, (10, 2)).astype(np.float32)

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

newRed = newcomer[results.ravel() == 0]
plt.scatter(newRed[:, 0], newRed[:, 1], 80, 'r', 'o')
newBlue = newcomer[results.ravel() == 1]
plt.scatter(newBlue[:, 0], newBlue[:, 1], 80, 'b', 'o')

print 'result: ', results, '\n'
print 'neighbours: ', neighbours, '\n'
print 'distance: ', dist

plt.show()
