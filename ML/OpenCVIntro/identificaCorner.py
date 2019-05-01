import cv2 as cv
import numpy as np

filename = 'soyrust.jpg'
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

#resultado dilatado para criar os cantos
dst = cv.dilate(dst, None)

# Threshold para um valor otimo (varia de acordo com a imagem)
img[dst>0.05*dst.max()] = [0,0,255]

cv.imwrite('soyrustdots.png',img)

cv.imshow('dst', img)
if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()
