import cv2 as cv
import numpy as np

filename = 'row_soy2.jpg'
img = cv.imread(filename)
hsv_green_lower = np.array([30,0,0])
hsv_green_upper = np.array([179,255,255])

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

mask = cv.inRange(hsv, hsv_green_lower, hsv_green_upper)

# res = cv.bitwise_and(img,img,mask=mask)
# bgr_mask = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
# gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

# gray = np.float32(gray)
# gray = cv.medianBlur(gray,5)
mask = cv.Canny(mask,50,150,apertureSize = 3)
# mask = cv.Sobel(mask,cv.CV_8U ,1,1)

# print gray
# Threshold para um valor otimo (varia de acordo com a imagem)
# mask[mask<60] = [0]
# cv.imshow('dst',gray)

# mask[np.int32(np.floor(np.mean(np.where(mask>=60), axis=0)))] = [255]
# ret, gray = cv.threshold(mask, 59, 255, cv.THRESH_BINARY)
# gray = cv.adaptiveThreshold(mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
# cv.imwrite('soyrustdots.png',img)
# print gray

lines = cv.HoughLines(mask,.05,np.pi/180,150)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv.imshow('mask',img)
# cv.imshow('res',res)
if cv.waitKey(0) & 0xff == 27:
	cv.destroyAllWindows()
