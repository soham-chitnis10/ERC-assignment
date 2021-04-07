import cv2
import numpy as np
import matplotlib.pyplot as plt
inputImage = cv2.imread("line_detection.jpeg")
inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
#Using Bilateral filter for keeping edges sharp.
gray_filtered = cv2.bilateralFilter(inputImageGray, 7, 10, 10)
#edge detection
edges = cv2.Canny(gray_filtered,122,220,apertureSize = 3)
#Since image produced from edge detection does not have any channels another variable is declared which will highlight the detected lines in final output image
detected_edges=cv2.cvtColor(inputImageGray,cv2.COLOR_GRAY2BGR)
#Colouring the edges and storing in a new variable
for i in range(269):
    for j in range(488):
        if edges[i][j]==255:
            detected_edges[i,j,0]=0
            detected_edges[i,j,1]=0
            detected_edges[i,j,2]=255
        else :
            detected_edges[i,j,0]=0
            detected_edges[i,j,1]=0
            detected_edges[i,j,2]=0 
res=cv2.addWeighted(inputImage,0.5,detected_edges,0.5,0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(res,"Yellow line", (150, 150), font, 0.5, (0,235,235))
cv2.putText(res,"White lines", (350, 200), font, 0.5, (235,235,235))
cv2.putText(res,"White lines", (15, 120), font, 0.5, (235,235,235))
#For increasing brightness
for i in range(269):
    for j in range(488):
        res[i,j,:]+=20
cv2.imshow('Output Image',res)
#Saving the final output image
cv2.imwrite('output.jpeg',res)        
cv2.waitKey(0)
cv2.destroyAllWindows()
