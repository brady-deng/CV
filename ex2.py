import cv2
img = cv2.imread("3.jpg")
font = cv2.FONT_HERSHEY_PLAIN
text1 = "Beauty:"
text2 = "Age:"
cv2.putText(img,text1,(30,30),font,4,(0,255,0),4,cv2.LINE_AA)
cv2.putText(img,text2,(30,80),font,4,(0,255,0),4,cv2.LINE_AA)
cv2.imshow("temp",img)
cv2.waitKey(0)
cv2.destroyAllWindows()