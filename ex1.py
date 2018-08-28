# #####################################################
# #读取图片，显示图片
# #####################################################
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread("1.jpg",1)
# cv2.namedWindow("img",cv2.WINDOW_NORMAL)
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# import numpy as np
# import cv2
#
# ######################################################
# #读取视频，显示视频
# ######################################################
# cap = cv2.VideoCapture(0)
# while(True):
#     ret,frame = cap.read()
#     cv2.namedWindow("img",cv2.WINDOW_FULLSCREEN)
#     cv2.imshow("img",frame)
#     if cv2.waitKey(1)&0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
######################################################
#添加标志
######################################################
# import numpy as np
# import cv2
#
# img = np.zeros((512,512,3),np.uint8)
# cv2.line(img,(0,0),(511,511),(255,0,0),5)#线条
# cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)#矩形
# cv2.circle(img,(314,0),25,(0,255,0),3)#圆
# cv2.ellipse(img,(154,0),(230,198),0,0,180,255,-1)#椭圆
# pts = np.array([[10,5],[20,30],[10,40],[50,80],[80,90]],np.int32)
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,255))#多边形
# font = cv2.FONT_HERSHEY_PLAIN
# cv2.putText(img,'OpenCV',(10,500),font,4,(255,255,255),2,cv2.LINE_AA)#写字
# cv2.imshow("temp1",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
######################################################
#鼠标响应
######################################################
# import cv2
# import numpy as np
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)#打印出所有的鼠标事件
#
# def draw_circle(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#
# img = np.zeros((512,512,3),np.uint8)
# cv2.namedWindow("image")
# cv2.setMouseCallback("image",draw_circle)
# while(1):
#     cv2.imshow("image",img)
#     if cv2.waitKey(20)&0xFF == 27:
#         break
# cv2.destroyAllWindows()
##########################################################
#More advanced demo
##########################################################
# drawing = False
# mode = True
# ix,iy = -1,-1
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:
#                 cv2.circle(img,(x,y),5,(0,0,255),-1)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv2.circle(img,(x,y),5,(0,0,255),-1)
# img = np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break
# cv2.destroyAllWindows()
##########################################################
#Track Bar
##########################################################
# import cv2
# import numpy as np
# def nothing(x):
#     pass
# img = np.zeros((300,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.createTrackbar('R','image',0,255,nothing)
# cv2.createTrackbar('G','image',0,255,nothing)
# cv2.createTrackbar('B','image',0,255,nothing)
# switch = '0:OFF \n1 : ON'
# cv2.createTrackbar(switch,'image',0,1,nothing)
# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1)&0xFF
#     if k == 27:
#         break
#
#     r = cv2.getTrackbarPos('R','image')
#     g = cv2.getTrackbarPos('G','image')
#     b = cv2.getTrackbarPos('B','image')
#     s = cv2.getTrackbarPos(switch,'image')
#     if s == 0:
#         img[:] = 0
#     else:
#         img[:] = [b,g,r]
# cv2.destroyAllWindows()
##########################################################
#Basic Operations on Images
##########################################################
# import cv2
# import numpy as np
# ##########Accessing and Modifying pixel values############
# img = cv2.imread('1.jpg')
# px = img[100,100]
# print(px)
# blue = img[100,100,0]
# print(blue)
# img[100,100] = [255,255,255]
# print(img[100,100])
# print(img.item(100,100,0))
# img.itemset((100,100,0),200)
# print(img.item(100,100,0))
# print(img.shape)
# print(img.size)
# print(img.dtype)
# ##########Image ROI#############################
# cv2.namedWindow("image",cv2.WINDOW_NORMAL)
# cv2.imshow("image",img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# face = img[1000:3000,1500:2500]
# img[0:2000,0:1000] = face
# cv2.namedWindow("temp1",cv2.WINDOW_NORMAL)
# cv2.imshow("temp1",img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# ############Splitting and Merging Image Channels###
# b,g,r = cv2.split(img)
# img2 = cv2.merge((b,g,r))
# temp_b = img[:,:,0]
# cv2.namedWindow("temp2",cv2.WINDOW_NORMAL)
# cv2.imshow("temp2",temp_b)
# cv2.waitKey()
# cv2.destroyAllWindows()
#########################################################
#Making borders for the picture
#########################################################
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# BLUE = [255,0,0]
#
# img1 = cv2.imread('1.jpg',1)
#
# replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
# constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
#
# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
#
# plt.show()
#####################################################################
#Image Addition
#####################################################################
import cv2
import numpy as np
# x = np.uint8([250])
# y = np.uint8([10])
# print(cv2.add(x,y))
# print(x+y)
# ##########Image Blending############################################
# img1 = cv2.imread("3.jpg")
# img2 = cv2.imread("2.jpg")
# dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
# cv2.namedWindow("temp3",cv2.WINDOW_FULLSCREEN)
# cv2.imshow("temp3",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#############Bitwise Operations#####################################
# rows,cols,channels = img2.shape
# roi = img1[0:rows,0:cols]
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret,mask = cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
# img2_fg = cv2.bitwise_and(img2,img2,mask=mask)
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows,0:cols] = dst
# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
##################################################################
#CV and Baidu API
##################################################################
import cv2
import function
import time
client = function.faceinit()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

# while(True):
#     ret,frame = cap.read()
#     cv2.imwrite("temp.jpg",frame)
#     res = function.facreg("temp.jpg", client)
#     loc = []
#     image = cv2.imread("temp.jpg")
#     if res['error_msg'] == 'SUCCESS':
#         for count in range(res['result']['face_num']):
#             loc.append(res['result']['face_list'][count]['location'])
#             pt1 = (int(loc[count]['left']), int(loc[count]['top']))
#             pt2 = (int(loc[count]['left'] + loc[count]['width']), int(loc[count]['top'] + loc[count]['height']))
#             pt3 = (pt2[0],pt2[1]-80)
#             pt4 = (pt2[0],pt3[1]+40)
#             cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
#             cv2.putText(image,"beauty:"+str(res['result']['face_list'][count]['beauty']),pt3,font,2,(0,255,0),2,cv2.LINE_AA)
#             cv2.putText(image,"age:"+str(res['result']['face_list'][count]['age']),pt4,font,2,(0,255,0),2,cv2.LINE_AA)
#         cv2.putText(image, "number of face:" + str(res['result']['face_num']), (40, 40), font, 2, (0, 255, 0), 2,
#                     cv2.LINE_AA)
#     else:
#         cv2.putText(image, "number of face:0", (40, 40), font, 4, (0, 255, 0), 2,cv2.LINE_AA)
#     cv2.namedWindow("img",cv2.WINDOW_FULLSCREEN)
#     cv2.imshow("img",image)
#     k = cv2.waitKey(300)
#     if k&0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

while(True):
    ret,frame = cap.read()
    cv2.imwrite("temp.jpg",frame)
    data = function.get_file_content("temp.jpg")
    res_reg = function.face_reg(data, client)
    res_search = function.face_search(data,client)
    loc = []
    image = cv2.imread("temp.jpg")
    if res_reg['error_msg'] == 'SUCCESS':
        for count in range(res_reg['result']['face_num']):
            loc.append(res_reg['result']['face_list'][count]['location'])
            pt1 = (int(loc[count]['left']), int(loc[count]['top']))
            pt2 = (int(loc[count]['left'] + loc[count]['width']), int(loc[count]['top'] + loc[count]['height']))
            pt3 = (pt2[0],pt2[1]-80)
            pt4 = (pt2[0],pt3[1]+40)
            pt5 = (pt2[0],pt3[1]-40)
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
            if res_search['error_msg'] == 'SUCCESS':
                if res_reg['result']['face_list'][count]['face_token'] == res_search['result']['face_token']:
                    cv2.putText(image, "name:" + str(res_search['result']['user_list'][0]['user_info']), pt5, font, 2,
                                (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, str(res_search['error_code']), pt5, font, 2,(0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image,"beauty:"+str(res_reg['result']['face_list'][count]['beauty']),pt3,font,2,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(image,"age:"+str(res_reg['result']['face_list'][count]['age']),pt4,font,2,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(image, "number of face:" + str(res_reg['result']['face_num']), (40, 40), font, 2, (0, 255, 0), 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(image, "number of face:0", (40, 40), font, 4, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.namedWindow("img",cv2.WINDOW_FULLSCREEN)
    cv2.imshow("img",image)
    k = cv2.waitKey(300)
    if k&0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()