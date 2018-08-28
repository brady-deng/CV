from aip import AipSpeech,AipOcr,AipFace
import base64
# -*- coding: utf-8 -*-
def faceinit():
    APP_ID = '11702269'
    API_KEY = 'waphyikGVmZ5sWeyozGbKGUY'
    SECRET_KEY = 'QGQB09ic3aMofghWVjFHnMEC1cGtkGkr '
    client = AipFace(APP_ID, API_KEY, SECRET_KEY)
    return client

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        base64_data = base64.b64encode(fp.read())
        data = base64_data.decode('utf-8')
    return data
def face_reg(image,client):


    ##########################################
    #人脸检测：检测图片中的人脸并标记出位置信息
    ##########################################



    ##########################################
    #将图片数据编码城base64格式
    ##########################################

    imageType = "BASE64"

    """ 如果有可选参数 """
    options = {}
    options["face_field"] = "age,beauty,faceshape,gender,glasses,race,expression"
    options["max_face_num"] = 3
    options["face_type"] = "LIVE"

    """ 带参数调用人脸检测 """
    temp2 = client.detect(image, imageType, options)
    return temp2
def face_search(image,client):

    imageType = "BASE64"
    # image = data1

    # imageType = "BASE64"

    groupIdList = "group1"

    """ 调用人脸搜索 """
    # print(client.search(data, imageType, groupIdList))

    """ 如果有可选参数 """
    options = {}
    options["quality_control"] = "NONE"
    options["liveness_control"] = "NONE"
    options["user_id"] = "user1"
    options["max_user_num"] = 3

    """ 带参数调用人脸搜索 """
    res = client.search(image, imageType, groupIdList, options)
    return res