#!/usr/bin/python3
# -*- coding: utf8 -*-
import cv2
import time
import datetime
import numpy as np
import os,csv
import shutil

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

class detection:
    #初期化関数
    def __init__(self):
        Config_tmp = []
        with open('Config/Config.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                Config_tmp.append(row)
        
        Area_tmp = []
        with open('Config/Area.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                Area_tmp.append(row)
        
        # メインウィンドウ作成
        #プログラム名の指定
        self.windowname = "detection into the bed"

        #カメラの解像度
        self.camera_width = int(Config_tmp[0][1])
        self.camera_height = int(Config_tmp[1][1])
        self.img_ratio = 80/640

        self.jarea_min_x = int(Area_tmp[0][1])
        self.jarea_min_y = int(Area_tmp[1][1])
        self.jarea_max_x = int(Area_tmp[2][1])
        self.jarea_max_y = int(Area_tmp[3][1])

        self.fps = ""
        self.vidfps = int(Config_tmp[2][1])
        self.elapsedTime = 0
        self.message = "Push [p] to take a background picture."
        self.flag_detection = False
        
        self.camera_size = (640,480)
        #最大検出人数
        self.Number_of_people = int(Config_tmp[3][1])

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, self.vidfps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        #self.fourcc = cv2.VideoWriter_fourcc(*'XVID')##movie save
        #self.writer = cv2.VideoWriter('output.avi',self.fourcc, self.vidfps,
        #                              (self.camera_width,self.camera_height))##movie save
        time.sleep(1)


    #マウスクリックEVENTを取得 
    def CallBackFunc(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.jarea_min_x = x
            self.jarea_min_y = y
            self.jarea_max_x = x
            self.jarea_max_y = y           
        elif event == cv2.EVENT_LBUTTONUP:
            self.jarea_max_x = x
            self.jarea_max_y = y
            #もし座標の位置がずれている場合は正しい形に入れ替え
            if (self.jarea_min_x >self.jarea_max_x) or (self.jarea_min_y >self.jarea_max_y):
                tmp_x = self.jarea_min_x
                tmp_y = self.jarea_min_y
                self.jarea_min_x =self.jarea_max_x 
                self.jarea_min_y =self.jarea_max_y 
                self.jarea_max_x = tmp_x
                self.jarea_max_y = tmp_y
            f = open('Config/Area.csv', 'w') # 書き込みモードで開く
            f.write("jarea_min_x," + str(self.jarea_min_x) +"\n") 
            f.write("jarea_min_y," + str(self.jarea_min_y) +"\n") 
            f.write("jarea_max_x," + str(self.jarea_max_x) +"\n") 
            f.write("jarea_max_y," + str(self.jarea_max_y) +"\n") 
            f.close() # ファイルを閉じる


    def id_class_name(self, class_id, classes):
        for key, value in classes.items():
            if class_id == key:
                return value

    #領域と点の内外判定
    def jugment_point(self, center_point_x, center_point_y):
        check = False
        if (self.jarea_min_x <= center_point_x) and (center_point_x <= self.jarea_max_x )and (self.jarea_min_y <= center_point_y ) and ( center_point_y <= self.jarea_max_y):
            check = True
        else:
            check = False
        return check

    #検出領域の中心に円で囲む
    def bounding_circle(self, img, x_min, y_min, x_max, y_max):
        #中心を出す．
        center_point_x = (x_min + x_max)//2
        center_point_y = (y_min + y_max)//2
        img =cv2.circle(img, (center_point_x, center_point_y), 5, (0, 255, 0), -1)
        checkresult = self.jugment_point(center_point_x, center_point_y)
        return img ,center_point_x ,center_point_y, checkresult

    def write_file(self,i):
        f = open('data/Log_Date.txt', 'a') # 書き込みモードで開く
        #現在の日付と日時
        dt_now = datetime.datetime.now()
        f.write("ID " + str(i) + ":ベッドから人が下りました:" + str(dt_now) +"\n") # 引数の文字列をファイルに書き込む
        f.close() # ファイルを閉じる

    def write_video(self,i,no):
        dt_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('./video/'+str(dt_now)+'.avi', fourcc , 20.0, (self.camera_width, self.camera_height))
        
        path = os.path.abspath('./video/'+ str(i) )         
        
        #30Frameを15秒間
        videorange = no - (30*15)
        if(videorange < 0):
                videorange = 0
        
        for x in range(videorange, no):
            img = cv2.imread(path +'/'+ str(x) +'.jpg')
            video.write(img)
            
        video.release()
        shutil.rmtree(path)
        os.makedirs(path)
        return video 

    def print_result(self,frame,i,back_state,state):
        #ベッドの外か，ベッドの中かを表示する
        if (state == True):
            cv2.putText(frame,("ID  " + str(i) + "  in the bed" ),(0, self.camera_height -20*i) ,cv2.FONT_HERSHEY_SIMPLEX,(.05*10),(0, 255, 255))
        else:
            cv2.putText(frame,("ID  " + str(i) + "  out of the bed" ) ,(0, self.camera_height -20*i) ,cv2.FONT_HERSHEY_SIMPLEX,(.05*10),(0, 255, 255))

    # メイン関数
    def main(self):
        # VideoCaptureのインスタンスを作成する。
        # 引数でカメラを選べれる。
        self.cap.set(cv2.CAP_PROP_FPS, self.vidfps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)    

        # Loading model
        model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',
                                       'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

        state = []
        back_state =[]
        confirm_cnt = []
        fail_cnt = []
        check = []
        center_px =[]
        center_py =[]
        n = []
        #初期化
        for i in range(self.Number_of_people):
            state.append(False)
            back_state.append(False)
            confirm_cnt.append(0)
            fail_cnt.append(0)
            check.append(False)
            center_px.append(0)
            center_py.append(0)
            n.append(0)
            if not os.path.exists("video/" + str(i)):
                os.mkdir("video/" + str(i))        

        # print(output[0,0,:,:].shape)
        while True:
            # VideoCaptureから1フレーム読み込む
            ret, frame = self.cap.read()

            # 検出領域
            frame =cv2.rectangle(frame, ( self.jarea_min_x, self.jarea_min_y), (self.jarea_max_x, self.jarea_max_y), (255, 0, 0), 3)                      
            model.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True))
            output = model.forward()
        
            # スクリーンショットを撮りたい関係で1/4サイズに縮小
            #frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))
            #検出されたオブジェクトの数だけループする
            for i in range(0,(output.shape[2])):
                #検出する最大人数分保持
                if (i < self.Number_of_people):
                    confidence = output[0, 0, i, 2]
                    #検出したオブジェクトの確信度が50%以上の時にのみ処理する
                    if confidence > 0.6:
                        class_id = int(output[0, 0, i, 1])
                        #人以外の検出は無視する
                        if (class_id == 1):
                            class_name = self.id_class_name(class_id, classNames)

                            #何%の確率でこれは何です という結果を表示
                            #print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                            box_x = int(output[0, 0, i, 3] * self.camera_width)
                            box_y = int(output[0, 0, i, 4] * self.camera_height)
                            box_width = int(output[0, 0, i, 5] * self.camera_width)
                            box_height = int(output[0, 0, i, 6] * self.camera_height)                        
                                
                            #検出した領域を四角で囲む
                            cv2.rectangle(frame, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)

                            # 検出したオブジェクトの中心を円で表示
                            frame, center_px[i], center_py[i], check[i]  = self.bounding_circle(frame, int(box_x), int(box_y), int(box_width), int(box_height))
                            # 検出した円にオブジェクト名前を表示
                            label = "User " + str(i) + ":" + class_name
                            cv2.putText(frame,label ,(center_px[i], center_py[i]) ,cv2.FONT_HERSHEY_SIMPLEX,(.05*15),(0, 255, 0))

                            #領域内に人がいるか否か
                            if (check[i] == True): #領域内で人が検出されたとき
                                confirm_cnt[i] = confirm_cnt[i] + 1
                                #FPS 30なので 1秒間に30回更新します．（90フレームなら3秒程度）
                                if (confirm_cnt[i] > 90):
                                    back_state[i] = state[i]
                                    state[i] = True #ベッドの上に人がいることを確定
                                    fail_cnt[i] = 0
                                else:
                                    back_state[i] = state[i]
                                    state[i] = False
                            else: #領域外で人が検出されたとき
                                fail_cnt[i] = fail_cnt[i] + 1
                                if (fail_cnt[i] > 60): #1秒間程度経過したら
                                    back_state[i] = state[i]
                                    state[i] = False #ベッドの上には人がいない
                                    confirm_cnt[i] = 0
                                else:
                                    back_state[i] = state[i]
                                    state[i] = True
                                    

                    #ベッドの外か，ベッドの中かを表示する
                    self.print_result(frame,i, back_state[i], state[i])

                    #ベッドの中から外に出た時はこの対応をする
                    if (back_state[i] == True):
                        #フレーム画像を保存する．
                        cv2.imwrite("video/"+str(i) +"/"+ str(n[i]) +".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        n[i] = n[i] + 1
                        if(state[i] == False): #ベッドの外に出た時
                            self.write_file(i)
                            back_state[i] = state[i]
                            video = self.write_video(i,n[i])
                            n[i] = 0

            # 加工なし画像を表示する
            cv2.imshow(self.windowname, frame)
            cv2.setMouseCallback(self.windowname, self.CallBackFunc)

            # キー入力を1ms待って、k が27（ESC）だったらBreakする
            k = cv2.waitKey(1)
            if k == 27:
                break

        # キャプチャをリリースして、ウィンドウをすべて閉じる
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = detection()
    main.main()