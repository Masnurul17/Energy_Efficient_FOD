# === 引入所需的函式庫 ===
from PIL import Image  # 處理影像
import time  # 計算時間與FPS
import numpy as np  # 數值運算與矩陣操作
from hailo_platform import __version__  # 取得Hailo SDK版本
from multiprocessing import Process  # 多處理程序（本程式未使用）
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)  # Hailo推論相關模組
from zenlog import log  # 彩色終端輸出（可用於除錯）
import cv2  # OpenCV，用於影像與影片處理
from class_name import *  # 匯入分類標籤的對應名稱

# === 1. 模型與參數設定 ===
#classifier_model='./hef_model/'+'mobilenetv2.hef'  # Hailo已編譯模型路徑 (.hef)
classifier_model='./hef_model/'+'resnet50.hef'
out_parser ='resnet50/fc1'  # 模型輸出層名稱，用於取得分類結果
datatest = 'hummingbird'  # 要偵測的目標類別標籤

# 動作偵測相關參數
treshold_val=200  # 二值化閾值
max_val=225  # 二值化最大值
area_tresh_min=0.000001  # ROI最小面積（占整張影像比例）
area_tresh_max=1  # ROI最大面積（占整張影像比例）
class_detection=True  # 是否啟用分類（此程式碼未實際使用該變數）
dim = (224, 224)  # 輸入影像尺寸（符合模型輸入要求）

# 圖像正規化參數（基於ImageNet的均值與標準差）
mean=[0.485, 0.456, 0.406]
stdv=[0.229, 0.224, 0.225]
mean_vec = np.array(mean)
stddev_vec = np.array(stdv)

# === 2. 載入模型並取得輸入形狀 ===
hef = HEF(classifier_model)  # 載入HEF模型
height, width, channels = hef.get_input_vstream_infos()[0].shape  # 取得模型輸入尺寸

devices = Device.scan()  # 掃描可用的Hailo裝置

# === 3. Softmax 函式，用於將模型輸出轉為機率 ===
def softmax(x):
    x = x.reshape(-1)  # 將輸入拉平成一維向量
    e_x = np.exp(x - np.max(x))  # 避免數值溢位
    return e_x / e_x.sum(axis=0)  # 正規化為機率分布

# === 4. 計算背景影像的工具函式（此程式碼未使用）===
def get_background(file_path):
    cap = cv2.VideoCapture(file_path)  # 開啟影片檔
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)  # 隨機選取50個影格
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    median_frame = np.median(frames, axis=0).astype(np.uint8)  # 計算中位數影像作為背景
    return median_frame

# === 5. 主推論流程開始 ===
with VDevice(device_ids=devices) as target:  # 使用Hailo虛擬裝置
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)  # 設定PCIe界面
        network_group = target.configure(hef, configure_params)[0]  # 配置模型
        network_group_params = network_group.create_params()  # 產生推論參數

        # 設定輸入與輸出串流參數
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

        height, width, channels = hef.get_input_vstream_infos()[0].shape  # 重新取得輸入形狀（可省略）

        capture = cv2.VideoCapture(datatest+'.mp4')  # 開啟目標影片
        frame_diff_list = []  # 儲存影格差異，用於動作偵測
        frame_length = 1  # 疊加多少影格來判斷變動

        previous_frame_time = 0
        new_frame_time = 0
        time_fps = []  # 儲存FPS資訊
        true_count=0  # 正確分類次數
        frame_number=0  # 處理過的總影格數
        lastx, lasty, lastw, lasth = None,None,None,None  # 上一個ROI位置（未使用）

        while True:
            success, lstframe = capture.read()  # 讀取前一張影格
            success, frame = capture.read()  # 讀取當前影格

            if success:
                frame_number +=1 
                new_frame_time = time.time()  # 記錄時間，用於計算FPS
                orig_frame = frame.copy()  # 備份原始影格

                # === 6. 動作偵測（影格差異法）===
                diff = cv2.absdiff(frame, lstframe)  # 計算影格差異
                diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # 轉為灰階
                diff = cv2.GaussianBlur(diff,(3,3),0)  # 模糊化降噪
                diff = cv2.dilate(diff, None, iterations=1)  # 擴張處理
                diff = cv2.threshold(diff, treshold_val, max_val, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]  # 二值化

                frame_diff_list.append(diff)  # 將差異影格加入暫存

                if len(frame_diff_list) == frame_length:
                    sum_frames = sum(frame_diff_list)  # 累加所有差異影格
                    frame_diff_list = []  # 清空暫存

                    # === 7. 擷取ROI（移動區域）===
                    x,y,w,h = cv2.boundingRect(sum_frames)  # 計算最小包圍盒
                    area=w*h  # 計算ROI面積
                    cols, rows, _= orig_frame.shape
                    frame_are = cols * rows 

                    if (frame_are * area_tresh_min) < area < (frame_are * area_tresh_max):  # 面積判斷過濾雜訊
                        cv2.rectangle(orig_frame, (500, 300), (x + w, y + h), (36,255,12), 1)  # 繪製方框

                        # 擷取並前處理ROI區域
                        crop_img = orig_frame[y:y+h, x:x+w]
                        img_rgb = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
                        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

                        image = np.array(img_rgb,  np.float32)
                        norm_img_data = np.zeros(image.shape).astype('float32')

                        # === 影像正規化處理 ===
                        for i in range(image.shape[2]):
                            norm_img_data[:,:,i] = (image[:,:,i]/255 - mean_vec[i]) / stddev_vec[i]

                        # === 8. 使用Hailo進行推論 ===
                        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                            input_data = {input_vstream_info.name: np.expand_dims(np.asarray(norm_img_data), axis=0).astype(np.float32)}    
                            with network_group.activate(network_group_params):
                                infer_results = infer_pipeline.infer(input_data)

                        out=np.argmax(softmax(infer_results.get(out_parser)))  # 取得預測分類索引
                        classified = class_name[out]  # 對應分類名稱
                        print(classified)

                        if classified == datatest:
                            true_count+=1  # 統計正確分類次數

                        cv2.putText(orig_frame, classified, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) )  # 顯示分類結果

                    # === 9. FPS 計算 ===
                    fps = 1/((new_frame_time)-(previous_frame_time))  # 計算FPS
                    previous_frame_time = new_frame_time 
                    time_fps.append(fps)

                    cv2.imshow("Resized image", orig_frame)  # 顯示即時影像結果

                    if cv2.waitKey(1) == 27:  # 按ESC離開
                        break
            else:
                break  # 影片播放結束

        # === 10. 最終統計與輸出 ===
        mean_fps = np.mean(time_fps)  # 計算平均FPS
        print(f"FPS:{mean_fps:0.3f}, Acc:{true_count*100/frame_number:0.3f}")  # 顯示準確率與速度

        capture.release()  # 關閉影片
        cv2.destroyAllWindows()  # 關閉所有OpenCV視窗