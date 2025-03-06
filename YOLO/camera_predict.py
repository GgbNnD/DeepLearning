import cv2
from ultralytics import YOLO
 
# 加载模型
model = YOLO("yolo11n.pt")
 
# 打开摄像头
cap = cv2.VideoCapture("http://admin:admin@10.196.119.249:8081")  # 0通常是默认摄像头
 
# 设置跳帧数，例如每5帧处理一次
skip_frames = 5
frame_count = 0
 
# 循环读取摄像头帧
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
 
    # 跳过一些帧以降低帧率
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue
 
    # 降低图像分辨率
    scale_percent = 50  # 缩放比例
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
 
    # 使用模型进行检测
    results = model(resized_frame)
 
    # 绘制检测结果
    annotated_frame = results[0].plot()
 
    # 显示结果
    cv2.imshow("老猫检测-YOLOv11物体检测", annotated_frame)
 
    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 释放资源
cap.release()
cv2.destroyAllWindows()