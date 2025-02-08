from ultralytics import YOLO
import cv2

# بارگیری مدل YOLO
model = YOLO('E:/python/drone yolov8n/Drone.pt')  # مسیر مدل را بررسی کنید

# باز کردن وبکم
cap = cv2.VideoCapture(0)  # عدد 0 وبکم پیش‌فرض را انتخاب می‌کند

if not cap.isOpened():
    print("خطا در دسترسی به وبکم!")
    exit()

while True:
    # دریافت فریم از وبکم
    ret, frame = cap.read()
    if not ret:
        print("خطا در دریافت فریم!")
        break

    # پردازش فریم با YOLO
    results = model(frame)

    # نمایش فریم با نتایج
    annotated_frame = results[0].plot()  # نمایش باکس‌ها و لیبل‌ها روی تصویر
    cv2.imshow('Webcam Detection', annotated_frame)

    # خروج با فشردن کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزادسازی منابع
cap.release()
cv2.destroyAllWindows()
