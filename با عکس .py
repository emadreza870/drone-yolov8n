from ultralytics import YOLO
import cv2

# بارگیری مدل YOLO
model = YOLO('E:/python/drone yolov8n/Drone.pt')  # مسیر مدل را بررسی کنید

# انتخاب تصویر ورودی
image_path = 'D:/drone/pic_311.jpg'  # مسیر تصویر خود را جایگزین کنید
image = cv2.imread(image_path)

if image is None:
    print("خطا در بارگذاری تصویر! مسیر را بررسی کنید.")
else:
    # پردازش تصویر با YOLO
    results = model(image)

    # نمایش تصویر با نتایج
    annotated_image = results[0].plot()  # نمایش باکس‌ها و لیبل‌ها روی تصویر
    cv2.imshow('Result', annotated_image)
    cv2.waitKey(0)  # منتظر می‌ماند تا کاربر دکمه‌ای را فشار دهد
    cv2.destroyAllWindows()

    # ذخیره تصویر پردازش شده (اختیاری)
    output_path = 'E:/python/drone yolov8n/output.jpg'
    cv2.imwrite(output_path, annotated_image)
    print(f"تصویر خروجی ذخیره شد: {output_path}")
