import cv2
import mediapipe as mp

# تنظیمات MediaPipe برای تشخیص حالت
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# دسترسی به دوربین
cap = cv2.VideoCapture(1)

# اجرای MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # تبدیل تصویر به RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # پردازش تصویر برای تشخیص اسکلت
        results = pose.process(image)

        # برگرداندن تصویر به BGR برای نمایش
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # رسم اسکلت بدن
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks)

        # نمایش تصویر
        cv2.imshow('Body Skeleton', image)

        # فشردن کلید 'q' برای خروج از برنامه
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# آزاد کردن منابع
cap.release()
cv2.destroyAllWindows()
