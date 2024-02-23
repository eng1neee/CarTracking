import cv2
import datetime


background_subtractor = cv2.createBackgroundSubtractorMOG2()
min_contour_area = 500

detected_cars = []


def detect_cars(frame, side):
    global detected_cars
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = background_subtractor.apply(gray)
    _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            car = (x, y, x + w, y + h)
            if car not in detected_cars:
                detected_cars.append(car)
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{side} at {now}")


def start_video_object_detection(video):
    try:
        video_capture = cv2.VideoCapture(video)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            height, width, _ = frame.shape
            left_frame = frame[:, :width // 2]
            right_frame = frame[:, width // 2:]

            detect_cars(left_frame, "From Poland to Russia")
            detect_cars(right_frame, "From Russia to Poland")

            cv2.imshow("Video Capture", frame)
            cv2.waitKey(1)

        video_capture.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    video = input("Path to video (or URL): ")
    start_video_object_detection(video)
