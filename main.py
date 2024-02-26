
import cv2
import datetime
import time

background_subtractor = cv2.createBackgroundSubtractorMOG2()
min_contour_area = 500

detected_cars = []
last_detection_times = {"From Poland to Russia": None, "From Russia to Poland": None}
time_threshold = 2
fur_detection = set()


def detect_cars(frame, side, line_y, line_direction, side_name, distance_threshold=20):
    global detected_cars, last_detection_times
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = background_subtractor.apply(gray)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, None)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, None)

    _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            car = (x, y, x + w, y + h)
            if car not in detected_cars:
                if (line_direction == 'up' and cy < line_y - h) or (line_direction == 'down' and cy > line_y + h):
                    distance_to_line = abs(cy - line_y)
                    if distance_to_line < distance_threshold:
                        current_time = time.time()
                        last_detection_time = last_detection_times[side_name]
                        if last_detection_time is None or current_time - last_detection_time > time_threshold:
                            detected_cars.append(car)
                            last_detection_times[side_name] = current_time
                            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Car detected on {side_name} side at {now}")


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

            left_line_y = height // 2
            right_line_y = height // 2
            left_line_direction = 'down'
            right_line_direction = 'up'

            detect_cars(left_frame, "left", left_line_y, left_line_direction, "From Poland to Russia")
            detect_cars(right_frame, "right", right_line_y, right_line_direction, "From Russia to Poland")

            cv2.line(left_frame, (0, left_line_y), (width // 2, left_line_y), (0, 0, 0), 2)
            cv2.line(right_frame, (0, right_line_y), (width // 2, right_line_y), (0, 0, 0), 2)

            cv2.imshow("Video Capture", frame)
            cv2.waitKey(1)

        video_capture.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    video = 'https://media.gov39.ru/webcam-rec/mapp_gzhehodki.stream/playlist.m3u8'
    start_video_object_detection(video)
