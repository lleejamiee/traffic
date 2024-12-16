import tkinter as tk
from tkinter import filedialog, Canvas, Text, StringVar, OptionMenu
import cv2
from PIL import Image, ImageTk
import threading
import time
# import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt


class VideoPlayer:
    percentage_output = ""
    file_name = ""

    def __init__(self, root):
        self.root = root
        self.root.title("Video Player with YOLO Object Detection")

        self.left_canvas = Canvas(root, width=800, height=600)
        self.left_canvas.pack(side=tk.LEFT)

        self.right_canvas = Canvas(root, width=800, height=600, bg='black')
        self.right_canvas.pack(side=tk.RIGHT)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(fill=tk.X)

        # Add dropdown menus
        self.yolo_version_var = StringVar(value="YOLOv8")
        self.yolo_variant_var = StringVar(value="yolov8n")

        self.yolo_versions = ["YOLOv8", "YOLOv9", "YOLOv10"]
        self.yolo_variants = {
            "YOLOv8": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            "YOLOv9": ["YOLOv9t", "YOLOv9s", "YOLOv9m", "YOLOv9c", "YOLOv9e"],
            "YOLOv10": ["YOLOv10-N", "YOLOv10-S", "YOLOv10-M", "YOLOv10-B", "YOLOv10-L", "YOLOv10-X"]
        }

        self.yolo_version_menu = OptionMenu(self.btn_frame, self.yolo_version_var, *self.yolo_versions,
                                            command=self.update_yolo_variant_menu)
        self.yolo_version_menu.pack(side=tk.LEFT)

        self.yolo_variant_menu = OptionMenu(self.btn_frame, self.yolo_variant_var, *self.yolo_variants["YOLOv8"])
        self.yolo_variant_menu.pack(side=tk.LEFT)

        self.btn_browse = tk.Button(self.btn_frame, text="Open", command=self.load_video)
        self.btn_browse.pack(side=tk.LEFT)

        self.btn_detect = tk.Button(self.btn_frame, text="Run", command=self.detect_objects)
        self.btn_detect.pack(side=tk.LEFT)

        # Add the stop and reset button
        self.btn_stop_reset = tk.Button(self.btn_frame, text="X", command=self.stop_and_reset)
        self.btn_stop_reset.pack(side=tk.LEFT)

        # Add the text areas below the buttons
        self.text_area = Text(root, height=16, width=50)
        self.text_area.pack(fill=tk.X, padx=10, pady=5)

        self.traffic_area = Text(root, height=16, width=50)
        self.traffic_area.pack(fill=tk.X, padx=10, pady=5)

        # Add the logging area
        self.logging_area = Text(root, height=16, width=50)
        self.logging_area.pack(fill=tk.X, padx=10, pady=5)

        self.cap = None
        self.frame = None
        self.running = False
        self.yolo = YOLO(self.get_yolo_model_path())  # Load initial YOLO model
        self.lock = threading.Lock()

        # Define the classes to be detected
        self.allowed_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}
        self.black_frame = None

        self.left_fps = 0
        self.right_fps = 0
        self.counter = 0
        self.start_time = time.time()

    def update_yolo_variant_menu(self, selected_version):
        self.yolo_variant_var.set(self.yolo_variants[selected_version][0])
        self.yolo_variant_menu["menu"].delete(0, "end")
        for variant in self.yolo_variants[selected_version]:
            self.yolo_variant_menu["menu"].add_command(label=variant, command=tk._setit(self.yolo_variant_var, variant))

    def get_yolo_model_path(self):
        version = self.yolo_version_var.get()
        variant = self.yolo_variant_var.get()
        self.file_name = version + "_" + variant + ".txt"
        return f"{variant}.pt"

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.*")])
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.running = True
            self.play_video()

    def play_video(self):
        if self.cap.isOpened():
            with self.lock:
                ret, frame = self.cap.read()

                if ret:
                    self.frame = frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = self.resize_frame(frame_rgb, 800, 600)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.left_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                    self.left_canvas.imgtk = imgtk

                    if self.black_frame is None:
                        self.black_frame = np.zeros_like(frame_rgb)

                    black_frame_with_boxes = self.draw_white_boxes(self.black_frame, [])
                    imgtk_black = ImageTk.PhotoImage(image=black_frame_with_boxes)
                    self.right_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk_black)
                    self.right_canvas.imgtk = imgtk_black

            if self.running:
                self.root.after(10, self.play_video)

    def resize_frame(self, frame, width, height):
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def detect_objects(self):

        self.start_time = time.time()
        if not self.running or self.frame is None:
            return

        # Reload the YOLO model with the selected version and variant
        self.yolo = YOLO(self.get_yolo_model_path())

        def detection_thread():

            while self.running:
                start_time = time.time()
                with self.lock:
                    ret, frame = self.cap.read()
                if ret:
                    # Drawing green boxes on the video. Passes in the current frame to yolo?
                    results = self.yolo(frame)
                    frame_with_boxes = self.draw_boxes(frame, results)
                    frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    frame_with_boxes = self.resize_frame(frame_with_boxes, 800, 600)
                    img_with_boxes = Image.fromarray(frame_with_boxes)
                    imgtk_with_boxes = ImageTk.PhotoImage(image=img_with_boxes)
                    self.left_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk_with_boxes)
                    self.left_canvas.imgtk = imgtk_with_boxes

                    # Drawing grey boxes on the black box on the right
                    self.black_frame = self.draw_white_boxes(self.black_frame, results)
                    black_frame_with_boxes = cv2.cvtColor(self.black_frame, cv2.COLOR_BGR2RGB)
                    black_frame_with_boxes = self.resize_frame(black_frame_with_boxes, 800, 600)
                    img_black_with_boxes = Image.fromarray(black_frame_with_boxes)
                    imgtk_black_with_boxes = ImageTk.PhotoImage(image=img_black_with_boxes)
                    self.right_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk_black_with_boxes)
                    self.right_canvas.imgtk = imgtk_black_with_boxes

                    # Writing frame per second in the black box
                    self.right_fps = 1.0 / (time.time() - start_time)
                    self.right_canvas.create_text(10, 10, anchor=tk.NW, text=f"FPS: {self.right_fps:.2f}", fill="white")
                    # Writing Elapsed time
                    self.update_logging_area()

                    # Update the text area with the percentage of objects
                    percentageOfObj = self.calculate_percentage_of_obj(self.black_frame)
                    self.percentage_output = self.percentage_output + str(percentageOfObj) + "\n"
                    self.update_text_area(percentageOfObj)

                time.sleep(0.01)

        threading.Thread(target=detection_thread).start()

    def stop_and_reset(self):
        self.running = False

        file_name = self.file_name
        print("save to" + file_name)
        text_file = open(self.file_name, "w")
        text_file.write(self.percentage_output)
        text_file.close()
        self.percentage_output = ""

        self.plot_gragh(file_name)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.frame = None
        self.left_fps = 0
        self.right_fps = 0
        self.black_frame = None
        self.counter = 0

        self.left_canvas.delete("all")
        self.right_canvas.delete("all")

        self.text_area.delete('1.0', tk.END)
        self.traffic_area.delete('1.0', tk.END)
        self.logging_area.delete('1.0', tk.END)
        self.right_canvas.configure(bg='black')

    # results is from YOLO library, loops through all the objects that are detected
    def draw_boxes(self, frame, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = self.allowed_classes[cls_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def draw_white_boxes(self, black_frame, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = self.resize_coords(x1, y1, black_frame.shape[1], black_frame.shape[0])
                    x2, y2 = self.resize_coords(x2, y2, black_frame.shape[1], black_frame.shape[0])
                    cv2.rectangle(black_frame, (x1, int((y1 + y2) / 2)), (x2, y2), (255, 255, 255), -1)
        # Create a mask where the frame is greater than 0
        mask = black_frame > 0

        black_frame[mask] = np.log1p(black_frame[mask]) / np.log(1.05)

        return black_frame

    def resize_coords(self, x, y, width, height):
        orig_width, orig_height = self.frame.shape[1], self.frame.shape[0]
        x = int(x * width / orig_width)
        y = int(y * height / orig_height)
        return x, y

    def calculate_percentage_of_obj(self, black_frame):
        non_zero_pixels = black_frame[black_frame > 0]
        car_pixels = black_frame[black_frame > 100]
        if non_zero_pixels.size == 0:
            return 0
        percentageOfObj = 100 * cv2.countNonZero(car_pixels) / cv2.countNonZero(non_zero_pixels)
        return percentageOfObj

    def update_text_area(self, percentageOfObj):
        # self.text_area.delete('1.0', tk.END)
        self.text_area.insert(tk.END, f"Percentage of Road Coverage: \t{percentageOfObj:.2f}\n")
        self.text_area.see(tk.END)

        # self.traffic_area.delete('1.0', tk.END)
        traffic_description = self.get_traffic_description(percentageOfObj)
        self.traffic_area.insert(tk.END, f"Traffic description: \t{traffic_description}\n")
        self.traffic_area.see(tk.END)

    def get_traffic_description(self, percentageOfObj):
        if percentageOfObj <= 1:
            return "Empty."
        elif percentageOfObj <= 5:
            return "Very Light Traffic."
        elif percentageOfObj <= 10:
            return "Light Traffic."
        elif percentageOfObj <= 15:
            return "Moderate Traffic."
        elif percentageOfObj <= 20:
            return "Busy."
        elif percentageOfObj <= 25:
            return "Very Busy."
        else:
            return "Congested."

    def update_logging_area(self):
        elapsed_time = time.time() - self.start_time
        self.percentage_output = self.percentage_output + str(elapsed_time) + ", "
        log_message = f"Elapsed time (s): \t{elapsed_time:.2f}"
        self.logging_area.insert(tk.END, log_message + '\n')
        self.logging_area.see(tk.END)

    def on_closing(self):

        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def plot_gragh(self, file_name):
        times = []
        coverage = []
        with open(file_name, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:  # Check if line has exactly two elements
                    try:
                        current_time, percent = map(float, parts)
                        times.append(current_time)
                        coverage.append(percent)
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")
                else:
                    print(f"Skipping line with unexpected format: {line.strip()}")

        plt.figure(figsize=(10, 5))
        plt.plot(times, coverage, color='black', linestyle='-')
        plt.xlabel('Time (ms)')
        plt.ylabel('Percentage of Road Coverage')
        plt.title('Percentage of Road Coverage')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", player.on_closing)
    root.mainloop()

