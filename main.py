

import cv2
import csv
import os
import time
import threading
import tkinter as tk
import numpy as np

from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
from collections import Counter

MODEL_PATH = "yolov8n (1).onnx"
CSV_FILE = "detections_report.csv"

CONFIDENCE = 0.60
IOU_THRESHOLD = 0.45
MAX_DISTANCE = 120
MAX_MISSING_FRAMES = 30
TRAIL_LENGTH = 20


def get_color(image):
    if image.size == 0:
        return "Unknown"

    pixels = image.reshape(-1, 3).astype(np.float32)
    pixels = pixels[pixels.max(axis=1) < 245]
    pixels = pixels[pixels.max(axis=1) > 35]

    if len(pixels) == 0:
        return "Unknown"

    b, g, r = np.mean(pixels, axis=0)

    if r > g * 1.25 and r > b * 1.25:
        return "Red"
    if g > r * 1.15 and g > b * 1.15:
        return "Green"
    if b > r * 1.15 and b > g * 1.10:
        return "Blue"
    if r > 150 and g > 150 and b < 120:
        return "Yellow"
    if r < 90 and g < 90 and b < 90:
        return "Black"
    if r > 180 and g > 180 and b > 180:
        return "White"

    return "Other"


class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection Dashboard")
        self.root.geometry("1050x700")
        self.root.minsize(850, 600)

        self.model = None
        self.camera = None
        self.running = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_counts = Counter()
        self.latest_fps = 0.0

        self.tracks = {}
        self.next_id = 0
        self.frame_number = 0

        self.records = []
        self.crossed_ids = set()
        self.down_count = 0
        self.up_count = 0
        self.last_error = None

        self.build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.after(40, self.update_gui)

    def build_gui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=10)
        header.grid(row=0, column=0, sticky="ew")

        ttk.Label(
            header,
            text="YOLO + OpenCV Object Detection",
            font=("Arial", 18, "bold")
        ).pack()

        self.status = tk.StringVar(value="Status: Stopped")
        ttk.Label(header, textvariable=self.status).pack(pady=5)

        body = ttk.Frame(self.root, padding=10)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(
            body, text="Click Start Camera",
            anchor="center", relief="solid"
        )
        self.video_label.grid(
            row=0, column=0, sticky="nsew", padx=(0, 10)
        )

        panel = ttk.LabelFrame(
            body, text="Live Dashboard", padding=10
        )
        panel.grid(row=0, column=1, sticky="nsew")

        self.fps_text = tk.StringVar(value="FPS: 0")
        self.total_text = tk.StringVar(value="Objects in view: 0")
        self.down_text = tk.StringVar(value="Crossed down: 0")
        self.up_text = tk.StringVar(value="Crossed up: 0")

        for variable in (
            self.fps_text, self.total_text,
            self.down_text, self.up_text
        ):
            ttk.Label(panel, textvariable=variable).pack(
                anchor="w", pady=5
            )

        ttk.Label(panel, text="Category counts:").pack(
            anchor="w", pady=(10, 4)
        )

        self.count_box = tk.Text(
            panel, width=22, height=14, state="disabled"
        )
        self.count_box.pack(fill="both", expand=True)

        buttons = ttk.Frame(self.root, padding=10)
        buttons.grid(row=2, column=0, sticky="ew")

        self.start_button = ttk.Button(
            buttons, text="Start Camera", command=self.start_camera
        )
        self.start_button.pack(side="left", padx=4)

        self.stop_button = ttk.Button(
            buttons, text="Stop Camera",
            command=self.stop_camera, state="disabled"
        )
        self.stop_button.pack(side="left", padx=4)

        self.photo_button = ttk.Button(
            buttons, text="Save Photo",
            command=self.save_photo, state="disabled"
        )
        self.photo_button.pack(side="left", padx=4)

        ttk.Button(
            buttons, text="Save CSV Report",
            command=self.save_report
        ).pack(side="left", padx=4)

    def start_camera(self):
        if self.running:
            return

        if not os.path.isfile(MODEL_PATH):
            messagebox.showerror(
                "Model missing",
                f"{MODEL_PATH} ko main.py ke same folder mein rakho."
            )
            return

        self.start_button.config(state="disabled")
        self.status.set("Status: Loading model...")

        threading.Thread(
            target=self.open_camera, daemon=True
        ).start()

    def open_camera(self):
        try:
            if self.model is None:
                self.model = YOLO(MODEL_PATH, task="detect")

            camera = cv2.VideoCapture(0)

            if not camera.isOpened():
                camera.release()
                raise RuntimeError(
                    "Camera open nahi hua. Camera permission check karo."
                )

            with self.lock:
                self.camera = camera

            self.stop_event.clear()
            self.running = True
            self.root.after(0, self.camera_started)

            threading.Thread(
                target=self.detection_loop, daemon=True
            ).start()

        except Exception as error:
            self.last_error = str(error)
            self.root.after(0, self.start_failed)

    def camera_started(self):
        self.status.set("Status: Running")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.photo_button.config(state="normal")

    def start_failed(self):
        self.running = False
        self.start_button.config(state="normal")
        self.status.set("Status: Error")
        messagebox.showerror("Start Error", self.last_error)

    def detection_loop(self):
        previous_time = time.perf_counter()

        while not self.stop_event.is_set():
            with self.lock:
                camera = self.camera

            if camera is None:
                break

            ok, frame = camera.read()
            if not ok:
                self.last_error = "Camera frame read nahi ho raha."
                break

            self.frame_number += 1
            height, width = frame.shape[:2]
            line_y = height // 2

            cv2.line(
                frame, (0, line_y), (width, line_y),
                (255, 0, 255), 2
            )
            cv2.putText(
                frame, "COUNTING LINE", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
            )

            # YOLO performs NMS to suppress many overlapping boxes.
            results = self.model.predict(
                frame,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                imgsz=640,
                max_det=20,
                verbose=False
            )

            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    name = str(
                        self.model.names[int(box.cls[0])]
                    )
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    colour = ""

                    # Do not estimate clothes/skin colour as a
                    # reliable person's colour.
                    if name.lower() != "person":
                        crop = frame[
                            y1 + (y2-y1)//4:y2 - (y2-y1)//4,
                            x1 + (x2-x1)//4:x2 - (x2-x1)//4
                        ]
                        colour = get_color(crop)

                    detections.append({
                        "name": name,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                        "center": ((x1+x2)//2, (y1+y2)//2),
                        "colour": colour
                    })

            # Match detections to nearby previous tracks.
            used_ids = set()
            counts = Counter()

            for det in detections:
                cx, cy = det["center"]
                best_id = None
                best_distance = MAX_DISTANCE

                for oid, track in self.tracks.items():
                    if oid in used_ids:
                        continue
                    if track["name"] != det["name"]:
                        continue
                    if (
                        self.frame_number - track["last_seen"]
                        > MAX_MISSING_FRAMES
                    ):
                        continue

                    ox, oy = track["center"]
                    distance = float(np.hypot(cx-ox, cy-oy))

                    if distance < best_distance:
                        best_distance = distance
                        best_id = oid

                if best_id is None:
                    self.next_id += 1
                    best_id = self.next_id

                    self.tracks[best_id] = {
                        "name": det["name"],
                        "center": (cx, cy),
                        "last_seen": self.frame_number,
                        "trail": [],
                        "previous_y": cy
                    }

                    self.records.append({
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "event": "DETECTED",
                        "tracking_id": best_id,
                        "object": det["name"],
                        "confidence_percent": round(
                            det["confidence"] * 100, 2
                        )
                    })

                track = self.tracks[best_id]
                old_y = track["previous_y"]
                track["center"] = (cx, cy)
                track["last_seen"] = self.frame_number
                track["previous_y"] = cy

                track["trail"].append((cx, cy))
                track["trail"] = track["trail"][-TRAIL_LENGTH:]
                used_ids.add(best_id)
                counts[det["name"]] += 1

                if best_id not in self.crossed_ids:
                    if old_y < line_y <= cy:
                        self.down_count += 1
                        self.crossed_ids.add(best_id)
                        event = "CROSSED_DOWN"
                    elif old_y > line_y >= cy:
                        self.up_count += 1
                        self.crossed_ids.add(best_id)
                        event = "CROSSED_UP"
                    else:
                        event = None

                    if event:
                        self.records.append({
                            "time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "event": event,
                            "tracking_id": best_id,
                            "object": det["name"],
                            "confidence_percent": ""
                        })

                x1, y1, x2, y2 = det["box"]

                # One readable label per accepted detection.
                label = f"{det['name']} ID:{best_id}"
                if det["colour"]:
                    label += f" {det['colour']}"
                label += f" {det['confidence']*100:.0f}%"

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 200, 0), 2
                )

                text_y = max(y1 - 8, 20)
                cv2.putText(
                    frame, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 150, 0), 2
                )

                trail = track["trail"]
                for i in range(1, len(trail)):
                    cv2.line(
                        frame, trail[i-1], trail[i],
                        (0, 255, 255), 2
                    )

            # Remove stale tracks.
            expired = [
                oid for oid, track in self.tracks.items()
                if self.frame_number - track["last_seen"]
                > MAX_MISSING_FRAMES
            ]
            for oid in expired:
                del self.tracks[oid]

            now = time.perf_counter()
            elapsed = now - previous_time
            fps = 1 / elapsed if elapsed > 0 else 0
            previous_time = now

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2
            )

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_counts = counts.copy()
                self.latest_fps = fps

        with self.lock:
            camera = self.camera
            self.camera = None

        if camera is not None:
            camera.release()

        self.running = False
        self.root.after(0, self.camera_stopped)

    def camera_stopped(self):
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.photo_button.config(state="disabled")

        if self.last_error:
            error = self.last_error
            self.last_error = None
            self.status.set("Status: Error")
            messagebox.showerror("Camera Error", error)
        else:
            self.status.set("Status: Stopped")

    def update_gui(self):
        with self.lock:
            frame = (
                self.latest_frame.copy()
                if self.latest_frame is not None else None
            )
            counts = self.latest_counts.copy()
            fps = self.latest_fps

        if frame is not None:
            h, w = frame.shape[:2]
            scale = min(700 / w, 480 / h)

            display = cv2.resize(
                frame, (int(w*scale), int(h*scale))
            )
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            photo = ImageTk.PhotoImage(Image.fromarray(display))
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo

            self.fps_text.set(f"FPS: {fps:.1f}")
            self.total_text.set(
                f"Objects in view: {sum(counts.values())}"
            )
            self.down_text.set(f"Crossed down: {self.down_count}")
            self.up_text.set(f"Crossed up: {self.up_count}")

            self.count_box.config(state="normal")
            self.count_box.delete("1.0", "end")
            lines = [
                f"{name}: {number}"
                for name, number in sorted(counts.items())
            ]
            self.count_box.insert(
                "1.0", "\n".join(lines) if lines
                else "No objects detected"
            )
            self.count_box.config(state="disabled")

        self.root.after(40, self.update_gui)

    def stop_camera(self):
        self.stop_event.set()
        self.status.set("Status: Stopping...")

    def save_photo(self):
        with self.lock:
            frame = (
                self.latest_frame.copy()
                if self.latest_frame is not None else None
            )

        if frame is None:
            messagebox.showinfo("Info", "Pehle camera start karo.")
            return

        filename = "tracking_" + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        ) + ".jpg"

        if cv2.imwrite(filename, frame):
            messagebox.showinfo(
                "Photo Saved", os.path.abspath(filename)
            )

    def save_report(self):
        if not self.records:
            messagebox.showinfo(
                "Info", "Pehle camera se objects detect karo."
            )
            return

        fields = [
            "time", "event", "tracking_id",
            "object", "confidence_percent"
        ]

        try:
            with open(
                CSV_FILE, "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.records)

            messagebox.showinfo(
                "CSV Saved", os.path.abspath(CSV_FILE)
            )
        except OSError as error:
            messagebox.showerror("CSV Error", str(error))

    def close_app(self):
        self.stop_event.set()

        with self.lock:
            camera = self.camera
            self.camera = None

        if camera is not None:
            camera.release()

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()
