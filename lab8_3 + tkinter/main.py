import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class AgeGenderDetector:
    def __init__(self, face_model_path, face_proto_path,
                 gender_model_path, gender_proto_path,
                 age_model_path, age_proto_path):
        model_files = {
            'Face model': face_model_path,
            'Face proto': face_proto_path,
            'Gender model': gender_model_path,
            'Gender proto': gender_proto_path,
            'Age model': age_model_path,
            'Age proto': age_proto_path
        }
        missing_files = []
        for name, path in model_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        if missing_files:
            raise FileNotFoundError(f"Файлы моделей не найдены:\n" + "\n".join(missing_files))

        self.faceNet = cv2.dnn.readNet(face_model_path, face_proto_path)
        self.genderNet = cv2.dnn.readNet(gender_model_path, gender_proto_path)
        self.ageNet = cv2.dnn.readNet(age_model_path, age_proto_path)

    def detect_faces(self, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
        return frameOpencvDnn, faceBoxes

    def predict_age_gender(self, face_crop):
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        self.genderNet.setInput(blob)
        genderPreds = self.genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        gender_confidence = genderPreds[0].max()

        self.ageNet.setInput(blob)
        agePreds = self.ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        age_confidence = agePreds[0].max()
        return gender, age, gender_confidence, age_confidence

    def analyze_faces(self, frame, conf_threshold=0.7, draw_boxes=True):
        result_img = frame.copy()
        _, faceBoxes = self.detect_faces(frame, conf_threshold)
        face_data = []

        for i, faceBox in enumerate(faceBoxes):
            x1, y1, x2, y2 = faceBox
            face_crop = frame[max(0, y1):min(y2, frame.shape[0] - 1),
                         max(0, x1):min(x2, frame.shape[1] - 1)]
            if face_crop.size == 0:
                continue

            gender, age, gender_conf, age_conf = self.predict_age_gender(face_crop)
            face_info = {'box': faceBox, 'gender': gender, 'age': age,
                         'gender_confidence': gender_conf, 'age_confidence': age_conf}
            face_data.append(face_info)

            if draw_boxes:
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{gender}, {age}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
                cv2.rectangle(result_img, (text_x, text_y - label_size[1] - 3),
                            (text_x + label_size[0], text_y + 3), (0, 0, 0), -1)
                cv2.putText(result_img, label, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return result_img, face_data


class App:
    def __init__(self, root, detector):
        self.root = root
        self.detector = detector
        self.root.title("Детектор возраста и пола")
        self.root.geometry("800x600")

        self.panel = tk.Label(root)
        self.panel.pack(padx=10, pady=10)

        self.info_text = tk.Text(root, height=10)
        self.info_text.pack(padx=10, pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_image = tk.Button(btn_frame, text="Выбрать изображение", command=self.select_image)
        self.btn_image.pack(side=tk.LEFT, padx=5)

        self.btn_camera = tk.Button(btn_frame, text="Использовать камеру", command=self.start_camera)
        self.btn_camera.pack(side=tk.LEFT, padx=5)

        self.cap = None
        self.is_running = False

    def select_image(self):
        self.stop_camera()
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            try:
                image = cv2.imread(path)
                result_img, face_data = self.detector.analyze_faces(image)
                self.show_image(result_img)
                self.display_face_info(face_data)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось обработать изображение:\n{e}")

    def start_camera(self):
        self.stop_camera()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Камера не найдена.")
            return
        self.is_running = True
        self.update_frame()

    def stop_camera(self):
        if self.is_running and self.cap:
            self.cap.release()
            self.is_running = False

    def update_frame(self):
        if not self.is_running:
            return
        ret, frame = self.cap.read()
        if ret:
            result_img, face_data = self.detector.analyze_faces(frame)
            self.show_image(result_img)
            self.display_face_info(face_data)
        self.root.after(10, self.update_frame)

    def show_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (700, 500))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

    def display_face_info(self, face_data):
        self.info_text.delete('1.0', tk.END)
        if face_data:
            for i, info in enumerate(face_data, 1):
                self.info_text.insert(tk.END, f"Лицо {i}:\n")
                self.info_text.insert(tk.END, f" Пол: {info['gender']} ({info['gender_confidence']:.2f})\n")
                self.info_text.insert(tk.END, f" Возраст: {info['age']} ({info['age_confidence']:.2f})\n\n")
        else:
            self.info_text.insert(tk.END, "Лица не распознаны.")


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"


if __name__ == "__main__":
    try:
        detector = AgeGenderDetector(faceModel, faceProto, genderModel, genderProto, ageModel, ageProto)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        exit(1)

    root = tk.Tk()
    app = App(root, detector)
    root.mainloop()