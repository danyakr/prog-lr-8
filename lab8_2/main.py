import cv2
import os
import sys
import numpy as np

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"


genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

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
            
            face_info = {
                'box': faceBox,
                'gender': gender,
                'age': age,
                'gender_confidence': gender_conf,
                'age_confidence': age_conf
            }
            face_data.append(face_info)
            
            if draw_boxes:
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 
                            int(round(frame.shape[0]/150)), 8)
                
                label = f'{gender}, {age}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
                
                cv2.rectangle(result_img, (text_x, text_y - label_size[1] - 3), 
                            (text_x + label_size[0], text_y + 3), (0, 0, 0), -1)
                
                cv2.putText(result_img, label, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        return result_img, face_data
    
    def process_image(self, image_path, output_path=None, conf_threshold=0.7):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удается загрузить изображение: {image_path}")
        
        result_img, face_data = self.analyze_faces(image, conf_threshold)
        
        if face_data:
            print(f"Найдено лиц: {len(face_data)}")
            for i, face_info in enumerate(face_data, 1):
                print(f"Лицо {i}:")
                print(f"  Координаты: {face_info['box']}")
                print(f"  Пол: {face_info['gender']} (уверенность: {face_info['gender_confidence']:.2f})")
                print(f"  Возраст: {face_info['age']} лет (уверенность: {face_info['age_confidence']:.2f})")
                print()
        else:
            print("Лица не распознаны")
        
        if output_path:
            cv2.imwrite(output_path, result_img)
            print(f"Результат сохранен: {output_path}")
        
        return result_img, face_data
    
    def process_video_stream(self, conf_threshold=0.7):
        video = cv2.VideoCapture(0)
        
        if not video.isOpened():
            raise RuntimeError("Не удается открыть камеру")
        
        print("Нажмите любую клавишу для выхода...")
        
        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()
            if not hasFrame:
                cv2.waitKey()
                break
            
            result_img, face_data = self.analyze_faces(frame, conf_threshold)
            
            if face_data:
                print(f"Кадр: найдено лиц - {len(face_data)}")
                for i, face_info in enumerate(face_data, 1):
                    print(f"  Лицо {i}: {face_info['gender']}, {face_info['age']}")
            else:
                print("Лица не распознаны")
            
            cv2.imshow("Detecting age and gender", result_img)
        
        video.release()
        cv2.destroyAllWindows()


def main():
    try:
        detector = AgeGenderDetector(
            faceModel, faceProto,
            genderModel, genderProto,
            ageModel, ageProto
        )
        
        print("=== Обработка изображения ===")
        image_filename = "image.jpg"
        
        if os.path.exists(image_filename):
            result_img, face_data = detector.process_image(
                image_filename, 
                output_path="result_age_gender_" + image_filename
            )
            
            cv2.imshow("Age and Gender Detection Result", result_img)
            print("Нажмите любую клавишу для продолжения...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Файл {image_filename} не найден. Пропускаем обработку изображения.")
        
        print("\n=== Работа с видеокамерой ===")
        user_input = input("Запустить видеокамеру? (y/n): ")
        if user_input.lower() == 'y':
            detector.process_video_stream()
    
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


def detect_age_gender_in_image(image_filename, output_filename=None, conf_threshold=0.7):
    detector = AgeGenderDetector(
        faceModel, faceProto,
        genderModel, genderProto,
        ageModel, ageProto
    )
    
    if output_filename is None:
        name, ext = os.path.splitext(image_filename)
        output_filename = f"{name}_age_gender{ext}"
    
    return detector.process_image(image_filename, output_filename, conf_threshold)


if __name__ == "__main__":
    main()