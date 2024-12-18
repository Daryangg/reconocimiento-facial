import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Inicializar Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Cargar modelos de edad y género
age_net = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
gender_net = cv2.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')

# Cargar el de deteccíon de emociones modelo sin compilar
emotion_model = load_model(
    'fer2013_mini_XCEPTION.110-0.65.hdf5', compile=False)

# Compilar manualmente el modelo con un optimizador actualizado
emotion_model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['accuracy'])


# Etiquetas de edad, género y emociones
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Male', 'Female']
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Sad', 'Surprise', 'Neutral']

# Capturar video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB para Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección de rostros con Mediapipe
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Obtener coordenadas del rostro
            bboxC = detection.location_data.relative_bounding_box
            xmin = int(bboxC.xmin * frame.shape[1])
            ymin = int(bboxC.ymin * frame.shape[0])
            width = int(bboxC.width * frame.shape[1])
            height = int(bboxC.height * frame.shape[0])

            # Ajustar coordenadas y agregar margen
            margin = 20
            xmin = max(0, xmin - margin)
            ymin = max(0, ymin - margin)
            xmax = min(frame.shape[1], xmin + width + margin)
            ymax = min(frame.shape[0], ymin + height + margin)

            # Recortar el rostro
            face = frame[ymin:ymax, xmin:xmax]

            if face.size == 0:
                continue  # Evita errores si el recorte está vacío

            # Preprocesar el rostro para edad y género
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Predicción de género
            gender_net.setInput(blob)
            gender = GENDER_LABELS[gender_net.forward()[0].argmax()]

            # Predicción de edad
            age_net.setInput(blob)
            age = AGE_LABELS[age_net.forward()[0].argmax()]

            # Preprocesar el rostro para emoción
            # Convertir a escala de grises
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Redimensionar a 64x64
            # Redimensionar a 64x64
            face_resized = cv2.resize(face_gray, (64, 64))
            face_resized = face_resized.reshape(
                1, 64, 64, 1) / 255.0  # Añadir canal y normalizar

            # Predicción de emoción
            emotion_preds = emotion_model.predict(face_resized)
            emotion = EMOTION_LABELS[np.argmax(emotion_preds)]

            # Dibujar resultados en el frame
            label = f'{gender}, {age}, {emotion}'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    # Mostrar el video
    cv2.imshow('Real-Time Analysis: Age, Gender, Emotion', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
