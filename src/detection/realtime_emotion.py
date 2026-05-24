import torch
import cv2
import numpy as np
from torchvision import transforms
import sys
import os
from facenet_pytorch import MTCNN 

# =========================
# Path
# =========================
sys.path.append(r"C:\Users\anind\OneDrive\Desktop\DL_Proj_Face_Emotion\src\models")

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 🔥 INIT MTCNN
# =========================
mtcnn = MTCNN(
    keep_all=True,
    device=device
)

# =========================
# Load checkpoint
# =========================
checkpoint_path = r"C:\Users\anind\OneDrive\Desktop\DL_Proj_Face_Emotion\src\training\models\final_best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

if isinstance(checkpoint, dict) and "model_name" in checkpoint:
    model_name = checkpoint["model_name"]
    state_dict = checkpoint["model_state"]
else:
    model_name = "vgg"
    state_dict = checkpoint

# =========================
# Model selection
# =========================
if model_name == "cnn":
    from all_models import Simple_CNN
    model = Simple_CNN()

elif model_name == "resnet":
    from all_models import ResNet_Custom
    model = ResNet_Custom()

elif model_name == "resnet_attention":
    from all_models import ResNet_Attention
    model = ResNet_Attention()

elif model_name == "mobilenet":
    from all_models import MobileNetCustom
    model = MobileNetCustom()

elif model_name == "cnn_attention":
    from all_models import CNN_Attention
    model = CNN_Attention()

elif model_name == "fusion":
    from all_models import CNN_Fusion
    model = CNN_Fusion()

elif model_name == "vgg":
    from all_models import VGG16_Custom
    model = VGG16_Custom()

else:
    raise ValueError("Unknown model type")

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print(f"Loaded Model: {model_name}")

# =========================
# Classes
# =========================
class_names = [
    "Anger", "Content", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "SillyFace", "Surprise"
]

emotion_colors = {
    "Anger": (0, 0, 255),
    "Content": (255, 255, 0),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Neutral": (0, 255, 0),
    "Sad": (255, 0, 0),
    "SillyFace": (255, 0, 255),
    "Surprise": (0, 165, 255)
}

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =========================
# Webcam + Video Save
# =========================
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_path = r"C:\Users\anind\OneDrive\Desktop\DL_Proj_Face_Emotion\src\detection\emotion_output.mp4"

out = cv2.VideoWriter(
    save_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    20.0,
    (frame_width, frame_height)
)

print("Press 'q' to exit")

# =========================
# Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 🔥 Convert for MTCNN (IMPORTANT)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Emotion model input
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # =========================
    # 🔥 MTCNN DETECTION
    # =========================
    boxes, probs = mtcnn.detect(rgb_frame)

    if boxes is not None:

        for box, prob in zip(boxes, probs):

            if prob < 0.90:   # 🔥 filter false detections
                continue

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face = gray[y1:y2, x1:x2]

            if face.size == 0:
                continue

            try:
                face = cv2.resize(face, (128,128))
            except:
                continue

            face_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                probs_out = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs_out, 1)

            label = class_names[pred.item()]
            confidence = conf.item() * 100
            color = emotion_colors[label]

            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # Text
            text = f"{label}: {confidence:.1f}%"
            cv2.putText(
                frame,
                text,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Emotion Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()