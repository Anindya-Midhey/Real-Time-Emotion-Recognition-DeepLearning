import torch
import cv2
import numpy as np
from torchvision import transforms
import os
import sys
from facenet_pytorch import MTCNN

# =========================
# Path (MODEL IMPORT)
# =========================
sys.path.append(r"C:\Users\anind\OneDrive\Desktop\DL_Proj_Face_Emotion\src\models")

# =========================
# Device (FORCE GPU)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 🔥 INIT MTCNN (GPU)
# =========================
mtcnn = MTCNN(
    keep_all=True,
    device=device
)

# =========================
# Load Model
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
# Model Selection
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
# INPUT & OUTPUT FOLDERS
# =========================
input_folder = r"C:\Users\anind\OneDrive\Desktop\DL_Proj_Face_Emotion\src\input_videos"
output_folder = r"C:\Users\anind\OneDrive\Desktop\DL_Proj_Face_Emotion\src\output_videos"

os.makedirs(output_folder, exist_ok=True)

video_files = [f for f in os.listdir(input_folder) if f.endswith((".mp4", ".avi", ".mov"))]

print(f"Total videos found: {len(video_files)}")

# =========================
# Process each video
# =========================
for idx, video_name in enumerate(video_files, start=1):

    input_path = os.path.join(input_folder, video_name)
    output_path = os.path.join(output_folder, f"video_output{idx}.mp4")

    print(f"\nProcessing: {video_name}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Failed to open {video_name}")
        continue

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 Convert BGR → RGB (IMPORTANT for MTCNN)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8,8))
        gray = clahe.apply(gray)

        # =========================
        # 🔥 GPU FACE DETECTION
        # =========================
        boxes, probs = mtcnn.detect(rgb_frame)

        if boxes is not None:

            for box, prob in zip(boxes, probs):

                if prob < 0.95:
                    continue

                x1, y1, x2, y2 = map(int, box)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face = gray[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face = cv2.resize(face, (128,128))

                face_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)
                    probs_out = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs_out, 1)

                label = class_names[pred.item()]
                confidence_text = conf.item() * 100
                color = emotion_colors[label]

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                text = f"{label}: {confidence_text:.1f}%"
                cv2.putText(
                    frame,
                    text,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # =========================
        # SAVE OUTPUT
        # =========================
        out.write(frame)

        # Optional preview
        cv2.imshow("Processing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

    print(f"Saved: {output_path}")

cv2.destroyAllWindows()

print("\n✅ All videos processed successfully!")