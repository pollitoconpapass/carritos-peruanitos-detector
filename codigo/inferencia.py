import cv2
import torch
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_REPO_ID = "pollitoconpapass/carritos-detectorV2"

hf_model_url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/model.pt"
ckpt_hf = torch.hub.load_state_dict_from_url(hf_model_url, map_location=DEVICE)

model_hf = fasterrcnn_resnet50_fpn(weights=None)
in_features = model_hf.roi_heads.box_predictor.cls_score.in_features
model_hf.roi_heads.box_predictor = FastRCNNPredictor(in_features, ckpt_hf["num_classes"])
model_hf.load_state_dict(ckpt_hf["model_state_dict"])
model_hf.to(DEVICE)
model_hf.eval()

categories_hf = ckpt_hf["categories"]
print(f"[INFO] Modelo cargado desde HF (epoca {ckpt_hf['best_epoch']}, mAP@50={ckpt_hf['best_map50']:.4f})")
print(f"[INFO] Clases: {categories_hf}")

def infer_image(model, image_path, categories, score_threshold=0.5):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)[0]

    keep = pred["scores"] >= score_threshold
    boxes = pred["boxes"][keep].cpu().numpy()
    labels = pred["labels"][keep].cpu().numpy()
    scores = pred["scores"][keep].cpu().numpy()

    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"Predicciones (threshold={score_threshold})")
    ax.axis("off")

    class_colors = {1: "red", 2: "cyan", 3: "lime"}
    h_orig, w_orig = image_rgb.shape[:2]
    sx, sy = w_orig / 640, h_orig / 640

    for box, label, score in zip(boxes, labels, scores):
        color = class_colors.get(label, "white")
        x1, y1 = box[0] * sx, box[1] * sy
        w, h = (box[2] - box[0]) * sx, (box[3] - box[1]) * sy
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{categories.get(label, label)}: {score:.2f}",
                color=color, fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    plt.tight_layout()
    plt.show()

    print(f"Detecciones: {len(boxes)}")
    for label, score in zip(labels, scores):
        print(f"  {categories.get(label, label):12s} conf={score:.3f}")

# Cambiar path con imagen propia
image_path = "/content/frame_0875.png"
infer_image(model_hf, image_path, categories_hf, score_threshold=0.5)
