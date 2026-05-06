import os
import time
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from barcode_scanner import BarcodeScanner

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
VISION_STREAM_URL  = "http://localhost:5001/stream"
BARCODE_STREAM_URL = "http://10.10.10.227:4747/video" 
YOLO_MODEL_PATH       = "yog_yolo26n.pt"
CLASSIFIER_MODEL_PATH = "best_model.pt"
FLAVOR_MODEL_PATH     = "./TrCustom"

PRESENCE_FRAMES      = 15
RESULT_HOLD_SECS     = 4.0
FLAVOR_CLASS_NAME    = "flavor"
FLAVOR_CONF          = 0.35
FLAVOR_SCALE         = 2.5
EXPIRY_CLASS_ID      = 1
YOLO_CONF_THRESHOLD  = 0.25

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TrOCRReader:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)

    def read(self, bgr_crop: np.ndarray) -> str:
        pil_img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        px = self.processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(px)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

class MultiHeadExpiryModel(nn.Module):
    def __init__(self, dropout: float = 0.20):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.defect_head = nn.Linear(feature_dim, 2)
        self.day_tens_head = nn.Linear(feature_dim, 4)
        self.day_units_head = nn.Linear(feature_dim, 10)
        self.month_head = nn.Linear(feature_dim, 12)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = torch.flatten(self.backbone(x), 1)
        feat = self.dropout(feat)
        return {"is_defect": self.defect_head(feat), "day_tens": self.day_tens_head(feat),
                "day_units": self.day_units_head(feat), "month": self.month_head(feat)}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InspectionResult:
    flavor_text: str = "WAITING"
    expiry_text: str = "WAITING"
    flavor_crop: Optional[np.ndarray] = None
    expiry_crop: Optional[np.ndarray] = None
    done: bool = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(YOLO_MODEL_PATH)
    ocr = TrOCRReader(FLAVOR_MODEL_PATH)
    
    ckpt = torch.load(CLASSIFIER_MODEL_PATH, map_location=device, weights_only=False)
    classifier = MultiHeadExpiryModel().to(device).eval()
    classifier.load_state_dict(ckpt["model_state_dict"])
    meta = {"image_size": tuple(ckpt.get("image_size", [128, 40])), 
            "idx_to_month": {int(k): v for k, v in ckpt.get("idx_to_month", {}).items()}}

    cap = cv2.VideoCapture(VISION_STREAM_URL)
    scanner = BarcodeScanner(BARCODE_STREAM_URL)
    state, presence_frames = "SCANNING", 0
    shared_res = InspectionResult()
    lock = threading.Lock()

    def worker(snap):
        y_res = yolo(snap, conf=0.3, verbose=False)[0]
        f_txt, e_txt = "NOT FOUND", "NOT FOUND"
        f_crop, e_crop = None, None

        for box in y_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            
            # Flavor Processing
            if yolo.names[cls] == FLAVOR_CLASS_NAME:
                f_crop = snap[max(0, y1-5):min(snap.shape[0], y2+5), max(0, x1-15):min(snap.shape[1], x2+15)]
                if f_crop.size > 0:
                    up = cv2.resize(f_crop, None, fx=FLAVOR_SCALE, fy=FLAVOR_SCALE, interpolation=cv2.INTER_LANCZOS4)
                    f_txt = ocr.read(up).strip()

            # Expiry Processing
            elif cls == EXPIRY_CLASS_ID:
                e_crop = snap[max(0, y1-10):min(snap.shape[0], y2+10), max(0, x1-10):min(snap.shape[1], x2+10)]
                if e_crop.size > 0:
                    gray = cv2.resize(cv2.cvtColor(e_crop, cv2.COLOR_BGR2GRAY), meta["image_size"])
                    gray = cv2.createCLAHE(clipLimit=3.0).apply(gray).astype(np.float32) / 255.0
                    img_t = torch.from_numpy(np.stack([gray]*3, axis=0)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = classifier(img_t)
                        probs = {k: torch.softmax(v, dim=1) for k, v in out.items()}
                    if int(probs["is_defect"].argmax(1)) == 1: e_txt = "DEFECTIVE"
                    else:
                        d = int(probs["day_tens"].argmax(1)) * 10 + int(probs["day_units"].argmax(1))
                        m = meta["idx_to_month"].get(int(probs["month"].argmax(1)), "??")
                        e_txt = f"{d:02d} {m}"

        with lock:
            shared_res.flavor_text, shared_res.expiry_text = f_txt, e_txt
            shared_res.flavor_crop, shared_res.expiry_crop = f_crop, e_crop
            shared_res.done = True
        
        print(f"\nLOG [{datetime.now().strftime('%H:%M:%S')}] F: {f_txt} | E: {e_txt}")

    while True:
        for _ in range(5): cap.grab()
        ret, frame = cap.retrieve()
        if not ret: 
            print("Niryo Stream Disconnected. Retrying...")
            time.sleep(1)
            cap = cv2.VideoCapture(VISION_STREAM_URL)
            continue
        
        disp = frame.copy()
        y_res = yolo.predict(disp, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
        current_barcode = scanner.get_latest()

        for box in y_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c_name = yolo.names[int(box.cls)]
            col = (0, 255, 0) if c_name == "flavor" else (255, 255, 0)
            cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
            cv2.putText(disp, c_name, (x1, y1-5), 1, 1, col, 1)

        if state == "SCANNING":
            if len(y_res.boxes) > 0:
                presence_frames += 1
                if presence_frames >= PRESENCE_FRAMES:
                    state, shared_res.done = "PROCESSING", False
                    threading.Thread(target=worker, args=(frame.copy(),), daemon=True).start()
            else: presence_frames = 0
        elif state == "PROCESSING" and shared_res.done:
            state, timer = "RESULT", time.time()
        elif state == "RESULT":
            # --- DIAGNOSTIC OVERLAY ---
            if shared_res.flavor_crop is not None:
                tmp_f = cv2.resize(shared_res.flavor_crop, (200, 50))
                disp[10:60, 10:210] = tmp_f
                cv2.putText(disp, shared_res.flavor_text, (10, 80), 1, 1.5, (0, 255, 0), 2)
            if shared_res.expiry_crop is not None:
                tmp_e = cv2.resize(shared_res.expiry_crop, (200, 50))
                disp[100:150, 10:210] = tmp_e
                cv2.putText(disp, shared_res.expiry_text, (10, 170), 1, 1.5, (255, 255, 0), 2)
            
            cv2.rectangle(disp, (0, disp.shape[0]-40), (disp.shape[1], disp.shape[0]), (50, 50, 50), -1)
            cv2.putText(disp, f"BARCODE: {current_barcode}", (20, disp.shape[0]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if time.time() - timer > RESULT_HOLD_SECS: state, presence_frames = "SCANNING", 0

        cv2.imshow("DroidCam AI Diagnostic", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    scanner.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()