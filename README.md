import os
import cv2
import csv
from ultralytics import YOLO

# -------------- CONFIG --------------
WEIGHTS = r"D:\13_august_2025\vishal\best (2).pt"
INPUT_DIR = r"D:\DDU\new_train\hopper_wagon_40-60kmph\202510160836"
OUTPUT_DIR = r"D:\DDU\new_train\hopper_crops\bmbs_rod_crop"
CONF_THRESHOLD = 0.3
BMB_CLASS_NAME = "bmbs_rod"
AXLE_CLASS_NAME = "axle"
BMB_MARGIN = 8
MARGIN_HORIZ = 10
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMG_MAX_SIZE = 1280
VISUALIZE = False
VIS_DIR = os.path.join(OUTPUT_DIR, "visuals")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary.csv")
# -------------- END CONFIG --------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
if VISUALIZE:
    os.makedirs(VIS_DIR, exist_ok=True)

model = YOLO(WEIGHTS)
names = getattr(model, "names", None) or {}

def is_image_file(fn):
    return os.path.splitext(fn.lower())[1] in IMG_EXTS

def clamp(v, a, b):
    return max(a, min(b, v))

def bbox_top(b): return b["xyxy"][1]
def bbox_bottom(b): return b["xyxy"][3]
def bbox_left(b): return b["xyxy"][0]
def bbox_right(b): return b["xyxy"][2]

rows = []
file_list = sorted([f for f in os.listdir(INPUT_DIR) if is_image_file(f)])
for idx, fname in enumerate(file_list, 1):
    in_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(in_path)
    if img is None:
        print(f"[{idx}/{len(file_list)}] Skipping unreadable image: {fname}")
        continue
    h_img, w_img = img.shape[:2]

    res = model(in_path, imgsz=IMG_MAX_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
    dets = []
    if hasattr(res, "boxes"):
        for box in res.boxes:
            xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box.xyxy, "cpu") else list(box.xyxy)
            conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(box.conf)
            cls_idx = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
            cls_name = names.get(cls_idx, str(cls_idx))
            if conf < CONF_THRESHOLD:
                continue
            dets.append({"xyxy": xyxy, "conf": conf, "cls": cls_name})
    else:
        print(f"[{idx}/{len(file_list)}] No detections for {fname}")
        continue

    bmbs = [d for d in dets if d["cls"] == BMB_CLASS_NAME]
    axles = [d for d in dets if d["cls"] == AXLE_CLASS_NAME]

    saved_for_image = 0
    if VISUALIZE:
        vis = img.copy()

    for i, bmb in enumerate(bmbs):
        cy = (bbox_top(bmb) + bbox_bottom(bmb)) / 2.0
        axles_above = [a for a in axles if (bbox_top(a) + bbox_bottom(a)) / 2.0 < cy]
        axles_below = [a for a in axles if (bbox_top(a) + bbox_bottom(a)) / 2.0 > cy]
        if not axles_above or not axles_below:
            continue
        axle_above = max(axles_above, key=lambda a: (bbox_top(a) + bbox_bottom(a)) / 2.0)
        axle_below = min(axles_below, key=lambda a: (bbox_top(a) + bbox_bottom(a)) / 2.0)

        x1 = int(min(bbox_left(bmb), bbox_left(axle_above), bbox_left(axle_below))) - MARGIN_HORIZ
        x2 = int(max(bbox_right(bmb), bbox_right(axle_above), bbox_right(axle_below))) + MARGIN_HORIZ

        axle_above_mid = int((bbox_top(axle_above) + bbox_bottom(axle_above)) / 2.0)
        axle_below_mid = int((bbox_top(axle_below) + bbox_bottom(axle_below)) / 2.0)

        bmb_top = int(bbox_top(bmb)) - BMB_MARGIN
        bmb_bottom = int(bbox_bottom(bmb)) + BMB_MARGIN

        y1 = min(bmb_top, axle_above_mid)
        y2 = max(bmb_bottom, axle_below_mid)

        x1 = clamp(x1, 0, w_img - 1)
        x2 = clamp(x2, 0, w_img - 1)
        y1 = clamp(y1, 0, h_img - 1)
        y2 = clamp(y2, 0, h_img - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2].copy()
        base_name = os.path.splitext(fname)[0]
        out_name = f"{base_name}_bmbs_rod_focus_{i+1}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, crop)
        saved_for_image += 1

        rows.append({
            "input_image": fname,
            "crop_file": out_name,
            "bmbs_rod_conf": round(bmb["conf"], 3),
            "axle_above_conf": round(axle_above["conf"], 3),
            "axle_below_conf": round(axle_below["conf"], 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

        if VISUALIZE:
            def draw_box(b, color, label=None):
                x1b, y1b, x2b, y2b = map(int, b["xyxy"])
                cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), color, 2)
                if label:
                    cv2.putText(vis, label, (x1b, max(12, y1b-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            draw_box(axle_above, (0,255,0), f"axle {axle_above['conf']:.2f}")
            draw_box(axle_below, (0,255,0), f"axle {axle_below['conf']:.2f}")
            draw_box(bmb, (255,0,0), f"bmbs_rod {bmb['conf']:.2f}")
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,0,255), 2)

    if saved_for_image == 0:
        print(f"[{idx}/{len(file_list)}] No valid bmbs_rod crops in {fname}")
    else:
        print(f"[{idx}/{len(file_list)}] Saved {saved_for_image} bmbs_rod crops for {fname}")

    if VISUALIZE:
        vis_out = os.path.join(VIS_DIR, f"{base_name}_vis.jpg")
        cv2.imwrite(vis_out, vis)

# write CSV summary
if rows:
    with open(SUMMARY_CSV, "w", newline="") as f:
        fieldnames = ["input_image","crop_file","bmbs_rod_conf","axle_above_conf","axle_below_conf","x1","y1","x2","y2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Summary CSV saved to {SUMMARY_CSV}")
else:
    print("No crops produced for any image.")
