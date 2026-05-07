"""
CVAT QC Metrics Calculator
Supports: CVAT XML (1.1), COCO JSON, Pascal VOC XML

Smart frame matching: only scores frames that exist in BOTH the ground
truth and the annotator's export. This handles Random-per-job GT setups
where each annotator only sees a subset of the full GT frame pool.
"""
import xml.etree.ElementTree as ET
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class BBox:
    label: str
    x: float
    y: float
    w: float
    h: float
    attributes: Dict = field(default_factory=dict)

    def area(self) -> float:
        return self.w * self.h

    def as_xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class FrameAnnotations:
    frame_id: str
    boxes: List[BBox] = field(default_factory=list)


def compute_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    union = a.area() + b.area() - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ── Parsers ──────────────────────────────────────────────────────────────────


def _extract_cvat_annotator(root) -> str:
    """
    Pull the annotator name/email from CVAT XML metadata.
    Priority: assignee username -> assignee email -> owner username -> owner email
    """
    meta = root.find("meta")
    if meta is None:
        return ""
    for tag in ("job", "task"):
        container = meta.find(tag)
        if container is None:
            continue
        for person_tag in ("assignee", "owner"):
            person = container.find(person_tag)
            if person is None:
                continue
            for field in ("username", "email"):
                val = (person.findtext(field) or "").strip()
                if val and val.lower() not in ("", "null", "none"):
                    return val
    return ""

def parse_cvat_xml(filepath: str) -> Dict[str, FrameAnnotations]:
    tree = ET.parse(filepath)
    root = tree.getroot()
    frames: Dict[str, FrameAnnotations] = {}

    for image in root.findall("image"):
        fid = image.get("id", image.get("name", "unknown"))
        fa = FrameAnnotations(frame_id=fid)
        for box in image.findall("box"):
            label = box.get("label", "")
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
            attrs = {a.get("name"): a.text for a in box.findall("attribute")}
            fa.boxes.append(BBox(label=label, x=xtl, y=ytl,
                                 w=xbr - xtl, h=ybr - ytl, attributes=attrs))
        frames[fid] = fa

    for track in root.findall("track"):
        label = track.get("label", "")
        for box in track.findall("box"):
            frame = box.get("frame", "0")
            if frame not in frames:
                frames[frame] = FrameAnnotations(frame_id=frame)
            if box.get("outside", "0") == "1":
                continue
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
            attrs = {a.get("name"): a.text for a in box.findall("attribute")}
            frames[frame].boxes.append(BBox(label=label, x=xtl, y=ytl,
                                            w=xbr - xtl, h=ybr - ytl, attributes=attrs))
    annotator_name = _extract_cvat_annotator(root)
    return frames, annotator_name


def parse_coco_json(filepath: str) -> Dict[str, FrameAnnotations]:
    with open(filepath) as f:
        data = json.load(f)
    id_to_label = {c["id"]: c["name"] for c in data.get("categories", [])}
    id_to_image = {img["id"]: str(img["id"]) for img in data.get("images", [])}
    frames: Dict[str, FrameAnnotations] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        fid = id_to_image.get(img_id, str(img_id))
        if fid not in frames:
            frames[fid] = FrameAnnotations(frame_id=fid)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        label = id_to_label.get(ann.get("category_id", 0), "unknown")
        frames[fid].boxes.append(BBox(label=label, x=bbox[0], y=bbox[1],
                                      w=bbox[2], h=bbox[3]))
    return frames, ""


def parse_voc_xml(filepath: str) -> Dict[str, FrameAnnotations]:
    tree = ET.parse(filepath)
    root = tree.getroot()
    filename = root.findtext("filename", default=os.path.basename(filepath))
    fa = FrameAnnotations(frame_id=filename)
    for obj in root.findall("object"):
        label = obj.findtext("name", default="unknown")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", 0))
        ymin = float(bndbox.findtext("ymin", 0))
        xmax = float(bndbox.findtext("xmax", 0))
        ymax = float(bndbox.findtext("ymax", 0))
        fa.boxes.append(BBox(label=label, x=xmin, y=ymin,
                             w=xmax - xmin, h=ymax - ymin))
    return {filename: fa}, ""


def detect_and_parse(filepath: str) -> Tuple[Dict[str, FrameAnnotations], str, str]:
    """Returns (frames, format_string, annotator_name)"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        frames, name = parse_coco_json(filepath)
        return frames, "COCO JSON", name
    if ext == ".xml":
        tree = ET.parse(filepath)
        root = tree.getroot()
        tag = root.tag.lower()
        if tag == "annotations" and root.find("image") is not None:
            frames, name = parse_cvat_xml(filepath)
            return frames, "CVAT XML", name
        if tag == "annotations" and root.find("track") is not None:
            frames, name = parse_cvat_xml(filepath)
            return frames, "CVAT XML (Track)", name
        if tag == "annotation" and root.find("object") is not None:
            frames, name = parse_voc_xml(filepath)
            return frames, "Pascal VOC XML", name
        frames, name = parse_cvat_xml(filepath)
        return frames, "CVAT XML", name
    raise ValueError(f"Unsupported file format: {ext}")


# ── Metrics Computation ───────────────────────────────────────────────────────

@dataclass
class FrameResult:
    frame_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_scores: List[float] = field(default_factory=list)


@dataclass
class MetricsReport:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    accuracy: float
    mean_iou: float
    iou_threshold: float
    frame_results: List[FrameResult]
    label_breakdown: Dict[str, Dict]
    total_gt: int
    total_pred: int
    format_gt: str
    format_pred: str
    matched_frames: int        # frames present in both GT and pred
    gt_only_frames: int        # GT frames the annotator never saw (excluded)
    pred_only_frames: int      # frames annotator did that aren't in GT


def compute_metrics(
    gt_frames: Dict[str, FrameAnnotations],
    pred_frames: Dict[str, FrameAnnotations],
    iou_threshold: float = 0.50,
    compare_labels: bool = True,
) -> MetricsReport:
    """
    Smart intersection-based scoring.

    Only frames that appear in BOTH the GT and the annotator's export
    are scored. GT-only frames are excluded — this handles Random-per-job
    GT setups where each annotator only saw a subset of the GT pool.
    Pred-only frames (annotator drew boxes on frames not in GT) still
    contribute False Positives since they shouldn't be there.
    """

    gt_frame_ids  = set(gt_frames.keys())
    pred_frame_ids = set(pred_frames.keys())

    # Frames in both — these are the ones we can fairly score
    shared_frames = gt_frame_ids & pred_frame_ids

    # GT frames the annotator never saw — excluded from scoring entirely
    gt_only_frames = gt_frame_ids - pred_frame_ids

    # Frames annotator submitted that have no GT counterpart — pure FP frames
    pred_only_frames = pred_frame_ids - gt_frame_ids

    all_iou_scores = []
    frame_results = []
    label_stats: Dict[str, Dict] = {}
    total_tp = total_fp = total_fn = 0

    # ── Score shared frames ───────────────────────────────────────────────
    for fid in sorted(shared_frames):
        gt  = gt_frames[fid]
        pred = pred_frames[fid]
        fr = FrameResult(frame_id=fid)
        matched_gt = set()
        matched_pred = set()

        iou_matrix = []
        for pi, pb in enumerate(pred.boxes):
            for gi, gb in enumerate(gt.boxes):
                if compare_labels and pb.label != gb.label:
                    iou = 0.0
                else:
                    iou = compute_iou(pb, gb)
                iou_matrix.append((iou, pi, gi))
        iou_matrix.sort(reverse=True)

        for iou, pi, gi in iou_matrix:
            if pi in matched_pred or gi in matched_gt:
                continue
            if iou >= iou_threshold:
                matched_pred.add(pi)
                matched_gt.add(gi)
                fr.tp += 1
                fr.iou_scores.append(iou)
                all_iou_scores.append(iou)
                lbl = pred.boxes[pi].label
                if lbl not in label_stats:
                    label_stats[lbl] = {"tp": 0, "fp": 0, "fn": 0}
                label_stats[lbl]["tp"] += 1

        fr.fp = len(pred.boxes) - len(matched_pred)
        fr.fn = len(gt.boxes) - len(matched_gt)

        for pi, pb in enumerate(pred.boxes):
            if pi not in matched_pred:
                lbl = pb.label
                if lbl not in label_stats:
                    label_stats[lbl] = {"tp": 0, "fp": 0, "fn": 0}
                label_stats[lbl]["fp"] += 1

        for gi, gb in enumerate(gt.boxes):
            if gi not in matched_gt:
                lbl = gb.label
                if lbl not in label_stats:
                    label_stats[lbl] = {"tp": 0, "fp": 0, "fn": 0}
                label_stats[lbl]["fn"] += 1

        total_tp += fr.tp
        total_fp += fr.fp
        total_fn += fr.fn
        frame_results.append(fr)

    # ── Pred-only frames contribute FPs (annotator shouldn't have boxes there) ──
    for fid in sorted(pred_only_frames):
        pred = pred_frames[fid]
        if not pred.boxes:
            continue
        fr = FrameResult(frame_id=fid, fp=len(pred.boxes))
        for pb in pred.boxes:
            lbl = pb.label
            if lbl not in label_stats:
                label_stats[lbl] = {"tp": 0, "fp": 0, "fn": 0}
            label_stats[lbl]["fp"] += 1
        total_fp += fr.fp
        frame_results.append(fr)

    # ── Final metrics ─────────────────────────────────────────────────────
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    mean_iou  = sum(all_iou_scores) / len(all_iou_scores) if all_iou_scores else 0.0

    if (precision + recall) > 0:
        accuracy = 2 * (precision * recall) / (precision + recall)
    else:
        accuracy = 0.0

    for lbl, s in label_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        s["precision"] = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0.0
        s["recall"]    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0

    # Count GT boxes only from shared frames (fair denominator)
    total_gt   = sum(len(gt_frames[f].boxes) for f in shared_frames)
    total_pred = sum(len(pred_frames[f].boxes) for f in (shared_frames | pred_only_frames))

    return MetricsReport(
        tp=total_tp, fp=total_fp, fn=total_fn,
        precision=round(precision * 100, 2),
        recall=round(recall * 100, 2),
        accuracy=round(accuracy * 100, 2),
        mean_iou=round(mean_iou * 100, 2),
        iou_threshold=iou_threshold * 100,
        frame_results=frame_results,
        label_breakdown=label_stats,
        total_gt=total_gt,
        total_pred=total_pred,
        format_gt="",
        format_pred="",
        matched_frames=len(shared_frames),
        gt_only_frames=len(gt_only_frames),
        pred_only_frames=len(pred_only_frames),
    )