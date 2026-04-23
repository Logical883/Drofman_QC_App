"""
CVAT QC Metrics Flask Application
"""
import os
import uuid
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from metrics import detect_and_parse, compute_metrics

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB

ALLOWED = {".xml", ".json"}


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    gt_file = request.files.get("gt_file")
    pred_file = request.files.get("pred_file")

    if not gt_file or not pred_file:
        return jsonify({"error": "Both ground truth and prediction files are required."}), 400

    if not allowed_file(gt_file.filename) or not allowed_file(pred_file.filename):
        return jsonify({"error": "Only .xml and .json files are supported."}), 400

    iou_threshold = float(request.form.get("iou_threshold", 50)) / 100
    compare_labels = request.form.get("compare_labels", "true") == "true"

    uid = uuid.uuid4().hex
    gt_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_gt_{secure_filename(gt_file.filename)}")
    pred_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_pred_{secure_filename(pred_file.filename)}")

    try:
        gt_file.save(gt_path)
        pred_file.save(pred_path)

        gt_frames, gt_fmt = detect_and_parse(gt_path)
        pred_frames, pred_fmt = detect_and_parse(pred_path)

        report = compute_metrics(
            gt_frames, pred_frames,
            iou_threshold=iou_threshold,
            compare_labels=compare_labels,
        )
        report.format_gt = gt_fmt
        report.format_pred = pred_fmt

        frame_data = [
            {
                "frame_id": fr.frame_id,
                "tp": fr.tp,
                "fp": fr.fp,
                "fn": fr.fn,
                "mean_iou": round(sum(fr.iou_scores) / len(fr.iou_scores) * 100, 1) if fr.iou_scores else 0,
            }
            for fr in report.frame_results
        ]

        return jsonify({
            "precision": report.precision,
            "recall": report.recall,
            "accuracy": report.accuracy,
            "mean_iou": report.mean_iou,
            "tp": report.tp,
            "fp": report.fp,
            "fn": report.fn,
            "iou_threshold": report.iou_threshold,
            "total_gt": report.total_gt,
            "total_pred": report.total_pred,
            "format_gt": report.format_gt,
            "format_pred": report.format_pred,
            "frame_results": frame_data,
            "label_breakdown": report.label_breakdown,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for p in [gt_path, pred_path]:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5050)
