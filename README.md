# CVAT QC Metrics Analyzer

A Python/Flask web application that accepts CVAT annotation exports and calculates **Precision**, **Recall**, **Accuracy (F1)**, and **Mean IoU** using proper IoU-based matching.

## Supported Export Formats

| Format | Extension | How to export from CVAT |
|--------|-----------|------------------------|
| CVAT for Images XML 1.1 | `.xml` | Task → Export → "CVAT for images 1.1" |
| COCO JSON | `.json` | Task → Export → "COCO 1.0" |
| Pascal VOC XML | `.xml` | Task → Export → "Pascal VOC 1.1" |

## How It Works

1. **Upload** your Ground Truth file (from a QC/reviewer job) and Prediction file (from an annotator job)
2. **Set the IoU Threshold** — minimum overlap required to count a detection as a True Positive
3. **Toggle label matching** — whether class names must match for a TP
4. **Run Analysis** — get instant metrics

## Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | How many annotator boxes were correct? |
| **Recall** | TP / (TP + FN) | How many GT objects did the annotator find? |
| **Accuracy (F1)** | 2 × P × R / (P + R) | Harmonic mean balancing both |
| **Mean IoU** | avg(IoU of matched pairs) | Average overlap quality of matches |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open in browser
http://localhost:5050
```

## IoU Threshold Guidelines

| Threshold | Standard | Use Case |
|-----------|----------|----------|
| 50% | Pascal VOC | General bounding box tasks |
| 75% | COCO strict | High-precision tasks |
| 40% | Lenient | Early-stage annotation review |

## Project Structure

```
cvat_qc/
├── app.py              # Flask web server
├── metrics.py          # Parser + IoU matching + metrics engine
├── requirements.txt
├── templates/
│   └── index.html      # Dashboard UI
├── sample_ground_truth.xml
└── sample_predictions.xml
```
