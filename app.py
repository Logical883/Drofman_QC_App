"""
CVAT QC Metrics Flask Application
"""
import os
import io
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from metrics import detect_and_parse, compute_metrics
from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

ALLOWED = {".xml", ".json"}


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED


def serialize_report(report, annotator_name):
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
    return {
        "annotator": annotator_name,
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
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Single annotator analysis (legacy endpoint)."""
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
        report = compute_metrics(gt_frames, pred_frames, iou_threshold=iou_threshold, compare_labels=compare_labels)
        report.format_gt = gt_fmt
        report.format_pred = pred_fmt
        return jsonify(serialize_report(report, "Annotator"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in [gt_path, pred_path]:
            if os.path.exists(p):
                os.remove(p)


@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    """Multi-annotator batch analysis."""
    gt_file = request.files.get("gt_file")
    if not gt_file:
        return jsonify({"error": "Ground truth file is required."}), 400

    iou_threshold = float(request.form.get("iou_threshold", 50)) / 100
    compare_labels = request.form.get("compare_labels", "true") == "true"

    pred_files = request.files.getlist("pred_files")
    annotator_names = request.form.getlist("annotator_names")

    if not pred_files:
        return jsonify({"error": "At least one annotator file is required."}), 400

    uid = uuid.uuid4().hex
    gt_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_gt_{secure_filename(gt_file.filename)}")

    try:
        gt_file.save(gt_path)
        gt_frames, gt_fmt = detect_and_parse(gt_path)

        results = []
        errors = []

        for i, pf in enumerate(pred_files):
            name = annotator_names[i] if i < len(annotator_names) and annotator_names[i].strip() else f"Annotator {i+1}"
            if not allowed_file(pf.filename):
                errors.append(f"{name}: unsupported file format")
                continue

            pred_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_pred{i}_{secure_filename(pf.filename)}")
            try:
                pf.save(pred_path)
                pred_frames, pred_fmt = detect_and_parse(pred_path)
                report = compute_metrics(gt_frames, pred_frames, iou_threshold=iou_threshold, compare_labels=compare_labels)
                report.format_gt = gt_fmt
                report.format_pred = pred_fmt
                results.append(serialize_report(report, name))
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
            finally:
                if os.path.exists(pred_path):
                    os.remove(pred_path)

        return jsonify({"results": results, "errors": errors})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(gt_path):
            os.remove(gt_path)


@app.route("/export-excel", methods=["POST"])
def export_excel():
    """Generate and return a formatted Excel report from batch results."""
    data = request.get_json()
    results = data.get("results", [])
    iou_threshold = data.get("iou_threshold", 50)
    session_name = data.get("session_name", "QC Report")

    if not results:
        return jsonify({"error": "No results to export."}), 400

    wb = Workbook()

    # ── Color palette ──────────────────────────────────────────────────────
    C_HEADER_BG   = "1A2332"
    C_HEADER_FG   = "FFFFFF"
    C_ACCENT      = "00C8A0"
    C_ACCENT_DARK = "008F72"
    C_SUBHEAD_BG  = "E8F8F4"
    C_GOOD_BG     = "D6F5ED"
    C_GOOD_FG     = "0A6B4A"
    C_WARN_BG     = "FFF3CD"
    C_WARN_FG     = "7D5A00"
    C_BAD_BG      = "FFE0E0"
    C_BAD_FG      = "8B1A1A"
    C_ROW_ALT     = "F7FFFE"
    C_BORDER      = "CCCCCC"

    thin = Side(style="thin", color=C_BORDER)
    thick_bottom = Side(style="medium", color=C_ACCENT_DARK)

    def hdr_font(size=11, bold=True, color=C_HEADER_FG):
        return Font(name="Arial", size=size, bold=bold, color=color)

    def cell_font(size=10, bold=False, color="000000"):
        return Font(name="Arial", size=size, bold=bold, color=color)

    def fill(hex_color):
        return PatternFill("solid", fgColor=hex_color)

    def border_all():
        return Border(left=Side(style="thin", color=C_BORDER),
                      right=Side(style="thin", color=C_BORDER),
                      top=Side(style="thin", color=C_BORDER),
                      bottom=Side(style="thin", color=C_BORDER))

    def score_style(val):
        if val >= 80:
            return C_GOOD_BG, C_GOOD_FG
        elif val >= 50:
            return C_WARN_BG, C_WARN_FG
        return C_BAD_BG, C_BAD_FG

    def set_cell(ws, row, col, value, font=None, bg=None, align="left", border=True, num_fmt=None):
        c = ws.cell(row=row, column=col, value=value)
        if font:
            c.font = font
        if bg:
            c.fill = fill(bg)
        c.alignment = Alignment(horizontal=align, vertical="center", wrap_text=False)
        if border:
            c.border = border_all()
        if num_fmt:
            c.number_format = num_fmt
        return c

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 1 — Summary Leaderboard
    # ══════════════════════════════════════════════════════════════════════
    ws = wb.active
    ws.title = "Summary"
    ws.sheet_view.showGridLines = False
    ws.row_dimensions[1].height = 40
    ws.row_dimensions[2].height = 18
    ws.row_dimensions[3].height = 18
    ws.row_dimensions[4].height = 18
    ws.row_dimensions[5].height = 30

    # Title block
    ws.merge_cells("A1:J1")
    title_cell = ws["A1"]
    title_cell.value = f"CVAT Quality Control — {session_name}"
    title_cell.font = Font(name="Arial", size=16, bold=True, color=C_HEADER_FG)
    title_cell.fill = fill(C_HEADER_BG)
    title_cell.alignment = Alignment(horizontal="left", vertical="center")
    ws["A1"].border = Border(bottom=Side(style="medium", color=C_ACCENT))

    ws.merge_cells("A2:J2")
    meta = ws["A2"]
    meta.value = f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}   |   IoU Threshold: {iou_threshold}%   |   Annotators: {len(results)}"
    meta.font = Font(name="Arial", size=9, color="888888")
    meta.fill = fill(C_HEADER_BG)
    meta.alignment = Alignment(horizontal="left", vertical="center")

    ws.merge_cells("A3:J3")
    ws["A3"].fill = fill(C_HEADER_BG)
    ws.merge_cells("A4:J4")
    ws["A4"].fill = fill("F0F0F0")

    # Column headers row
    headers = ["Rank", "Annotator Name", "Precision (%)", "Recall (%)",
               "Accuracy/F1 (%)", "Mean IoU (%)", "True Positives",
               "False Positives", "False Negatives", "Total GT Boxes"]
    col_widths = [8, 24, 16, 14, 17, 14, 16, 17, 17, 16]

    for col, (h, w) in enumerate(zip(headers, col_widths), 1):
        c = ws.cell(row=5, column=col, value=h)
        c.font = hdr_font(10)
        c.fill = fill(C_HEADER_BG)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = Border(bottom=Side(style="medium", color=C_ACCENT),
                          left=Side(style="thin", color="444444"),
                          right=Side(style="thin", color="444444"))
        ws.column_dimensions[get_column_letter(col)].width = w

    ws.row_dimensions[5].height = 32

    # Sort by accuracy descending
    sorted_results = sorted(results, key=lambda r: r["accuracy"], reverse=True)

    for i, r in enumerate(sorted_results):
        row = 6 + i
        ws.row_dimensions[row].height = 22
        row_bg = "FFFFFF" if i % 2 == 0 else C_ROW_ALT
        rank = i + 1
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))

        values = [
            medal,
            r["annotator"],
            r["precision"],
            r["recall"],
            r["accuracy"],
            r["mean_iou"],
            r["tp"],
            r["fp"],
            r["fn"],
            r["total_gt"],
        ]
        aligns = ["center", "left", "center", "center", "center", "center",
                  "center", "center", "center", "center"]

        for col, (val, aln) in enumerate(zip(values, aligns), 1):
            c = ws.cell(row=row, column=col, value=val)
            c.font = cell_font(10, bold=(col == 2))
            c.alignment = Alignment(horizontal=aln, vertical="center")
            c.border = border_all()

            # Color-code metric columns
            if col in (3, 4, 5, 6) and isinstance(val, (int, float)):
                bg, fg = score_style(val)
                c.fill = fill(bg)
                c.font = Font(name="Arial", size=10, bold=True, color=fg)
                c.number_format = '0.00"%"'
            elif col in (7, 8, 9, 10):
                c.fill = fill(row_bg)
                c.number_format = "#,##0"
            else:
                c.fill = fill(row_bg)

    # Team averages row
    if len(results) > 1:
        avg_row = 6 + len(results)
        ws.row_dimensions[avg_row].height = 24
        avg_vals = {
            "precision": sum(r["precision"] for r in results) / len(results),
            "recall": sum(r["recall"] for r in results) / len(results),
            "accuracy": sum(r["accuracy"] for r in results) / len(results),
            "mean_iou": sum(r["mean_iou"] for r in results) / len(results),
        }
        avg_data = ["", "TEAM AVERAGE", avg_vals["precision"], avg_vals["recall"],
                    avg_vals["accuracy"], avg_vals["mean_iou"], "", "", "", ""]
        for col, val in enumerate(avg_data, 1):
            c = ws.cell(row=avg_row, column=col, value=val)
            c.font = Font(name="Arial", size=10, bold=True, color=C_HEADER_FG)
            c.fill = fill(C_ACCENT_DARK)
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = Border(top=Side(style="medium", color="FFFFFF"))
            if col in (3, 4, 5, 6) and isinstance(val, float):
                c.number_format = '0.00"%"'

    # Legend
    legend_row = 6 + len(results) + (2 if len(results) > 1 else 1)
    ws.row_dimensions[legend_row].height = 18
    legend_items = [
        ("≥ 80%", C_GOOD_BG, C_GOOD_FG, "Good"),
        ("50–79%", C_WARN_BG, C_WARN_FG, "Needs Improvement"),
        ("< 50%", C_BAD_BG, C_BAD_FG, "Poor"),
    ]
    ws.cell(row=legend_row, column=1, value="Legend:").font = cell_font(9, bold=True)
    for j, (rng, bg, fg, label) in enumerate(legend_items):
        col = 2 + j * 2
        c = ws.cell(row=legend_row, column=col, value=f"{rng} = {label}")
        c.font = Font(name="Arial", size=9, bold=True, color=fg)
        c.fill = fill(bg)
        c.alignment = Alignment(horizontal="center")

    # Freeze panes
    ws.freeze_panes = "A6"

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 2+ — Per-annotator detail sheets
    # ══════════════════════════════════════════════════════════════════════
    for r in results:
        safe_name = r["annotator"][:28].replace("/", "-").replace("\\", "-").replace(":", "-").replace("*", "-").replace("?", "-").replace("[", "(").replace("]", ")")
        ws2 = wb.create_sheet(title=safe_name)
        ws2.sheet_view.showGridLines = False

        # Header
        ws2.merge_cells("A1:H1")
        ws2["A1"].value = f"{r['annotator']} — Detailed Report"
        ws2["A1"].font = Font(name="Arial", size=13, bold=True, color=C_HEADER_FG)
        ws2["A1"].fill = fill(C_HEADER_BG)
        ws2["A1"].alignment = Alignment(horizontal="left", vertical="center")
        ws2.row_dimensions[1].height = 34

        # Metric summary cards (row 3)
        ws2.row_dimensions[3].height = 14
        ws2.row_dimensions[4].height = 28
        ws2.row_dimensions[5].height = 22

        metric_cards = [
            ("Precision", r["precision"], 1),
            ("Recall", r["recall"], 3),
            ("Accuracy (F1)", r["accuracy"], 5),
            ("Mean IoU", r["mean_iou"], 7),
        ]
        for label, val, col in metric_cards:
            bg, fg = score_style(val)
            ws2.merge_cells(start_row=3, start_column=col, end_row=3, end_column=col+1)
            ws2.merge_cells(start_row=4, start_column=col, end_row=4, end_column=col+1)
            ws2.merge_cells(start_row=5, start_column=col, end_row=5, end_column=col+1)
            lbl_cell = ws2.cell(row=3, column=col, value=label)
            lbl_cell.font = Font(name="Arial", size=9, color="888888")
            lbl_cell.fill = fill("F5F5F5")
            lbl_cell.alignment = Alignment(horizontal="center")
            val_cell = ws2.cell(row=4, column=col, value=val)
            val_cell.font = Font(name="Arial", size=18, bold=True, color=fg)
            val_cell.fill = fill(bg)
            val_cell.alignment = Alignment(horizontal="center", vertical="center")
            val_cell.number_format = '0.00"%"'
            pct_cell = ws2.cell(row=5, column=col, value="%")
            pct_cell.font = Font(name="Arial", size=9, color=fg)
            pct_cell.fill = fill(bg)
            pct_cell.alignment = Alignment(horizontal="center")

        for col in range(1, 9):
            ws2.column_dimensions[get_column_letter(col)].width = 14

        # Per-frame table
        frame_hdr_row = 7
        ws2.row_dimensions[frame_hdr_row].height = 26
        frame_hdrs = ["Frame", "True Positives", "False Positives", "False Negatives", "Mean IoU (%)"]
        for col, h in enumerate(frame_hdrs, 1):
            c = ws2.cell(row=frame_hdr_row, column=col, value=h)
            c.font = hdr_font(10)
            c.fill = fill(C_HEADER_BG)
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = Border(bottom=Side(style="medium", color=C_ACCENT))

        ws2.column_dimensions["A"].width = 20
        for col_i in range(2, 6):
            ws2.column_dimensions[get_column_letter(col_i)].width = 18

        for fi, fr in enumerate(r.get("frame_results", [])):
            frow = frame_hdr_row + 1 + fi
            ws2.row_dimensions[frow].height = 20
            row_bg = "FFFFFF" if fi % 2 == 0 else C_ROW_ALT
            for col, val in enumerate([fr["frame_id"], fr["tp"], fr["fp"], fr["fn"], fr["mean_iou"]], 1):
                c = ws2.cell(row=frow, column=col, value=val)
                c.font = cell_font(9)
                c.fill = fill(row_bg)
                c.alignment = Alignment(horizontal="center" if col > 1 else "left", vertical="center")
                c.border = border_all()
                if col == 5:
                    c.number_format = '0.0"%"'

        # Per-label table
        if r.get("label_breakdown"):
            label_start_col = 7
            lbl_hdr_row = 7
            ws2.row_dimensions[lbl_hdr_row].height = 26
            lbl_hdrs = ["Class", "TP", "FP", "FN", "Precision (%)", "Recall (%)"]
            ws2.column_dimensions["G"].width = 18
            ws2.column_dimensions["H"].width = 10
            ws2.column_dimensions["I"].width = 10
            ws2.column_dimensions["J"].width = 10
            ws2.column_dimensions["K"].width = 16
            ws2.column_dimensions["L"].width = 14

            for col, h in enumerate(lbl_hdrs, label_start_col):
                c = ws2.cell(row=lbl_hdr_row, column=col, value=h)
                c.font = hdr_font(10)
                c.fill = fill(C_HEADER_BG)
                c.alignment = Alignment(horizontal="center", vertical="center")
                c.border = Border(bottom=Side(style="medium", color=C_ACCENT))

            for li, (lbl, s) in enumerate(r["label_breakdown"].items()):
                lrow = lbl_hdr_row + 1 + li
                ws2.row_dimensions[lrow].height = 20
                row_bg = "FFFFFF" if li % 2 == 0 else C_ROW_ALT
                row_vals = [lbl, s["tp"], s["fp"], s["fn"], s["precision"], s["recall"]]
                for col, val in enumerate(row_vals, label_start_col):
                    c = ws2.cell(row=lrow, column=col, value=val)
                    c.font = cell_font(9)
                    c.fill = fill(row_bg)
                    c.alignment = Alignment(horizontal="center" if col > label_start_col else "left", vertical="center")
                    c.border = border_all()
                    if col >= label_start_col + 4:
                        bg2, fg2 = score_style(val)
                        c.fill = fill(bg2)
                        c.font = Font(name="Arial", size=9, bold=True, color=fg2)
                        c.number_format = '0.0"%"'

    # ══════════════════════════════════════════════════════════════════════
    # SHEET — Raw Data
    # ══════════════════════════════════════════════════════════════════════
    ws_raw = wb.create_sheet(title="Raw Data")
    ws_raw.sheet_view.showGridLines = False
    raw_hdrs = ["Annotator", "Precision (%)", "Recall (%)", "Accuracy/F1 (%)",
                "Mean IoU (%)", "TP", "FP", "FN", "Total GT", "Total Pred", "IoU Threshold (%)"]
    for col, h in enumerate(raw_hdrs, 1):
        c = ws_raw.cell(row=1, column=col, value=h)
        c.font = hdr_font(10)
        c.fill = fill(C_HEADER_BG)
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws_raw.column_dimensions[get_column_letter(col)].width = 18

    for i, r in enumerate(results, 2):
        row_vals = [r["annotator"], r["precision"], r["recall"], r["accuracy"],
                    r["mean_iou"], r["tp"], r["fp"], r["fn"], r["total_gt"],
                    r["total_pred"], r["iou_threshold"]]
        for col, val in enumerate(row_vals, 1):
            c = ws_raw.cell(row=i, column=col, value=val)
            c.font = cell_font(10)
            c.alignment = Alignment(horizontal="center" if col > 1 else "left", vertical="center")
            c.border = border_all()

    # Output
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"CVAT_QC_{session_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return send_file(buf, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name=filename)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5050)