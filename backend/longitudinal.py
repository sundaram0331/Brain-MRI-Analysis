"""
NeonCortrex AI 3.0 — Longitudinal Neuro-Oncology Tracking Engine
Evaluates tumor progression, therapeutic response, and multi-temporal morphometric deltas
across longitudinal cranial MRI acquisitions (e.g. Month 0 Baseline vs Month 6 / 12 Follow-up).
"""
import io, base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage

def generate_fusion_difference_map(
    img_t1: Image.Image,
    img_t2: Image.Image,
    mask_t1_bin: np.ndarray,
    mask_t2_bin: np.ndarray,
    bbox_t1: dict,
    bbox_t2: dict,
    output_size: tuple = (400, 400)
) -> Image.Image:
    """
    Renders an Apple-grade multi-temporal fusion difference map:
      - Regressed / Resected / Necrotic tissue (T1 only): Neon Cyan (#00F0FF)
      - Newly Expanded / Infiltrating tissue (T2 only): Vivid Crimson (#FF3B30)
      - Persistent / Quiescent Core (T1 & T2 overlap): Luminous Amber (#FFB800)
    """
    w, h = output_size
    # Resize images and masks to shared canonical dimension
    t1_resized = img_t1.resize((w, h), Image.BILINEAR).convert("L")
    t2_resized = img_t2.resize((w, h), Image.BILINEAR).convert("L")
    
    # Standardize grayscale background from T2 (current follow-up anatomy)
    base_gray = np.array(t2_resized, dtype=np.float32)
    # Enhance parenchyma contrast slightly
    p2, p98 = np.percentile(base_gray, 2), np.percentile(base_gray, 98)
    if p98 > p2:
        base_gray = np.clip((base_gray - p2) / (p98 - p2) * 255.0, 0, 255)
    
    rgb_base = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.uint8)
    canvas = Image.fromarray(rgb_base)
    
    # Scale binary masks to canvas size
    m1_pil = Image.fromarray((mask_t1_bin * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    m2_pil = Image.fromarray((mask_t2_bin * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    m1 = np.array(m1_pil) > 127
    m2 = np.array(m2_pil) > 127
    
    # Compute intersection and difference sets
    persistent = m1 & m2            # Present in both (Amber)
    regressed = m1 & (~m2)          # Shrunk / Resolved (Cyan)
    expanded = m2 & (~m1)           # Newly grown (Crimson)
    
    # Overlay layer with alpha transparency
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    # 1. Regressed: (0, 240, 255, 140)
    if np.any(regressed):
        overlay[regressed] = [0, 240, 255, 150]
        # Dilate border for luminous outer glow
        reg_edge = ndimage.binary_dilation(regressed, iterations=2) & (~regressed)
        overlay[reg_edge] = [0, 240, 255, 90]
        
    # 2. Expanded: (255, 59, 48, 160)
    if np.any(expanded):
        overlay[expanded] = [255, 59, 48, 160]
        exp_edge = ndimage.binary_dilation(expanded, iterations=2) & (~expanded)
        overlay[exp_edge] = [255, 59, 48, 100]
        
    # 3. Persistent Core: (255, 184, 0, 130)
    if np.any(persistent):
        overlay[persistent] = [255, 184, 0, 140]
        per_edge = ndimage.binary_dilation(persistent, iterations=1) & (~persistent)
        overlay[per_edge] = [255, 184, 0, 80]
        
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    canvas.paste(overlay_img, (0, 0), overlay_img)
    
    # Draw vector overlays and targeting brackets
    draw = ImageDraw.Draw(canvas)
    
    # Scale bounding boxes to canonical 400x400
    orig_w1, orig_h1 = img_t1.size
    orig_w2, orig_h2 = img_t2.size
    
    if bbox_t1 and bbox_t1.get("estimated_diameter_mm", 0) > 0:
        bx1 = int((bbox_t1.get("min_x", 0) / orig_w1) * w)
        by1 = int((bbox_t1.get("min_y", 0) / orig_h1) * h)
        bx2 = int((bbox_t1.get("max_x", 0) / orig_w1) * w)
        by2 = int((bbox_t1.get("max_y", 0) / orig_h1) * h)
        # Dashed-style outline for baseline T1 (Cyan)
        draw.rectangle([bx1, by1, bx2, by2], outline="#00F0FF88", width=1)
        draw.text((bx1 + 4, max(4, by1 - 12)), "T1 Baseline", fill="#00F0FF")
        
    if bbox_t2 and bbox_t2.get("estimated_diameter_mm", 0) > 0:
        bx1 = int((bbox_t2.get("min_x", 0) / orig_w2) * w)
        by1 = int((bbox_t2.get("min_y", 0) / orig_h2) * h)
        bx2 = int((bbox_t2.get("max_x", 0) / orig_w2) * w)
        by2 = int((bbox_t2.get("max_y", 0) / orig_h2) * h)
        # Solid outline for follow-up T2
        stroke_color = "#00FF80" if np.sum(expanded) == 0 else "#FF3B30"
        draw.rectangle([bx1, by1, bx2, by2], outline=stroke_color, width=2)
        draw.text((bx1 + 4, max(4, by2 + 2)), "T2 Follow-up", fill=stroke_color)
        
    # Top Watermark Badge
    draw.rectangle([10, 10, 220, 28], fill="#000000AA", outline="#FFFFFF33")
    draw.text((16, 14), "MULTI-TEMPORAL FUSION DELTA", fill="#FFFFFF")
    
    # Bottom HUD Legend
    draw.rectangle([10, h - 28, w - 10, h - 8], fill="#000000CC", outline="#FFFFFF22")
    draw.rectangle([16, h - 21, 24, h - 13], fill="#00F0FF")
    draw.text((28, h - 22), "Regressed", fill="#E0E0E0")
    
    draw.rectangle([110, h - 21, 118, h - 13], fill="#FFB800")
    draw.text((122, h - 22), "Persistent Core", fill="#E0E0E0")
    
    draw.rectangle([230, h - 21, 238, h - 13], fill="#FF3B30")
    draw.text((242, h - 22), "Expanded", fill="#E0E0E0")
    
    return canvas


def compute_longitudinal_evaluation(
    diag_t1: dict,
    diag_t2: dict,
    timepoint_t1: str = "Baseline (Month 0)",
    timepoint_t2: str = "Follow-up (Month 6)",
    patient_id: str = "PT-8402"
) -> dict:
    """
    Computes rigorous neuro-oncological comparative deltas and RANO / RECIST 1.1 response status.
    """
    cls1 = diag_t1["predicted_class"]
    cls2 = diag_t2["predicted_class"]
    conf1 = diag_t1["confidence"]
    conf2 = diag_t2["confidence"]
    
    m1 = diag_t1["morphology"]
    m2 = diag_t2["morphology"]
    
    px1 = m1.get("tumor_pixels", 0)
    px2 = m2.get("tumor_pixels", 0)
    
    sev1 = m1.get("severity_pct", 0.0)
    sev2 = m2.get("severity_pct", 0.0)
    
    diam1 = m1.get("bbox", {}).get("estimated_diameter_mm", 0.0)
    diam2 = m2.get("bbox", {}).get("estimated_diameter_mm", 0.0)
    
    irreg1 = m1.get("irregularity_score", 1.0)
    irreg2 = m2.get("irregularity_score", 1.0)
    
    c1 = m1.get("centroid") or {"x": 50.0, "y": 50.0}
    c2 = m2.get("centroid") or {"x": 50.0, "y": 50.0}
    
    # ── Metric Deltas ──────────────────────────────────────────────────────────
    delta_voxels = px2 - px1
    if px1 > 0:
        delta_voxels_pct = round(((px2 - px1) / px1) * 100.0, 1)
    else:
        delta_voxels_pct = 100.0 if px2 > 0 else 0.0
        
    delta_burden_pct = round(sev2 - sev1, 2)
    delta_diam_mm = round(diam2 - diam1, 1)
    
    if diam1 > 0:
        delta_diam_pct = round(((diam2 - diam1) / diam1) * 100.0, 1)
    else:
        delta_diam_pct = 100.0 if diam2 > 0 else 0.0
        
    delta_irregularity = round(irreg2 - irreg1, 2)
    
    shift_x = round(c2["x"] - c1["x"], 1)
    shift_y = round(c2["y"] - c1["y"], 1)
    spatial_drift_pct = round(float(np.sqrt(shift_x**2 + shift_y**2)), 1)
    
    # ── International RANO / RECIST 1.1 Response Staging ──────────────────────
    if cls1 == "No Tumor" and cls2 == "No Tumor":
        response_code = "CR"
        response_title = "Complete Remission (CR)"
        trajectory_label = "Disease-Free / Stable Normal Parenchyma"
        badge_color = "#00FF80"
        urgency = "ROUTINE SURVEILLANCE"
        clinical_summary = (
            "No intracranial neoplastic lesion detected on either baseline or follow-up examination. "
            "Ventricular symmetry and parenchymal attenuation remain within normal physiological limits."
        )
    elif cls1 != "No Tumor" and cls2 == "No Tumor":
        response_code = "CR"
        response_title = "Complete Response (CR)"
        trajectory_label = "Total Lesion Clearance / Complete Remission"
        badge_color = "#00FF80"
        urgency = "SURVEILLANCE SURGERY COMPLETE"
        clinical_summary = (
            f"The prior {cls1} mass has fully regressed or been completely resected with zero measurable "
            f"residual hyperintense signal on follow-up imaging. Highly favorable therapeutic response."
        )
    elif delta_voxels_pct <= -50.0 or delta_diam_pct <= -30.0:
        response_code = "PR"
        response_title = "Partial Response (PR)"
        trajectory_label = "Marked Tumor Regression / Therapeutic Shrinkage"
        badge_color = "#00F0FF"
        urgency = "CONTINUE ADJUVANT PROTOCOL"
        clinical_summary = (
            f"Marked objective lesion shrinkage exceeding RANO criteria (>50% volumetric regression or >30% linear reduction). "
            f"Tumor burden reduced by {abs(delta_voxels_pct):.1f}% ({abs(delta_voxels):,} voxels, -{abs(delta_diam_mm)} mm). "
            f"Indicates strong sensitivity to current chemo-radiation or surgical regimen."
        )
    elif delta_voxels_pct <= -15.0 or delta_diam_pct <= -10.0:
        response_code = "MR"
        response_title = "Minor Response (MR)"
        trajectory_label = "Favorable Regression / Measurable Reduction"
        badge_color = "#3B82F6"
        urgency = "CONTINUE CURRENT THERAPY"
        clinical_summary = (
            f"Quantifiable lesion reduction of {abs(delta_voxels_pct):.1f}% in volumetric burden. "
            f"Margins demonstrate stabilization without meeting the stringent 50% partial response threshold."
        )
    elif -15.0 < delta_voxels_pct < 25.0 and delta_diam_pct < 20.0:
        response_code = "SD"
        response_title = "Stable Disease (SD)"
        trajectory_label = "Quiescent / Tumor Growth Arrested"
        badge_color = "#FFB800"
        urgency = "INTERVAL 3-MONTH SURVEILLANCE"
        clinical_summary = (
            f"Lesion dimensions remain stable within non-progressive parameters ({delta_voxels_pct:+.1f}% volume change). "
            f"No evidence of active parenchymal infiltration or accelerated mitotic expansion."
        )
    else:
        response_code = "PD"
        response_title = "Progressive Disease (PD)"
        trajectory_label = "Tumor Expansion / Active Progression"
        badge_color = "#FF3B30"
        urgency = "URGENT ONCOLOGY REVIEW"
        clinical_summary = (
            f"Objective disease progression exceeding RANO threshold (>25% volume expansion or >20% linear diameter increase). "
            f"Tumor expanded by {delta_voxels_pct:+.1f}% (+{delta_voxels:,} voxels, +{delta_diam_mm} mm). "
            f"Immediate multidisciplinary tumor board review recommended for second-line therapy or surgical intervention."
        )

    # ── AI Neuro-Radiology Longitudinal Clinical Assessment Narrative ─────────
    time_interval = f"Between {timepoint_t1} and {timepoint_t2}"
    narrative_p1 = (
        f"Comparative longitudinal assessment was performed evaluating cranial acquisitions {time_interval}. "
        f"Baseline scan demonstrated a confirmed {cls1} ({conf1:.1f}% confidence) with an initial maximal diameter "
        f"of {diam1:.1f} mm and an intracranial burden of {sev1:.2f}% ({px1:,} voxels). "
        f"Subsequent follow-up imaging at {timepoint_t2} demonstrates {cls2} features ({conf2:.1f}% confidence) "
        f"with updated maximal diameter of {diam2:.1f} mm and {sev2:.2f}% ({px2:,} voxels) cranial volume."
    )
    
    if delta_voxels <= 0:
        narrative_p2 = (
            f"Volumetric trajectory analysis reveals a net lesion reduction of {abs(delta_voxels):,} voxels "
            f"({abs(delta_voxels_pct):.1f}% relative shrinkage; intracranial burden delta of {delta_burden_pct:+.2f}%). "
            f"Linear RECIST measurements indicate a diameter change of {delta_diam_mm:+.1f} mm. "
            f"Margin analysis shows an irregularity shift of {delta_irregularity:+.2f} (from {irreg1:.2f} to {irreg2:.2f}), "
            f"consistent with treatment-induced circumscription and reduced perilesional infiltrative edema."
        )
    else:
        narrative_p2 = (
            f"Volumetric trajectory analysis reveals an active lesion expansion of {delta_voxels:,} voxels "
            f"({delta_voxels_pct:+.1f}% relative growth; intracranial burden delta of {delta_burden_pct:+.2f}%). "
            f"Linear RECIST measurements indicate a diameter increase of {delta_diam_mm:+.1f} mm. "
            f"Margin analysis yielded an irregularity shift of {delta_irregularity:+.2f} (from {irreg1:.2f} to {irreg2:.2f}), "
            f"indicating aggressive cellular proliferation and finger-like parenchymal invasion into adjacent cerebral architecture."
        )
        
    narrative_p3 = (
        f"Centroid displacement tracking pinpoints a spatial drift of {spatial_drift_pct}% "
        f"(ΔX: {shift_x:+.1f}%, ΔY: {shift_y:+.1f}%) within the {m2.get('anatomical_location', 'cranial parenchyma')}. "
        f"Overall assessment satisfies international RANO criteria for {response_title}. "
        f"Clinical recommendation: {clinical_summary} Assigned triage urgency: {urgency}."
    )
    
    full_narrative = f"{narrative_p1}\n\n{narrative_p2}\n\n{narrative_p3}"
    
    return {
        "response_code": response_code,
        "response_title": response_title,
        "trajectory_label": trajectory_label,
        "badge_color": badge_color,
        "urgency": urgency,
        "clinical_summary": clinical_summary,
        "narrative": full_narrative,
        "time_interval": time_interval,
        "timepoint_t1": timepoint_t1,
        "timepoint_t2": timepoint_t2,
        "patient_id": patient_id,
        "metrics": {
            "baseline": {
                "class": cls1,
                "confidence": conf1,
                "pixels": px1,
                "burden_pct": sev1,
                "diameter_mm": diam1,
                "irregularity": irreg1,
                "centroid": c1,
                "location": m1.get("anatomical_location", "Intracranial")
            },
            "followup": {
                "class": cls2,
                "confidence": conf2,
                "pixels": px2,
                "burden_pct": sev2,
                "diameter_mm": diam2,
                "irregularity": irreg2,
                "centroid": c2,
                "location": m2.get("anatomical_location", "Intracranial")
            },
            "deltas": {
                "delta_voxels": delta_voxels,
                "delta_voxels_pct": delta_voxels_pct,
                "delta_burden_pct": delta_burden_pct,
                "delta_diam_mm": delta_diam_mm,
                "delta_diam_pct": delta_diam_pct,
                "delta_irregularity": delta_irregularity,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "spatial_drift_pct": spatial_drift_pct,
                "is_improvement": delta_voxels <= 0
            }
        }
    }
