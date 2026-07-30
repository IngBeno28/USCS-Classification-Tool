# main.py  -  USCS Soil Classification System (ASTM D2487)
# Automation_hub Engineering Group Limited

import io
import os
import tempfile
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
from PIL import Image

from branding import (
    CLIENT_NAME, APP_TITLE, PRIMARY_COLOR, LOGO_PATH, FOOTER_NOTE, LOGO_ALT_TEXT,
    COMPANY_ADDRESS, COMPANY_PHONE, COMPANY_EMAIL, COMPANY_WEBSITE
)

st.set_page_config(page_title=APP_TITLE, page_icon="🧱", layout="wide")

if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =============================================================================
# 1. GRADATION CURVE HELPERS (particle size distribution)
# =============================================================================

# Standard sieve set used across the app (mm). Users can add/remove rows in
# the single-sample editor; batch CSVs use exactly these columns.
DEFAULT_SIEVES_MM = [19.0, 9.5, 4.75, 2.0, 0.425, 0.075]


def interp_percent_passing_at_size(sizes_mm, percent_passing, target_size_mm):
    """% passing at a given grain size, via log(size)-linear interpolation
    (standard practice for reading a gradation curve)."""
    pairs = sorted(zip(sizes_mm, percent_passing))
    d = [p[0] for p in pairs]
    p = [p[1] for p in pairs]
    if target_size_mm <= d[0]:
        return p[0]
    if target_size_mm >= d[-1]:
        return p[-1]
    for i in range(len(d) - 1):
        if d[i] <= target_size_mm <= d[i + 1]:
            if d[i] == d[i + 1]:
                return p[i]
            logd1, logd2 = np.log10(d[i]), np.log10(d[i + 1])
            logt = np.log10(target_size_mm)
            frac = (logt - logd1) / (logd2 - logd1)
            return p[i] + frac * (p[i + 1] - p[i])
    return None


def interp_size_at_percent_passing(sizes_mm, percent_passing, target_percent):
    """Grain size (D-value) at a given % passing, via log(size)-linear
    interpolation. Returns None if the curve doesn't reach that % passing
    (common when fines are low and D10 would fall below the finest sieve)."""
    pairs = sorted(zip(percent_passing, sizes_mm))
    p = [x[0] for x in pairs]
    d = [x[1] for x in pairs]
    if target_percent < p[0] or target_percent > p[-1]:
        return None
    if target_percent == p[0]:
        return d[0]
    if target_percent == p[-1]:
        return d[-1]
    for i in range(len(p) - 1):
        if p[i] <= target_percent <= p[i + 1]:
            if p[i] == p[i + 1]:
                return d[i]
            logd1, logd2 = np.log10(d[i]), np.log10(d[i + 1])
            frac = (target_percent - p[i]) / (p[i + 1] - p[i])
            return 10 ** (logd1 + frac * (logd2 - logd1))
    return None


def gradation_summary(sizes_mm, percent_passing):
    """Derive %gravel / %sand / %fines and Cu/Cc from a gradation curve."""
    pass_475 = interp_percent_passing_at_size(sizes_mm, percent_passing, 4.75)
    pass_075 = interp_percent_passing_at_size(sizes_mm, percent_passing, 0.075)
    pct_gravel = max(0.0, 100.0 - pass_475)
    pct_sand = max(0.0, pass_475 - pass_075)
    pct_fines = max(0.0, pass_075)

    D10 = interp_size_at_percent_passing(sizes_mm, percent_passing, 10)
    D30 = interp_size_at_percent_passing(sizes_mm, percent_passing, 30)
    D60 = interp_size_at_percent_passing(sizes_mm, percent_passing, 60)
    Cu = (D60 / D10) if (D10 and D60) else None
    Cc = ((D30 ** 2) / (D10 * D60)) if (D10 and D30 and D60) else None

    return {
        "pct_gravel": round(pct_gravel, 1),
        "pct_sand": round(pct_sand, 1),
        "pct_fines": round(pct_fines, 1),
        "D10": D10, "D30": D30, "D60": D60,
        "Cu": round(Cu, 2) if Cu else None,
        "Cc": round(Cc, 2) if Cc else None,
    }


# =============================================================================
# 2. PLASTICITY CHART HELPERS
# =============================================================================

def a_line_PI(LL):
    return max(0.0, 0.73 * (LL - 20))


def u_line_PI(LL):
    return max(0.0, 0.9 * (LL - 8))


def plots_above_A_line(LL, PI):
    return PI >= a_line_PI(LL)


# =============================================================================
# 3. USCS CLASSIFICATION ENGINE (ASTM D2487)
# =============================================================================

def classify_fine_grained(LL, PI, is_np, is_organic):
    if is_np:
        PI = 0.0
    high_plasticity = LL >= 50
    above_A = plots_above_A_line(LL, PI)

    if is_organic:
        symbol = "OH" if high_plasticity else "OL"
        borderline = False
    elif not high_plasticity:
        if PI < 4:
            symbol, borderline = "ML", False
        elif PI > 7:
            symbol, borderline = ("CL", False) if above_A else ("ML", False)
        else:  # 4 <= PI <= 7 : hatched CL-ML zone
            symbol, borderline = ("CL-ML", True) if above_A else ("ML", False)
    else:
        symbol, borderline = ("CH", False) if above_A else ("MH", False)

    return symbol, borderline


def classify_coarse_grained(is_gravel, pct_fines, Cu, Cc, LL, PI, is_np, is_organic):
    prefix = "G" if is_gravel else "S"
    if is_np:
        PI = 0.0
    above_A = plots_above_A_line(LL, PI) if LL else False

    # Well- vs poorly-graded (only meaningful when fines are low enough
    # that a gradation-based symbol is used at all)
    if Cu is not None and Cc is not None:
        if is_gravel:
            well_graded = (Cu >= 4) and (1 <= Cc <= 3)
        else:
            well_graded = (Cu >= 6) and (1 <= Cc <= 3)
        grading_symbol = "W" if well_graded else "P"
        grading_known = True
    else:
        grading_symbol = "P"
        grading_known = False  # not enough gradation data to confirm  -  flagged in notes

    # Silty (M) vs clayey (C) vs borderline, based on fines plasticity
    if PI < 4 or not above_A:
        fines_symbol, fines_borderline = "M", False
    elif 4 <= PI <= 7 and above_A:
        fines_symbol, fines_borderline = "C-M", True
    else:
        fines_symbol, fines_borderline = "C", False

    notes = []
    INSUFFICIENT_GRADING_NOTE = (
        "Insufficient gradation data to confirm Cu/Cc  -  grading (W/P) assumed poorly-graded "
        "pending full sieve data (this is expected when the required D-value falls below the "
        "finest sieve tested, e.g. D10 when fines are near or above 10%)."
    )

    if pct_fines < 5:
        if not grading_known:
            notes.append(INSUFFICIENT_GRADING_NOTE)
        return f"{prefix}{grading_symbol}", False, notes
    elif pct_fines > 12:
        # Grading (Cu/Cc) is not part of the ASTM procedure once fines exceed
        # 12% - classification is driven entirely by the fines' plasticity -
        # so an unresolved Cu/Cc here is irrelevant and not flagged.
        if fines_borderline:
            notes.append("Fines plot in the 4<=PI<=7 hatched zone on/above the A-line  -  borderline silty/clayey classification.")
            return f"{prefix}C-{prefix}M", True, notes
        return f"{prefix}{fines_symbol}", False, notes
    else:
        # 5-12% fines: dual symbol required, and grading DOES matter here
        if not grading_known:
            notes.append(INSUFFICIENT_GRADING_NOTE)
        if fines_borderline:
            notes.append(
                "Both the gradation (5-12% fines, dual symbol required) and the fines plasticity "
                "(4<=PI<=7 hatched zone) are borderline. Report as a dual symbol using engineering "
                "judgment on the fines behavior, e.g. "
                f"{prefix}{grading_symbol}-{prefix}C or {prefix}{grading_symbol}-{prefix}M."
            )
            return f"{prefix}{grading_symbol}-{prefix}C/{prefix}M", True, notes
        symbol = f"{prefix}{grading_symbol}-{prefix}{fines_symbol}"
        return symbol, True, notes


def classify_uscs(pct_gravel, pct_sand, pct_fines, Cu, Cc, LL, PI, is_np, is_organic, is_peat):
    """Top-level ASTM D2487 classification. Returns a result dict."""
    if is_peat:
        return {
            "symbol": "PT", "borderline": False, "category": "Highly Organic Soil",
            "notes": ["Identified by visual-manual examination (color, odor, fibrous/spongy texture) "
                      "per ASTM D2487  -  the standard classification procedure does not apply to peat."]
        }

    if pct_fines >= 50:
        symbol, borderline = classify_fine_grained(LL, PI, is_np, is_organic)
        return {"symbol": symbol, "borderline": borderline,
                "category": "Fine-Grained Soil (≥50% passing No. 200)", "notes": []}
    else:
        is_gravel = pct_gravel > pct_sand
        symbol, borderline, notes = classify_coarse_grained(
            is_gravel, pct_fines, Cu, Cc, LL, PI, is_np, is_organic
        )
        category = "Coarse-Grained Soil  -  Gravel" if is_gravel else "Coarse-Grained Soil  -  Sand"
        return {"symbol": symbol, "borderline": borderline, "category": category, "notes": notes}


USCS_DESCRIPTIONS = {
    "GW": "Well-graded gravel, little to no fines. Excellent bearing capacity, permeable, minimal settlement  -  ideal for structural fill, drainage layers, and road base.",
    "GP": "Poorly-graded gravel, little to no fines. Good bearing capacity but more uniform void structure than GW; still generally excellent as fill/base with good compaction.",
    "GM": "Silty gravel, gravel-sand-silt mixture. Good bearing capacity but fines increase moisture/frost sensitivity; permeability reduced versus clean gravel.",
    "GC": "Clayey gravel, gravel-sand-clay mixture. Fair to good strength; clay fines add cohesion but reduce permeability and increase moisture sensitivity.",
    "SW": "Well-graded sand, little to no fines. Good bearing capacity, free-draining  -  suitable for fill and drainage applications.",
    "SP": "Poorly-graded (uniform) sand, little to no fines. Moderate bearing capacity, free-draining, but more prone to particle rearrangement/settlement under load or vibration.",
    "SM": "Silty sand, sand-silt mixture. Fair bearing capacity; fines make it moisture- and frost-sensitive, reduce permeability.",
    "SC": "Clayey sand, sand-clay mixture. Fair strength with some cohesion from clay fines; reduced permeability, moisture-sensitive.",
    "ML": "Inorganic silt, low plasticity. Low strength, high frost susceptibility, poor drainage  -  generally a poor subgrade/fill material.",
    "CL": "Inorganic clay, low to medium plasticity (lean clay). Moderate strength when compacted and dry; shrink-swell and moisture sensitivity are moderate.",
    "CL-ML": "Borderline silt/lean clay (hatched zone). Behavior intermediate between ML and CL  -  treat with the more conservative (ML-like) assumption unless further testing narrows it down.",
    "MH": "Inorganic silt, high plasticity (elastic silt). Poor engineering behavior  -  high compressibility, low strength, poor drainage.",
    "CH": "Inorganic clay, high plasticity (fat clay). High shrink-swell potential, low permeability, low strength when wet  -  a difficult subgrade material without treatment.",
    "OL": "Organic silt/clay, low plasticity. Poor engineering properties, compressible, prone to decomposition/settlement over time  -  unsuitable for structural support.",
    "OH": "Organic silt/clay, high plasticity. Very poor engineering properties  -  highly compressible and weak; unsuitable for structural support without significant treatment.",
    "PT": "Peat and other highly organic soils. Extremely poor engineering properties  -  very high compressibility, very low strength; unsuitable for foundations without full removal/replacement.",
}


def get_description(symbol: str) -> str:
    if symbol in USCS_DESCRIPTIONS:
        return USCS_DESCRIPTIONS[symbol]
    # Dual/borderline symbols (e.g. GW-GM, SC-SM): combine component descriptions
    for sep in ["-", "/"]:
        if sep in symbol:
            parts = [p for p in symbol.replace("/", "-").split("-") if p in USCS_DESCRIPTIONS]
            if len(parts) == 2:
                return (f"Dual/borderline classification between {parts[0]} and {parts[1]}. "
                         f"{parts[0]}: {USCS_DESCRIPTIONS[parts[0]]} {parts[1]}: {USCS_DESCRIPTIONS[parts[1]]}")
    return "Borderline/dual classification  -  refer to component symbol descriptions and apply engineering judgment."


def generate_interpretation(result: dict, grad: dict, LL: float, PI: float, is_np: bool) -> str:
    symbol = result["symbol"]
    lines = [f"USCS Classification: {symbol}", "", get_description(symbol), ""]

    if grad:
        lines.append(f"Gradation: {grad['pct_gravel']}% gravel, {grad['pct_sand']}% sand, {grad['pct_fines']}% fines.")
        if grad.get("Cu") is not None:
            lines.append(f"Coefficient of Uniformity (Cu) = {grad['Cu']}, Coefficient of Curvature (Cc) = {grad['Cc']}.")

    if not is_np and LL:
        lines.append(f"Liquid Limit = {LL}, Plasticity Index = {PI}  -  "
                     f"{'plots above' if plots_above_A_line(LL, PI) else 'plots below'} the A-line.")
    elif is_np:
        lines.append("Soil is non-plastic (NP).")

    if result.get("borderline"):
        lines.append("This is a borderline/dual classification  -  both components should be considered in design.")

    if result.get("notes"):
        lines.append("")
        lines.append("Notes:")
        for n in result["notes"]:
            lines.append(f"- {n}")

    return "\n".join(lines)


# =============================================================================
# 4. CHARTS
# =============================================================================

def create_gradation_chart(sizes_mm, percent_passing, label="Sample"):
    fig, ax = plt.subplots(figsize=(6, 4))
    pairs = sorted(zip(sizes_mm, percent_passing))
    d = [p[0] for p in pairs]
    p = [p[1] for p in pairs]
    ax.plot(d, p, marker='o', color='#0052cc', linewidth=2, label=label)
    ax.axvline(4.75, color='green', linestyle='--', linewidth=1, label='No. 4 (4.75mm)')
    ax.axvline(0.075, color='red', linestyle='--', linewidth=1, label='No. 200 (0.075mm)')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel("Grain Size (mm)")
    ax.set_ylabel("% Passing")
    ax.set_ylim(0, 105)
    ax.set_title("Particle Size Distribution")
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def create_plasticity_chart(points: list):
    """points: list of (label, LL, PI) tuples to plot on the Casagrande chart."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    LL_range = np.linspace(0, 110, 300)
    a_vals = [a_line_PI(ll) for ll in LL_range]
    u_vals = [u_line_PI(ll) for ll in LL_range]
    ax.plot(LL_range, a_vals, 'k-', linewidth=1.5, label="A-line: PI=0.73(LL-20)")
    ax.plot(LL_range, u_vals, 'k--', linewidth=1, label="U-line: PI=0.9(LL-8)")
    ax.axvline(50, color='grey', linestyle=':', linewidth=1)

    # Hatched CL-ML zone (4 <= PI <= 7, on/above A-line, LL < 50)
    ll_fill = np.linspace(20, 50, 200)
    a_vals_fill = np.array([a_line_PI(ll) for ll in ll_fill])
    upper = np.minimum(a_vals_fill, 7)
    lower = np.full_like(ll_fill, 4.0)
    mask = upper >= lower
    ax.fill_between(ll_fill[mask], lower[mask], upper[mask], color='grey', alpha=0.3, label="CL-ML hatched zone")

    ax.text(15, 55, "MH or OH", fontsize=8, color='dimgrey')
    ax.text(70, 55, "CH", fontsize=8, color='dimgrey')
    ax.text(15, 12, "ML or OL", fontsize=8, color='dimgrey')
    ax.text(60, 15, "CL", fontsize=8, color='dimgrey')

    show_legend_points = len(points) <= 8
    for i, (label, LL, PI) in enumerate(points):
        ax.plot(LL, PI, marker='o', markersize=9, label=label if show_legend_points else None)
        if not show_legend_points:
            ax.annotate(label, (LL, PI), fontsize=7, xytext=(3, 3), textcoords='offset points')

    ax.set_xlim(0, 110)
    ax.set_ylim(0, 70)
    ax.set_xlabel("Liquid Limit (LL)")
    ax.set_ylabel("Plasticity Index (PI)")
    ax.set_title("Casagrande Plasticity Chart")
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()


# =============================================================================
# 5. PDF REPORT
# =============================================================================

def hex_to_rgb(hex_color):
    try:
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (0, 82, 204)


class BrandedPDF(FPDF):
    def footer(self):
        contact_parts = []
        if COMPANY_PHONE:
            contact_parts.append(f"Tel: {COMPANY_PHONE}")
        if COMPANY_EMAIL:
            contact_parts.append(f"Email: {COMPANY_EMAIL}")
        if COMPANY_WEBSITE:
            contact_parts.append(f"Web: {COMPANY_WEBSITE}")
        if COMPANY_ADDRESS:
            contact_parts.append(COMPANY_ADDRESS)
        contact_line = " | ".join(contact_parts)

        self.set_y(-24 if contact_line else -18)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.set_font("Arial", '', 8)
        self.set_text_color(120, 120, 120)
        footer_left = f"{CLIENT_NAME} | {FOOTER_NOTE}" if FOOTER_NOTE else CLIENT_NAME
        self.cell(0, 6, footer_left.encode('latin-1', errors='replace').decode('latin-1'), 0, 0, 'L')
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", 0, 1, 'R')
        if contact_line:
            self.set_x(10)
            self.cell(0, 6, contact_line.encode('latin-1', errors='replace').decode('latin-1'), 0, 1, 'L')
        self.set_text_color(0, 0, 0)


def create_pdf_report(samples: list, project_name: str, client_name: str = "",
                       engineer_name: str = "", stamp_image_path: str = None) -> Optional[bytes]:
    """samples: list of dicts, each with keys:
    sample_id, result (from classify_uscs), grad, LL, PL, PI, is_np,
    gradation_chart_png, plasticity_point (label, LL, PI)
    """
    try:
        pdf = BrandedPDF()
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=22)

        def safe_text(text):
            if not isinstance(text, str):
                text = str(text)
            return text.encode('latin-1', errors='replace').decode('latin-1')

        def draw_table_row(col_widths, values, aligns=None, line_height=5, min_row_height=8, bold=False):
            if aligns is None:
                aligns = ['L'] * len(values)
            pdf.set_font("Arial", 'B' if bold else '', 10)
            x_start = (pdf.w - sum(col_widths)) / 2

            def wrap(text, width):
                text = safe_text(text)
                usable = width - 2
                words = text.split(' ')
                lines, current = [], ""
                for word in words:
                    trial = (current + " " + word).strip()
                    if not current or pdf.get_string_width(trial) <= usable:
                        current = trial
                    else:
                        lines.append(current)
                        current = word
                if current:
                    lines.append(current)
                return lines or [""]

            wrapped = [wrap(v, w) for v, w in zip(values, col_widths)]
            n_lines = max(len(w) for w in wrapped)
            row_height = max(min_row_height, n_lines * line_height)

            if pdf.get_y() + row_height > pdf.h - pdf.b_margin:
                pdf.add_page()

            y_start = pdf.get_y()
            x = x_start
            for width, lines, align in zip(col_widths, wrapped, aligns):
                pdf.rect(x, y_start, width, row_height)
                pdf.set_xy(x, y_start + (row_height - len(lines) * line_height) / 2)
                for line in lines:
                    pdf.set_x(x)
                    pdf.cell(width, line_height, line, 0, 2, align)
                x += width
            pdf.set_y(y_start + row_height)
            pdf.set_x(pdf.l_margin)  # never leave the cursor at a column's x position

        # --- Cover Page ---
        pdf.add_page()
        accent_rgb = hex_to_rgb(PRIMARY_COLOR)
        pdf.set_fill_color(*accent_rgb)
        pdf.rect(0, 0, pdf.w, 10, 'F')

        logo_bottom = 28
        if LOGO_PATH and os.path.exists(LOGO_PATH):
            try:
                with Image.open(LOGO_PATH) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    temp_logo_path = os.path.join(tempfile.gettempdir(),
                                                f"temp_logo_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
                    img.save(temp_logo_path, format='JPEG', quality=95)
                    pdf.image(temp_logo_path, x=(pdf.w - 40) / 2, y=22, w=40)
                    os.unlink(temp_logo_path)
                logo_bottom = 22 + 40 + 8
            except Exception as e:
                st.error(f"Logo processing error: {str(e)}")

        pdf.set_y(logo_bottom)
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(*accent_rgb)
        pdf.cell(0, 14, safe_text("USCS Soil Classification Report"), 0, 1, 'C')
        pdf.set_text_color(90, 90, 90)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, safe_text(APP_TITLE), 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)

        pdf.ln(4)
        pdf.set_draw_color(*accent_rgb)
        pdf.set_line_width(0.6)
        pdf.line(50, pdf.get_y(), pdf.w - 50, pdf.get_y())
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        pdf.ln(12)

        info_rows = [("Project", project_name)]
        if client_name:
            info_rows.append(("Prepared For", client_name))
        info_rows.append(("Prepared By", CLIENT_NAME))
        info_rows.append(("Date Generated", datetime.now().strftime('%Y-%m-%d %H:%M')))
        info_rows.append(("Total Samples", str(len(samples))))

        panel_w, label_w, row_h = 150, 55, 9
        x0 = (pdf.w - panel_w) / 2
        y0 = pdf.get_y()
        panel_h = row_h * len(info_rows)
        pdf.set_draw_color(200, 200, 200)
        pdf.rect(x0, y0, panel_w, panel_h)
        for idx, (label, value) in enumerate(info_rows):
            y = y0 + idx * row_h
            if idx > 0:
                pdf.line(x0, y, x0 + panel_w, y)
            pdf.set_xy(x0 + 4, y)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(label_w - 4, row_h, safe_text(label), 0, 0, 'L')
            pdf.set_font("Arial", '', 11)
            pdf.cell(panel_w - label_w - 4, row_h, safe_text(value), 0, 0, 'L')
        pdf.set_draw_color(0, 0, 0)
        pdf.set_y(y0 + panel_h + 14)

        if FOOTER_NOTE:
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 8, safe_text(FOOTER_NOTE), 0, 1, 'C')
            pdf.set_text_color(0, 0, 0)

        # --- Per-Sample Pages ---
        for i, s in enumerate(samples, 1):
            pdf.add_page()
            result = s["result"]
            grad = s.get("grad") or {}

            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, safe_text(f"Sample: {s.get('sample_id', f'Sample {i}')}"), 0, 1, 'C')
            pdf.set_font("Arial", 'B', 20)
            pdf.set_text_color(*accent_rgb)
            pdf.cell(0, 12, safe_text(f"USCS Symbol: {result['symbol']}"), 0, 1, 'C')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 11)
            pdf.cell(0, 7, safe_text(result['category']), 0, 1, 'C')
            pdf.ln(4)

            col_widths = [70, 60, 30]
            draw_table_row(col_widths, ["Parameter", "Value", "Unit"], aligns=['L', 'C', 'C'], bold=True)
            rows = []
            if grad:
                rows += [
                    ("Gravel", grad.get('pct_gravel', '-'), "%"),
                    ("Sand", grad.get('pct_sand', '-'), "%"),
                    ("Fines (Passing No.200)", grad.get('pct_fines', '-'), "%"),
                    ("Cu (Coeff. of Uniformity)", grad.get('Cu') if grad.get('Cu') is not None else "N/A", "-"),
                    ("Cc (Coeff. of Curvature)", grad.get('Cc') if grad.get('Cc') is not None else "N/A", "-"),
                ]
            rows += [
                ("Liquid Limit (LL)", s.get('LL', '-') if not s.get('is_np') else "N/A (NP)", "%"),
                ("Plastic Limit (PL)", s.get('PL', '-') if not s.get('is_np') else "N/A (NP)", "%"),
                ("Plasticity Index (PI)", s.get('PI', 0) if not s.get('is_np') else 0, "%"),
                ("Organic", "Yes" if s.get('is_organic') else "No", ""),
                ("Peat (Visual ID)", "Yes" if s.get('is_peat') else "No", ""),
            ]
            for p, v, u in rows:
                draw_table_row(col_widths, [p, v, u], aligns=['L', 'C', 'C'])

            pdf.ln(4)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, safe_text("Engineering Interpretation"), 0, 1, 'L')
            pdf.set_font("Arial", '', 10)
            interp = generate_interpretation(result, grad, s.get('LL', 0), s.get('PI', 0), s.get('is_np', False))
            for line in interp.split("\n"):
                line = line.strip()
                if not line:
                    pdf.ln(2)
                    continue
                clean_line = line.lstrip("- ").strip()
                if not clean_line:
                    continue
                if line.endswith(":") or line.startswith("USCS Classification"):
                    pdf.set_font("Arial", 'B', 10)
                else:
                    pdf.set_font("Arial", '', 10)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5.5, safe_text(clean_line))

            if s.get("gradation_chart_png"):
                pdf.ln(4)
                try:
                    chart_path = os.path.join(tempfile.gettempdir(),
                                            f"temp_grad_{i}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
                    with open(chart_path, 'wb') as f:
                        f.write(s["gradation_chart_png"])
                    if pdf.get_y() + 80 > pdf.h - pdf.b_margin:
                        pdf.add_page()
                    pdf.image(chart_path, x=(pdf.w - 150) / 2, w=150)
                    os.unlink(chart_path)
                except Exception:
                    pass

        # --- Combined Plasticity Chart Page ---
        plasticity_points = [s["plasticity_point"] for s in samples if s.get("plasticity_point")]
        if plasticity_points:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 12, safe_text("Casagrande Plasticity Chart"), 0, 1, 'C')
            pdf.ln(2)
            try:
                fig = create_plasticity_chart(plasticity_points)
                png = fig_to_png_bytes(fig)
                plt.close(fig)
                chart_path = os.path.join(tempfile.gettempdir(),
                                        f"temp_plasticity_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
                with open(chart_path, 'wb') as f:
                    f.write(png)
                pdf.image(chart_path, x=(pdf.w - 160) / 2, w=160)
                os.unlink(chart_path)
            except Exception as e:
                st.warning(f"Could not embed plasticity chart in PDF: {str(e)}")

        # --- Certification Page ---
        pdf.add_page()
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 15, safe_text("Certification"), 0, 1, 'C')
        pdf.ln(4)
        pdf.set_font("Arial", '', 11)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, safe_text(
            "This soil classification report has been reviewed and is certified as suitable "
            "for the stated project and engineering requirements."
        ))
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 11)
        pdf.cell(60, 8, safe_text("Engineer Name:"), 0, 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, safe_text(engineer_name), 'B', 1)
        pdf.ln(6)

        pdf.set_font("Arial", 'B', 11)
        pdf.cell(60, 8, safe_text("Date:"), 0, 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, safe_text(datetime.now().strftime('%Y-%m-%d')), 'B', 1)
        pdf.ln(15)

        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, safe_text("Signature / Stamp"), 0, 1)
        box_y = pdf.get_y()
        box_w, box_h = 70, 35
        if stamp_image_path and os.path.exists(stamp_image_path):
            try:
                pdf.image(stamp_image_path, x=15, y=box_y, w=box_w, h=box_h)
            except Exception:
                pdf.rect(15, box_y, box_w, box_h)
        else:
            pdf.rect(15, box_y, box_w, box_h)
        pdf.set_y(box_y + box_h + 8)

        pdf.set_font("Arial", '', 9)
        pdf.set_text_color(120, 120, 120)
        prepared_by = f"Report prepared using {APP_TITLE} by {CLIENT_NAME}."
        if FOOTER_NOTE:
            prepared_by += f" {FOOTER_NOTE}"
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, safe_text(prepared_by))
        pdf.set_text_color(0, 0, 0)

        pdf_output = pdf.output()
        if isinstance(pdf_output, (bytes, bytearray)):
            return bytes(pdf_output)
        return pdf_output.encode('latin-1', errors='replace')

    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None


# =============================================================================
# 6. UI
# =============================================================================

if LOGO_PATH and os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=180)
else:
    st.caption(
        f"⚠️ Logo not found at `{LOGO_PATH}` (checked from `{os.getcwd()}`). "
        "The PDF report's cover page will render without a logo until this is fixed."
    )
    with st.expander("🔧 Logo diagnostics (what this deployment actually sees)"):
        cwd = os.getcwd()
        st.write("**Working directory:**", f"`{cwd}`")
        try:
            root_items = sorted(os.listdir(cwd))
            st.write("**Repo root contents:**")
            st.code("\n".join(repr(item) for item in root_items))  # repr() reveals hidden spaces/case
        except Exception as e:
            st.write(f"Could not list root directory: {e}")

        assets_dir = os.path.join(cwd, "assets")
        if os.path.isdir(assets_dir):
            try:
                assets_items = sorted(os.listdir(assets_dir))
                st.write("**`assets/` folder contents:**")
                st.code("\n".join(repr(item) for item in assets_items))
            except Exception as e:
                st.write(f"Could not list assets directory: {e}")
        else:
            st.write("**No `assets/` folder found at the repo root of this deployment.**")
            st.caption(
                "If GitHub's web UI shows the folder but it's missing here, this deployment is likely "
                "running a stale checkout from before that commit. Try Manage app -> Reboot, and if "
                "that doesn't pick it up, delete and redeploy the app to force a fresh clone."
            )

st.title("🧱 USCS Soil Classification System")
st.caption(f"⚡ Powered by {CLIENT_NAME}  -  ASTM D2487")

st.markdown("**Project Name**")
project_name = st.text_input("", "Unnamed Project", key="project_name_input", label_visibility="collapsed")

st.markdown("**Client / Project Owner**")
client_name = st.text_input("", "", key="client_name_input", label_visibility="collapsed")

with st.expander("🖋️ Certification Details (for PDF Report)"):
    st.markdown("**Engineer Name**")
    engineer_name = st.text_input("", st.session_state.get('engineer_name', ''), key="engineer_name_input")
    st.session_state['engineer_name'] = engineer_name

    st.markdown("**Signature / Stamp Image (optional)**")
    stamp_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], key="stamp_upload")
    if stamp_file is not None:
        st.session_state['stamp_bytes'] = stamp_file.getvalue()
    if st.session_state.get('stamp_bytes'):
        st.image(st.session_state['stamp_bytes'], width=150, caption="Stamp preview")
    st.caption("Leave blank to print an empty signature box in the report for a physical wet stamp instead.")

tab_single, tab_batch = st.tabs(["🔬 Single Sample", "📦 Batch Processing"])

# -----------------------------------------------------------------------
# TAB 1: SINGLE SAMPLE
# -----------------------------------------------------------------------
with tab_single:
    st.subheader("Sample Identification")
    sample_id = st.text_input("Sample / Borehole ID", "Sample 1", key="single_sample_id")

    st.subheader("📊 Particle Size Distribution")
    st.caption("Enter % passing at each sieve size. Add/remove rows as needed for your gradation curve.")
    default_grad_df = pd.DataFrame({
        "Sieve Size (mm)": DEFAULT_SIEVES_MM,
        "% Passing": [100.0, 95.0, 80.0, 60.0, 40.0, 25.0]
    })
    grad_df = st.data_editor(
        default_grad_df, num_rows="dynamic", hide_index=True, use_container_width=True,
        key="single_grad_editor"
    )

    st.subheader("🔍 Atterberg Limits")
    is_np = st.checkbox("Non-Plastic (NP)", key="single_is_np")
    col1, col2 = st.columns(2)
    with col1:
        LL = st.number_input("Liquid Limit (LL)", min_value=0.0, max_value=200.0, value=30.0,
                             disabled=is_np, key="single_ll")
    with col2:
        PL = st.number_input("Plastic Limit (PL)", min_value=0.0, max_value=200.0, value=18.0,
                             disabled=is_np, key="single_pl")
    PI = 0.0 if is_np else max(0.0, LL - PL)
    if not is_np:
        st.write(f"Plasticity Index (PI) = **{PI:.1f}**")

    st.subheader("🌿 Organic Soils & Peat")
    col3, col4 = st.columns(2)
    with col3:
        is_organic = st.checkbox(
            "Organic Soil (LL oven-dried / not-dried < 0.75)", key="single_organic",
            help="Per ASTM D2487, organic soils are identified when the oven-dried LL is less than 75% of the not-oven-dried LL."
        )
    with col4:
        is_peat = st.checkbox(
            "Peat (predominantly organic, dark, fibrous, organic odor)", key="single_peat",
            help="Peat is classified as PT by visual-manual examination  -  the standard sieve/Atterberg procedure does not apply."
        )

    if st.button("🚀 Classify Sample", key="single_classify_btn"):
        sizes = grad_df["Sieve Size (mm)"].tolist()
        passing = grad_df["% Passing"].tolist()
        grad = gradation_summary(sizes, passing)
        result = classify_uscs(
            grad['pct_gravel'], grad['pct_sand'], grad['pct_fines'],
            grad['Cu'], grad['Cc'], LL, PI, is_np, is_organic, is_peat
        )

        grad_fig = create_gradation_chart(sizes, passing, label=sample_id)
        grad_png = fig_to_png_bytes(grad_fig)

        st.session_state['single_result'] = {
            "sample_id": sample_id, "result": result, "grad": grad,
            "LL": LL, "PL": PL, "PI": round(PI, 1), "is_np": is_np,
            "is_organic": is_organic, "is_peat": is_peat,
            "gradation_chart_png": grad_png,
            "plasticity_point": (sample_id, LL, PI) if not is_np and not is_peat else None,
            "sizes": sizes, "passing": passing,
        }

    if st.session_state.get('single_result'):
        r = st.session_state['single_result']
        result = r['result']

        st.markdown("---")
        st.subheader("📊 Classification Results")
        st.success(f"🎯 USCS Symbol: **{result['symbol']}**")
        st.info(f"🧱 Category: **{result['category']}**")
        if result.get('borderline'):
            st.warning("⚠️ This is a borderline/dual classification.")

        st.markdown("**Description**")
        st.write(get_description(result['symbol']))

        if result.get('notes'):
            st.markdown("**Notes**")
            for n in result['notes']:
                st.caption(f"• {n}")

        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Particle Size Distribution**")
            st.image(r['gradation_chart_png'])
        with col6:
            if r['plasticity_point']:
                st.markdown("**Casagrande Plasticity Chart**")
                pfig = create_plasticity_chart([r['plasticity_point']])
                st.pyplot(pfig)
                plt.close(pfig)
            else:
                st.info("Plasticity chart not applicable (non-plastic sample or peat).")

        st.markdown("**🤖 Engineering Interpretation**")
        st.markdown(generate_interpretation(result, r['grad'], r['LL'], r['PI'], r['is_np']).replace("\n", "\n\n"))

        st.subheader("📥 Downloads")
        export_df = pd.DataFrame({
            "Sample ID": [r['sample_id']], "USCS Symbol": [result['symbol']],
            "Category": [result['category']], "Gravel %": [r['grad']['pct_gravel']],
            "Sand %": [r['grad']['pct_sand']], "Fines %": [r['grad']['pct_fines']],
            "Cu": [r['grad']['Cu']], "Cc": [r['grad']['Cc']],
            "LL": [r['LL']], "PL": [r['PL']], "PI": [r['PI']],
            "Organic": [r['is_organic']], "Peat": [r['is_peat']],
        })
        st.download_button("📊 Download as CSV", export_df.to_csv(index=False),
                          f"uscs_{r['sample_id']}.csv", "text/csv", key="single_csv_dl")

        if st.button("📄 Generate PDF Report", key="single_pdf_btn"):
            stamp_path = None
            if st.session_state.get('stamp_bytes'):
                stamp_path = os.path.join(tempfile.gettempdir(),
                                        f"temp_stamp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
                with open(stamp_path, 'wb') as f:
                    f.write(st.session_state['stamp_bytes'])
            pdf_data = create_pdf_report([r], project_name, client_name,
                                        st.session_state.get('engineer_name', ''), stamp_path)
            if stamp_path and os.path.exists(stamp_path):
                os.unlink(stamp_path)
            if pdf_data:
                st.download_button("⬇️ Download PDF Report", data=pdf_data,
                                  file_name=f"uscs_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                  mime="application/pdf", key="single_pdf_dl")

# -----------------------------------------------------------------------
# TAB 2: BATCH PROCESSING
# -----------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Sample Upload")
    st.caption(
        "Upload a CSV with one row per sample. Download the template below to get the exact column format."
    )

    template_df = pd.DataFrame([{
        "Sample_ID": "BH-1 @ 1.5m",
        "Pass_19mm": 100, "Pass_9.5mm": 92, "Pass_4.75mm": 78,
        "Pass_2.0mm": 60, "Pass_0.425mm": 38, "Pass_0.075mm": 22,
        "LL": 32, "PL": 19, "Non_Plastic": "N", "Organic": "N", "Peat": "N"
    }])
    st.download_button("📥 Download CSV Template", template_df.to_csv(index=False),
                      "uscs_batch_template.csv", "text/csv", key="batch_template_dl")

    batch_file = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_uploader")

    if batch_file is not None:
        try:
            batch_input_df = pd.read_csv(batch_file)
            st.dataframe(batch_input_df, use_container_width=True, hide_index=True)

            if st.button("🚀 Classify All Samples", key="batch_classify_btn"):
                sieve_cols = {"Pass_19mm": 19.0, "Pass_9.5mm": 9.5, "Pass_4.75mm": 4.75,
                            "Pass_2.0mm": 2.0, "Pass_0.425mm": 0.425, "Pass_0.075mm": 0.075}
                batch_results = []
                for _, row in batch_input_df.iterrows():
                    sizes = [mm for col, mm in sieve_cols.items() if col in row and pd.notna(row[col])]
                    passing = [float(row[col]) for col, mm in sieve_cols.items() if col in row and pd.notna(row[col])]
                    is_np_b = str(row.get("Non_Plastic", "N")).strip().upper().startswith("Y")
                    is_organic_b = str(row.get("Organic", "N")).strip().upper().startswith("Y")
                    is_peat_b = str(row.get("Peat", "N")).strip().upper().startswith("Y")
                    LL_b = float(row.get("LL", 0) or 0)
                    PL_b = float(row.get("PL", 0) or 0)
                    PI_b = 0.0 if is_np_b else max(0.0, LL_b - PL_b)

                    grad_b = gradation_summary(sizes, passing) if len(sizes) >= 2 else {
                        "pct_gravel": 0, "pct_sand": 0, "pct_fines": 100, "Cu": None, "Cc": None
                    }
                    result_b = classify_uscs(
                        grad_b['pct_gravel'], grad_b['pct_sand'], grad_b['pct_fines'],
                        grad_b['Cu'], grad_b['Cc'], LL_b, PI_b, is_np_b, is_organic_b, is_peat_b
                    )
                    grad_fig_b = create_gradation_chart(sizes, passing, label=str(row.get("Sample_ID", "Sample")))
                    grad_png_b = fig_to_png_bytes(grad_fig_b)
                    plt.close(grad_fig_b)

                    batch_results.append({
                        "sample_id": str(row.get("Sample_ID", "Sample")), "result": result_b, "grad": grad_b,
                        "LL": LL_b, "PL": PL_b, "PI": round(PI_b, 1), "is_np": is_np_b,
                        "is_organic": is_organic_b, "is_peat": is_peat_b,
                        "gradation_chart_png": grad_png_b,
                        "plasticity_point": (str(row.get("Sample_ID", "Sample")), LL_b, PI_b) if not is_np_b and not is_peat_b else None,
                    })
                st.session_state['batch_results'] = batch_results

        except Exception as e:
            st.error(f"Could not read that CSV: {str(e)}")

    if st.session_state.get('batch_results'):
        results = st.session_state['batch_results']
        st.markdown("---")
        st.subheader(f"📊 Batch Results ({len(results)} samples)")

        summary_rows = []
        for r in results:
            summary_rows.append({
                "Sample ID": r['sample_id'], "USCS Symbol": r['result']['symbol'],
                "Category": r['result']['category'],
                "Gravel %": r['grad'].get('pct_gravel'), "Sand %": r['grad'].get('pct_sand'),
                "Fines %": r['grad'].get('pct_fines'), "LL": r['LL'], "PI": r['PI'],
                "Borderline": r['result'].get('borderline', False)
            })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        plasticity_points = [r["plasticity_point"] for r in results if r.get("plasticity_point")]
        if plasticity_points:
            st.markdown("**Combined Casagrande Plasticity Chart**")
            bfig = create_plasticity_chart(plasticity_points)
            st.pyplot(bfig)
            plt.close(bfig)

        st.subheader("📥 Downloads")
        st.download_button("📊 Download Batch Results as CSV", summary_df.to_csv(index=False),
                          "uscs_batch_results.csv", "text/csv", key="batch_csv_dl")

        if st.button("📄 Generate Batch PDF Report", key="batch_pdf_btn"):
            stamp_path = None
            if st.session_state.get('stamp_bytes'):
                stamp_path = os.path.join(tempfile.gettempdir(),
                                        f"temp_stamp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
                with open(stamp_path, 'wb') as f:
                    f.write(st.session_state['stamp_bytes'])
            pdf_data = create_pdf_report(results, project_name, client_name,
                                        st.session_state.get('engineer_name', ''), stamp_path)
            if stamp_path and os.path.exists(stamp_path):
                os.unlink(stamp_path)
            if pdf_data:
                st.download_button("⬇️ Download Batch PDF Report", data=pdf_data,
                                  file_name=f"uscs_batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                  mime="application/pdf", key="batch_pdf_dl")

st.markdown("---")
st.caption(f"© 2025 USCS Soil Classification System | Built by {CLIENT_NAME}")
