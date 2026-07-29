# USCS Soil Classification System

A Streamlit application for classifying soils per **ASTM D2487** (Unified Soil
Classification System), built by **Automation_hub Engineering Group Limited**.

## Features

- ✔ Particle size distribution (editable gradation curve, log-interpolated D10/D30/D60)
- ✔ Liquid Limit / Plastic Limit / Plasticity Index
- ✔ Dual symbols (e.g. GW-GM, SC-SM) for the 5–12% fines range
- ✔ Borderline cases (CL-ML hatched zone, mixed dual/borderline notes)
- ✔ Organic soil identification (OL/OH)
- ✔ Peat identification (PT, visual-manual per ASTM D2487)
- ✔ Casagrande plasticity chart with automatic point plotting
- ✔ Automatic engineering interpretation per classification
- ✔ Branded PDF report (cover page, results, charts, certification/sign-off)
- ✔ CSV export
- ✔ Batch processing (upload a CSV of multiple samples, classify all at once)

## Project Structure

```
.
├── main.py              # App entry point — UI, classification engine, PDF generation
├── branding.py          # Company name, colors, logo path, contact details
├── style.css            # Visual styling (buttons, tabs, cards) — auto-loaded if present
├── requirements.txt     # Python dependencies
├── assets/
│   └── 2.png            # Your logo (add this yourself — see below)
└── README.md
```

## Setup

1. **Clone the repo and install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your logo** (optional but recommended): place a PNG at `assets/2.png`.
   If it's missing, the app and PDF report still work — the logo slot is just
   left blank.

3. **Customize branding**: edit `branding.py` to set your company name, app
   title, brand color, and contact details (address/phone/email/website —
   leave any of these as `""` to omit them from the PDF).

4. **Run locally:**

   ```bash
   streamlit run main.py
   ```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   and set the main file path to `main.py`.
3. No secrets or extra configuration are required — this app has no external
   service dependencies (no database, no API keys).

## Using the App

### Single Sample
Enter a sample ID, edit the particle size distribution table (sieve size vs.
% passing), enter Liquid Limit / Plastic Limit (or mark Non-Plastic), flag
organic soil or peat if applicable, then classify. Results include the USCS
symbol, category, engineering interpretation, gradation chart, and Casagrande
plasticity chart — each downloadable individually or as a combined PDF report.

### Batch Processing
Download the CSV template, fill in one row per sample (sieve % passing at six
standard sizes, plus LL/PL/flags), upload it, and classify every row at once.
Results include a combined summary table, a combined plasticity chart with
every sample plotted, and one PDF report covering the whole batch.

## Engineering Notes

- The classification engine implements the ASTM D2487 flowchart, including
  the A-line/U-line plasticity chart, well-/poorly-graded gravel and sand
  (Cu/Cc), and dual/borderline symbol logic.
- A few genuinely ambiguous ASTM edge cases (e.g. 5–12% fines *and* PI in the
  4–7 hatched zone at the same time) don't have one canonical answer even in
  the standard itself — the app flags these explicitly with a note rather
  than silently picking one, so review is expected on those samples.
- This tool supports engineering judgment; it does not replace a licensed
  geotechnical engineer's review and sign-off, which is why the PDF report
  includes a certification/signature section.

## License / Ownership

© Automation_hub Engineering Group Limited. Internal engineering tool.
