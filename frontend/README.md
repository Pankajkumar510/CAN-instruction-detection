# TinyCNNCANNet — Research Showcase Frontend

> A premium, interactive research paper showcase frontend for **Lightweight CNN Architectures for CAN Bus Intrusion Detection**, featuring the proposed **TinyCNNCANNet** (~13K parameters) and the **SynCAN2025** benchmark dataset.

![TinyCNNCANNet Banner](https://img.shields.io/badge/IEEE-Research%202025-00f5ff?style=flat-square&labelColor=0a0e1a)
![Parameters](https://img.shields.io/badge/Parameters-~13K-7c3aed?style=flat-square&labelColor=0a0e1a)
![Datasets](https://img.shields.io/badge/Datasets-4%20Evaluated-10b981?style=flat-square&labelColor=0a0e1a)
![Detection Rate](https://img.shields.io/badge/Detection%20Rate-99%25%2B-ef4444?style=flat-square&labelColor=0a0e1a)

---

## 📌 Overview

This frontend serves as a **research paper showcase** for evaluators, reviewers, and collaborators. It presents the key contributions, model architectures, datasets, experimental results, and an interactive live demo — all in a single-page, visually rich web application.

### Research Contributions Highlighted

1. **Three Lightweight CNN Architectures** — TinyCNNCANNet (~13K params), MicroCNNCANNet (~5.5K), SlimCNNCANNet (~28K)
2. **SynCAN2025 Dataset** — A new synthetic CAN intrusion detection benchmark with realistic simulation and controlled attack injection
3. **Extensive Multi-Dataset Evaluation** — Experiments on 4 diverse datasets (OTIDS, ROAD, SynCAN2021, SynCAN2025)

---

## 🗂️ File Structure

```
frontend/
├── index.html      # Main single-page HTML structure
├── style.css       # Full dark-theme CSS with animations
├── app.js          # JavaScript — particles, tabs, demo classifier
└── README.md       # This file
```

---

## 🚀 Running Locally

No build tools or dependencies required. This is pure HTML/CSS/JavaScript.

### Option 1 — Python HTTP Server (recommended)

```powershell
# From the project root
python -m http.server 7823 --directory frontend
```

Then open: **http://localhost:7823**

### Option 2 — Open directly in browser

Double-click `frontend/index.html` to open it directly.  
> ⚠️ Some browsers may restrict local file access. Use Option 1 for best results.

### Option 3 — VS Code Live Server

Install the [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) extension, right-click `index.html` → **Open with Live Server**.

---

## 🎨 Sections

| # | Section | Description |
|---|---------|-------------|
| 01 | **Hero** | Animated particle canvas, live CAN bus ECU network diagram, key stats with counter animation |
| 02 | **Abstract** | Threat landscape, research contributions, three key contribution cards |
| 03 | **Architecture** | Tabbed viewer for all 3 CNN model variants with layer diagrams and specs; CAN frame feature pipeline |
| 04 | **Dataset** | Cards for OTIDS, SynCAN2025 (new), ROAD, and SynCAN2021 |
| 05 | **Results** | Animated bar chart (model comparison), metric cards, interactive confusion matrix heatmap |
| 06 | **Live Demo** | CAN frame input form, quick scenarios, real-time CNN classifier simulation, monitor log |
| 07 | **Footer** | Navigation and resource links |

---

## ✨ Features

- **Dark futuristic theme** — Cyan (`#00f5ff`) + Purple (`#7c3aed`) accent palette on deep navy backgrounds
- **Animated particle canvas** — Floating network nodes with connecting lines in the hero
- **Live CAN bus ECU SVG diagram** — Animated node ring with dashed data flow lines
- **Scroll-reveal animations** — Cards animate in as the user scrolls
- **Animated stat counters** — Numbers count up on first viewport entry
- **Tabbed architecture viewer** — Switch between TinyCNNCANNet / MicroCNNCANNet / SlimCNNCANNet
- **Animated bar chart** — Bars animate in on scroll with model macro-F1 comparison
- **Interactive confusion matrix** — Color-coded 5×5 heatmap for OTIDS results
- **Live demo classifier** — Simulate any CAN frame; choose from 4 quick attack scenarios
- **Terminal-style monitor log** — Timestamped log of every classified frame
- **Fully responsive** — Works on desktop, tablet, and mobile
- **No external JS dependencies** — Pure Vanilla JS (no React, Vue, jQuery)

---

## 🖥️ Live Demo — How It Works

The **Live Demo** section simulates TinyCNNCANNet inference on a CAN bus frame:

1. **Choose a quick scenario** (Normal / DoS Attack / Fuzzy / RPM Spoof) or enter custom values
2. Fill in the CAN frame fields: Timestamp, CAN ID (hex), DLC, and Data bytes D0–D7
3. Click **Run Classification**
4. The model simulates inference using lightweight heuristics matching real attack patterns:
   - All-`0xFF` bytes + high frequency → **DoS Attack (99.3%)**
   - High byte entropy + mid-range sum → **Fuzzy Attack**
   - ID `0x03xx` + `0xFFFF` header bytes → **RPM Spoof**
   - Otherwise → **Normal**
5. Results show: predicted label, confidence bar, per-class probabilities, and inference time
6. Each classification is logged to the **CAN Bus Monitor Log**

---

## 📊 Model Performance (OTIDS Dataset)

| Model | Parameters | Macro F1 | Inference |
|-------|-----------|----------|-----------|
| TinyCNNCANNet | ~13,000 | **0.987** | < 0.5 ms |
| SlimCNNCANNet | ~28,000 | 0.975 | < 0.8 ms |
| MicroCNNCANNet | ~5,500 | 0.962 | < 0.2 ms |
| XGBoost (baseline) | — | 0.749 | — |
| LightGBM (baseline) | — | 0.739 | — |
| Random Forest (baseline) | — | 0.743 | — |

---

## 🎯 Attack Classes

| Label | Description |
|-------|------------|
| `0` Normal | Regular ECU communication |
| `1` DoS | Denial-of-Service flood attack |
| `2` Fuzzy | Random frame injection |
| `3` Gear | Gear spoofing / RPM-linked |
| `4` RPM | Engine RPM value spoofing |

---

## 🔧 Customization

### Update Results / Metrics
Edit the bar chart values and confusion matrix numbers in `index.html` (search for `<!-- Row 0: Normal -->` etc.).

### Add a Real Model Backend
Replace the `simulateInference()` function in `app.js` with a `fetch()` call to your Python prediction API:

```javascript
async function simulateInference(canId, timestamp, dlc, bytes) {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canId, timestamp, dlc, bytes })
  });
  return response.json(); // { label, isAttack, conf, probs }
}
```

### Connect to `predict.py`
Wrap `predict.py` with a Flask or FastAPI server and POST to it from the demo form.

---

## 🌐 Deployment

### GitHub Pages
```bash
# From the repo root
git subtree push --prefix frontend origin gh-pages
```

### Netlify / Vercel
Drag and drop the `frontend/` folder into [Netlify Drop](https://app.netlify.com/drop) for instant hosting.

---

## 📚 Related Files

| File | Description |
|------|-------------|
| `../1.py` | Main training script (RF, XGBoost, LightGBM, MLP) |
| `../predict.py` | CLI inference script using saved model artifact |
| `../artifacts/can_ids_model.pkl` | Best trained model (XGBoost) |
| `../DoS_dataset.csv` | DoS attack CAN frames |
| `../Fuzzy_dataset.csv` | Fuzzy attack CAN frames |
| `../gear_dataset.csv` | Gear spoofing CAN frames |
| `../RPM_dataset.csv` | RPM spoofing CAN frames |
| `../normal_run_data.txt` | Benign CAN bus traffic |

---

## 📄 Citation

If you use this work or the SynCAN2025 dataset in your research, please cite:

```bibtex
@inproceedings{tinycannet2025,
  title     = {Lightweight CNN Architectures for Real-Time CAN Bus Intrusion Detection},
  booktitle = {IEEE Conference on Vehicular Technology},
  year      = {2025},
  note      = {TinyCNNCANNet, MicroCNNCANNet, SlimCNNCANNet; SynCAN2025 Dataset}
}
```

---

## 📝 License

This frontend is part of an academic research internship project.  
© 2025 CAN-IDS Research Group · All rights reserved.
