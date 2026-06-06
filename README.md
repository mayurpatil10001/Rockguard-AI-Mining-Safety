<div align="center">

# ⛏️ RockGuard AI — Mining Safety System

### Real-time rockfall prediction with continuous online learning & geotechnical monitoring

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Folium](https://img.shields.io/badge/Folium-GIS_Maps-77B829?style=for-the-badge)](https://python-visualization.github.io/folium/)
[![SQLite](https://img.shields.io/badge/SQLite-Real--time_DB-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **Mining accidents kill 15,000+ workers globally every year.**  
> **RockGuard AI detects the risk before it becomes a disaster.**

<br/>

![RockGuard Banner](https://via.placeholder.com/1000x400/1e3c72/ffffff?text=RockGuard+AI+%7C+Real-Time+Mining+Safety+System)

</div>

---

## 📌 Table of Contents

- [📖 Overview](#-overview)
- [✨ Features](#-features)
- [🤖 AI Architecture](#-ai-architecture)
- [📡 Sensor Monitoring](#-sensor-monitoring)
- [🗺️ Risk Mapping](#️-risk-mapping)
- [🔄 Online Learning Pipeline](#-online-learning-pipeline)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📊 Dashboard Modules](#-dashboard-modules)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 📖 Overview

**RockGuard AI** is a production-grade Streamlit platform for real-time rockfall risk prediction at mining sites across India. It combines **Gradient Boosting ML** with **continuous online learning**, geotechnical sensor data streams, interactive Folium risk maps, and an automated alert system — retraining itself every 5 minutes with fresh data from active mining sites.

The system monitors 14 parameters across weather, geotechnical, and blast domains, classifies risk into Low/Medium/High with confidence scores, and immediately broadcasts critical alerts with evacuation protocols.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Adaptive AI Model** | Gradient Boosting Classifier with online learning — auto-retrains every 5 minutes |
| 📡 **14-Parameter Monitoring** | Weather, geotechnical & blast parameters tracked simultaneously |
| 🎯 **Confidence Scoring** | Every prediction includes a confidence interval from decision boundary distance |
| 🗺️ **Interactive Risk Map** | Folium map with color-coded site markers and popup sensor readings |
| 🔄 **Online Learning** | Model continuously improves with new site data — version-controlled |
| 🚨 **Tiered Alert System** | 3-level alerts (Low/Medium/High) with specific action protocols |
| 📊 **72-Hour History** | Multi-parameter time series with hourly temporal analysis |
| ⚙️ **System Diagnostics** | CPU, memory, model accuracy, sensor connectivity & training health |
| 📱 **SMS + Email Alerts** | Configurable notification channels with test capability |
| 🏭 **4 Indian Mining Sites** | Jharia, Bailadila, Kolar Gold Fields, Rajpura-Dariba pre-loaded |

---

## 🤖 AI Architecture

### Model Pipeline

```
Sensor Input (14 features)
        │
        ├── Weather: temperature, humidity, rainfall, wind_speed, pressure
        ├── Geotechnical: piezometer, vibration_ppv, crack_displacement,
        │                 tilt_angle, groundwater_level
        └── Blast: blast_intensity, distance_to_slope, explosive_weight, blast_frequency
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              AdaptiveRockfallPredictor                       │
│                                                             │
│  GradientBoostingClassifier                                 │
│  • n_estimators=150, learning_rate=0.1, max_depth=6        │
│  • predict_proba() → [P(Low), P(Med), P(High)]             │
│  • decision_function() → confidence score                  │
│                                                             │
│  Online Learning Loop (every 5 min):                        │
│  ┌──────────────────────────────────────────┐              │
│  │ 1. Collect 20 samples/site × 4 sites     │              │
│  │ 2. Add to deque buffer (maxlen=1000)      │              │
│  │ 3. Combine with 400 historical samples   │              │
│  │ 4. Retrain on combined dataset           │              │
│  │ 5. Evaluate accuracy → version++         │              │
│  └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Risk Level + Probability + Confidence → Alert System
```

### Risk Classification

| Risk Score | Level | Color | Action |
|---|---|---|---|
| ≥ 10 | 🔴 **High** | Dark Red | Stop blasting, evacuate, alert emergency |
| 5–9 | 🟡 **Medium** | Orange | Reduce blast intensity 30%, brief safety team |
| 0–4 | 🟢 **Low** | Green | Normal operations, standard monitoring |

### Risk Score Calculation

```python
# Weather-based risks
if rainfall_24h > 50: risk_score += 3
if temperature > 35 or < 5: risk_score += 1
if wind_speed > 20: risk_score += 1

# Geotechnical risks  
if vibration_ppv > 5: risk_score += 4   ← Highest weight
if crack_displacement > 2: risk_score += 3
if piezometer_pressure > 20: risk_score += 2
if abs(tilt_angle) > 2: risk_score += 2

# Blast-induced risks
if blast_intensity > 5 and distance < 100: risk_score += 3
if explosive_weight > 50 and distance < 200: risk_score += 2

# Apply site-specific multiplier (1.0 – 1.5)
risk_score *= site_risk_base
```

### Temporal Data Patterns

The system generates **realistic sensor data** with:
- **Seasonal temperature variation** using sinusoidal day-of-year model
- **Monsoon-influenced rainfall** with exponential distribution
- **Memory effects** — current readings influenced by previous state (70% memory factor)
- **3% label noise** to simulate real-world sensor uncertainty

---

## 📡 Sensor Monitoring

### 10 Live Sensor Parameters

| Parameter | Threshold | Alert Condition |
|---|---|---|
| Temperature | 35°C | Above threshold |
| Humidity | 90% | Above threshold |
| Rainfall (24h) | 50mm | Above threshold |
| Wind Speed | 25km/h | Above threshold |
| Pressure | 980hPa | Below threshold |
| Piezometer Pressure | 20kPa | Above threshold |
| Vibration PPV | 5.0mm/s | Above threshold |
| Crack Displacement | 2.0mm | Above threshold |
| Tilt Angle | ±3° | Outside range |
| Groundwater Level | 15m | Above threshold |

### Live Sensor Table

Every reading is displayed with:
- Current value in engineering units
- Threshold value for comparison
- Status indicator (🟢 Normal / 🔴 Alert)

---

## 🗺️ Risk Mapping

Interactive Folium map with:

```
Site Marker Colors:
  🟢 Dark Green   → Low risk, high confidence (>70%)
  🟢 Green        → Low risk, medium confidence
  🟠 Dark Orange  → Medium risk, high confidence
  🟠 Orange       → Medium risk, medium confidence
  🔴 Dark Red     → High risk, high confidence
  🔴 Red          → High risk, medium confidence

High-risk sites get an additional pulsing outer ring
```

**Pre-loaded Indian Mining Sites:**

| Site | Type | Location | Risk Multiplier |
|---|---|---|---|
| Jharia Coalfield | Coal | Jharkhand | 1.2x |
| Bailadila Iron Ore | Iron | Chhattisgarh | 1.0x |
| Kolar Gold Fields | Gold | Karnataka | 1.5x |
| Rajpura-Dariba | Lead-Zinc | Rajasthan | 1.1x |

---

## 🔄 Online Learning Pipeline

```
Every 5 minutes:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  STEP 1: Data Collection                                │
│  • 20 samples × 4 sites = 80 new data points           │
│  • Temporal patterns + site-specific stress factors     │
│  • Realistic label generation with 3% noise            │
│                                                         │
│  STEP 2: Buffer Management                              │
│  • Add to deque(maxlen=1000)                           │
│  • Auto-trim: keep last 500 if overflow                │
│                                                         │
│  STEP 3: Retraining (if buffer ≥ 50 samples)          │
│  • Combine: 400 historical + buffer samples            │
│  • 80/20 train-test split                              │
│  • GradientBoostingClassifier.fit()                    │
│                                                         │
│  STEP 4: Version Control                                │
│  • Evaluate accuracy on test set                        │
│  • Increment model version                              │
│  • Store performance metrics                            │
│                                                         │
│  STEP 5: Broadcast                                      │
│  • Update dashboard metrics                             │
│  • Send alerts if High risk detected                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly Express, Plotly Graph Objects, Plotly Subplots |
| **ML/AI** | Scikit-learn (GradientBoosting, RandomForest, IsolationForest) |
| **GIS Maps** | Folium, streamlit-folium |
| **Database** | SQLite (sensor_data, model_performance, training_buffer) |
| **Data** | Pandas, NumPy |
| **Threading** | Python threading, collections.deque |

</div>

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
pip
```

### 1. Clone the Repository

```bash
git clone https://github.com/mayurpatil10001/rockguard-ai-mining-safety.git
cd rockguard-ai-mining-safety
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy scikit-learn plotly folium \
            streamlit-folium requests joblib pickle5 sqlite3
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

The system will:
1. Automatically perform **initial model training** (800 samples)
2. Begin **collecting real-time sensor data** from all 4 sites
3. Schedule **retraining every 5 minutes** automatically

---

## 📊 Dashboard Modules

### Tab 1: 🎯 Live Monitoring
- Real-time risk level with confidence score
- 5 live metric cards (temperature, rainfall, vibration, crack movement)
- Risk probability bar chart with confidence overlay
- Color-coded alert panel with specific action protocols
- Complete 10-parameter sensor readings table with threshold comparison

### Tab 2: 📊 Risk Analysis
- Interactive Folium risk map — all 4 mining sites
- Feature importance bar chart (live model)
- Model performance evolution line chart
- 8×8 parameter correlation matrix heatmap

### Tab 3: 📈 Historical Trends
- 72-hour 4-panel time series (weather / stability / pressure / risk)
- Daily pattern analysis — average readings by hour
- Statistical summary with risk event counts

### Tab 4: 🚨 Alert Center
- Live alert feed table (last 10 predictions)
- Configurable threshold sliders (6 parameters)
- SMS + email notification configuration
- Test alert system button

### Tab 5: ⚙️ System Status
- Training session history table with accuracy per version
- Training buffer risk distribution analysis
- Model accuracy evolution chart
- Resource utilization (CPU, memory, storage, network)
- System health indicator with performance index

---

## 📁 Project Structure

```
rockguard-ai-mining-safety/
│
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── rockfall_realtime.db           # SQLite database (auto-created)
│
├── models/                        # (recommended refactor)
│   ├── adaptive_predictor.py      # AdaptiveRockfallPredictor class
│   └── online_learning.py         # Online training pipeline
│
├── data/                          # (recommended refactor)
│   ├── realtime_stream.py         # RealTimeDataStream class
│   └── mining_sites.py           # Site coordinates & risk configs
│
├── services/
│   ├── database_manager.py        # EnhancedDatabaseManager class
│   └── alert_service.py           # SMS/email alert system
│
├── utils/
│   └── risk_map.py                # Folium map generator
│
└── README.md
```

---

## 🏭 Real-World Impact

RockGuard AI addresses India's critical mining safety challenges:

- **Coal India Limited** — world's largest coal producer, 300K+ workers
- **NMDC, SAIL, Tata Steel** — major iron ore mining operations
- **DGMS (Directorate General of Mines Safety)** — regulatory compliance
- **CMPDIL** — Central Mine Planning & Design Institute

The system's online learning capability means it **gets smarter over time** — adapting to site-specific geology, seasonal patterns, and operational changes without manual retraining.

---

## 🤝 Contributing

Contributions welcome!

```bash
git checkout -b feature/your-feature
git commit -m "feat: add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

**Ideas for contributions:**
- Integration with actual IoT sensors (Raspberry Pi GPIO)
- DGMS compliance report generation
- WhatsApp alert via Twilio API
- LSTM model for sequence-based prediction
- Microseismic event detection module
- Integration with OpenWeatherMap API for live weather

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Safety First. Every Mine. Every Day.**

Built with ❤️ by [Mayur Patil](https://github.com/mayurpatil10001)

⭐ Star this repo if it could save a miner's life!

</div>
