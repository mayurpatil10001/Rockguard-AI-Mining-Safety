<div align="center">

# 🚜 Digital Farm Management AI

### AI-powered livestock health & antimicrobial resistance management for Indian farmers

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Languages](https://img.shields.io/badge/Languages-8_Indian-00C851?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **75% of India's livestock farmers have no digital health records.**  
> **Antimicrobial resistance kills 700,000 people globally every year.**  
> This platform fights both — in your language.

<br/>

![Farm Management Banner](https://via.placeholder.com/1000x400/1b5e20/a5d6a7?text=Digital+Farm+Management+AI+%7C+Krishi+Swasthya+Platform)

</div>

---

## 📌 Table of Contents

- [📖 Overview](#-overview)
- [✨ Features](#-features)
- [🌐 Multi-language Support](#-multi-language-support)
- [🤖 AI Models](#-ai-models)
- [📊 Platform Modules](#-platform-modules)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [👥 User Roles](#-user-roles)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 📖 Overview

**Digital Farm Management AI** is a comprehensive Streamlit-based platform for managing livestock health records, tracking antimicrobial treatments, predicting resistance risks, and ensuring regulatory compliance — built specifically for **Indian farmers, veterinarians, government officials, and researchers**.

The system uses an ensemble of ML models (Random Forest, Gradient Boosting, MLP Neural Network, Isolation Forest) to predict antimicrobial resistance risk, recommend optimal dosages, detect anomalous treatment patterns, and forecast future treatment needs — all while enforcing MRL (Maximum Residue Limit) compliance to ensure food safety.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **AI Resistance Prediction** | Random Forest model predicts antimicrobial resistance probability |
| 💊 **Optimal Dosage Recommendation** | Gradient Boosting recommends weight-based dosages |
| 🔮 **Treatment Efficacy Forecast** | MLP Neural Network predicts treatment success rate |
| 🚨 **Anomaly Detection** | Isolation Forest detects abnormal treatment patterns |
| 📊 **MRL Compliance Monitor** | Real-time Maximum Residue Limit tracking with score |
| 📦 **Drug Inventory Management** | Stock levels, expiry alerts & low-stock warnings |
| 💰 **Financial Tracking** | Treatment costs, monthly trends & cost-per-animal |
| 🌐 **8 Indian Languages** | English, Hindi, Bengali, Telugu, Tamil, Gujarati, Malayalam, Punjabi |
| 📈 **Plotly Analytics Dashboard** | Interactive species distribution, usage trends & risk histograms |
| 📤 **CSV Export** | Download filtered treatment records |
| 🗓️ **Treatment Forecasting** | 30-day treatment volume prediction with seasonal factors |
| 🔐 **Secure Auth** | SHA-256 hashed passwords with role-based access |

---

## 🌐 Multi-language Support

The platform is fully localised for **8 major Indian languages** — every label, button, alert, and navigation item adapts to the selected language:

| Language | Script | Coverage |
|---|---|---|
| 🇬🇧 English | Latin | 100% (primary) |
| 🇮🇳 Hindi (हिंदी) | Devanagari | 100% |
| 🇧🇩 Bengali (বাংলা) | Bengali | Core UI |
| 🌊 Telugu (తెలుగు) | Telugu | Core UI |
| 🌺 Tamil (தமிழ்) | Tamil | Core UI |
| 💎 Gujarati (ગુજરાતી) | Gujarati | Core UI |
| 🌴 Malayalam (മലയാളം) | Malayalam | Core UI |
| 🌾 Punjabi (ਪੰਜਾਬੀ) | Gurmukhi | Core UI |

---

## 🤖 AI Models

### Model Architecture

```
Treatment Input
    │
    ├── LabelEncoder (species, drug_name, reason, breed)
    ├── StandardScaler (dosage, age, weight, withdrawal_period)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  AdvancedAIModelManager                 │
│                                                         │
│  RandomForestRegressor    → Resistance Risk Score       │
│  GradientBoostingRegressor → Optimal Dosage            │
│  MLPRegressor             → Treatment Efficacy         │
│  IsolationForest          → Anomaly Detection          │
│  KMeans                   → Farm Pattern Clustering    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Real-time predictions displayed during treatment entry
```

### Resistance Risk Model

| Input Feature | Description |
|---|---|
| `species` | Animal species (Cattle/Buffalo/Goat/Sheep/Pig/Poultry) |
| `drug_name` | Antimicrobial drug selected |
| `reason` | Treatment reason (30 categories) |
| `age` | Animal age in months |
| `weight` | Animal weight in kg |
| `dosage` | Dosage in mg/kg |
| `withdrawal_period` | Drug-specific withdrawal days |

**Output:** Risk score 0.0–1.0 displayed at treatment entry

### Compliance Scoring (5 weighted criteria)

| Criterion | Weight | Description |
|---|---|---|
| MRL Compliance | 40% | Animals sold before safe date |
| Dosage Appropriateness | 25% | Deviation from weight-based expected dosage |
| Resistance Risk | 20% | Treatments with AI risk score >0.7 |
| Record Completeness | 10% | Missing required fields |
| Environmental Impact | 5% | High-impact drug usage |

### Antimicrobials Database (15 drugs)

| Drug | Category | Withdrawal (days) | Resistance Risk |
|---|---|---|---|
| Ceftiofur | Cephalosporin | 4 | 🟢 Low (0.15) |
| Amoxicillin | Beta-lactam | 7 | 🟢 Low (0.30) |
| Enrofloxacin | Fluoroquinolone | 10 | 🟡 Med (0.35) |
| Oxytetracycline | Tetracycline | 14 | 🟡 Med (0.45) |
| Streptomycin | Aminoglycoside | 21 | 🔴 High (0.55) |
| Tilmicosin | Macrolide | 28 | 🟢 Low (0.25) |

---

## 📊 Platform Modules

### 1. 📊 Dashboard
- Total animals, active treatments, pending alerts, total cost
- Species-wise treatment distribution (pie chart)
- Treatment usage trends over time (line chart)
- Recent treatment records table

### 2. 💊 Add Treatment Record
- Full animal profile (species, breed, age, weight)
- Drug selection with auto-populated withdrawal period
- Real-time AI predictions (risk, dosage, efficacy)
- Automatic safe sale date calculation
- Inventory auto-deduction on submission

### 3. 📋 View Records
- Filter by species, drug, date range
- Full-text search across all fields
- CSV export of filtered results

### 4. 🤖 AI Analytics
- Resistance risk distribution histogram
- Treatment efficacy distribution
- Isolation Forest anomaly detection with flagged records
- Model training status & performance

### 5. 🚨 Smart Alerts
- MRL violation alerts (animals still in withdrawal)
- Low inventory stock warnings
- High resistance risk treatment flags
- Severity-coded color display (High/Medium/Low)

### 6. 🔮 AI Predictions
- 30-day treatment volume forecast (seasonal adjustment)
- Resistance risk trend over time
- Moving average-based forecasting

### 7. ✅ Compliance Monitor
- Compliance score gauge (0–100%)
- Violation breakdown with counts
- Actionable recommendations per violation type

### 8. 📦 Drug Inventory
- Add/track drug batches with expiry dates
- Low stock alerts (below minimum level)
- Items expiring within 30 days
- Auto-update on treatment submission

### 9. 💰 Financial Tracking
- Total, monthly, per-animal, per-treatment costs
- Monthly cost bar chart
- Drug-wise cost distribution pie chart
- Species × drug cost breakdown table

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly Express, Plotly Graph Objects |
| **ML Models** | Scikit-learn (RF, GBR, MLP, IsolationForest, KMeans) |
| **Database** | SQLite (via Python sqlite3) |
| **Data** | Pandas, NumPy |
| **Auth** | SHA-256 (hashlib) |
| **Localisation** | Custom LANGUAGES dict (8 languages) |

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
git clone https://github.com/mayurpatil10001/digital-farm-management-ai.git
cd digital-farm-management-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy scikit-learn plotly sqlite3 \
            matplotlib seaborn uuid
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

### 4. Create Demo Data

Click **"🧪 Create Demo Data"** in the sidebar to instantly populate the app with:
- Demo user: `username: demo`, `password: demo123`
- 10 sample treatment records across multiple species and drugs

---

## 👥 User Roles

| Role | Description | Access |
|---|---|---|
| 🌾 **Farmer** | Livestock owner managing own herd | Own farm records |
| 🩺 **Veterinarian** | Licensed vet prescribing treatments | Treatment + compliance |
| 🏛️ **Government Official** | Regulatory oversight | Compliance + analytics |
| 🔬 **Researcher** | Academic/research analysis | Full analytics |
| 🤝 **Cooperative Member** | Farmer cooperative participant | Own farm records |

All 28 Indian states with districts are pre-loaded for registration.

---

## 📁 Project Structure

```
digital-farm-management-ai/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── enhanced_farm_management.db   # SQLite database (auto-created)
│
├── modules/                      # (recommended refactor)
│   ├── ai_models.py              # AdvancedAIModelManager
│   ├── database.py               # EnhancedDatabaseManager
│   ├── compliance.py             # calculate_compliance_score
│   ├── forecasting.py            # generate_treatment_forecast
│   └── languages.py             # LANGUAGES dict
│
├── data/
│   ├── antimicrobials_db.py      # 15 antimicrobials with properties
│   ├── species_breeds.py         # 6 species × Indian breeds
│   └── states_districts.py      # All 28 states + districts
│
└── README.md
```

---

## 🌍 Impact & SDG Alignment

This platform directly addresses:

- **SDG 2** (Zero Hunger) — Better livestock health → better food production
- **SDG 3** (Good Health) — Antimicrobial resistance is a global health emergency
- **SDG 12** (Responsible Consumption) — MRL compliance ensures safe food
- **SDG 15** (Life on Land) — Environmental impact scoring for drug use

**India-specific impact:**
- Covers all **28 states** and major districts
- Supports **6 livestock species** critical to Indian agriculture
- Includes **Indian breeds** (Gir, Sahiwal, Murrah, Jamunapari)
- Available in **8 Indian languages** including Hindi, Bengali, Telugu, Tamil

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
- FSSAI & AGMARK compliance integration
- Geospatial disease heatmaps by district
- WhatsApp/SMS alert integration via Twilio
- Voice input in regional languages
- Offline mode with local SQLite sync
- Integration with eNAM (National Agriculture Market)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**किसान समृद्ध तो देश समृद्ध | Prosperous Farmer, Prosperous Nation**

Built with ❤️ by [Mayur Patil](https://github.com/mayurpatil10001)

⭐ Star this repo if it helped Indian farmers!

</div>
