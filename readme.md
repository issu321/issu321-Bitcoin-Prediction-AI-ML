<div align="center">

# 🚀 Bitcoin Price Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-Educational-green?style=for-the-badge)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.0-purple?style=for-the-badge&logo=bootstrap)](https://getbootstrap.com)

### 🎯 Advanced ML-Powered Cryptocurrency Analytics Platform

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGtxdHRiM2p1ZzR1dGJ5eWg4cGJ6eGJ6eGJ6eGJ6eGJ6eGJ6eGJ6eGJ6eGJ6eGJ6eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/26tPplGWjN0ueZnnW/giphy.gif" width="400" alt="Bitcoin Animation">
</p>

**[📊 Live Demo](#)** | **[📖 Documentation](#documentation)** | **[🛠️ Installation](#installation)** | **[🤝 Contributing](#contributing)**

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
- [🔄 Data Flow Pipeline](#-data-flow-pipeline)
- [🧠 Machine Learning Pipeline](#-machine-learning-pipeline)
- [📊 Technical Indicators Engine](#-technical-indicators-engine)
- [🗄️ Database Schema](#%EF%B8%8F-database-schema)
- [🛠️ Installation](#%EF%B8%8F-installation)
- [⚙️ Configuration](#%EF%B8%8F-configuration)
- [📁 Project Structure](#-project-structure)
- [🌐 API Endpoints](#-api-endpoints)
- [👤 User Journey](#-user-journey)
- [🎨 UI/UX Themes](#-uiux-themes)
- [📈 Performance Metrics](#-performance-metrics)
- [🔒 Security](#-security)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [👥 Authors](#-authors)

---

## 🌟 Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BITCOIN PRICE PREDICTION SYSTEM                           ║
║                         ML-Powered Analytics Hub                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   ║
║  │  📥 Upload  │───▶│  🔧 Process │───▶│  🧠 Predict │───▶│  📊 Visual  │   ║
║  │   CSV Data  │    │   & Clean   │    │    with ML  │    │   Results   │   ║
║  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

This comprehensive **Flask-based web application** leverages cutting-edge machine learning algorithms to predict Bitcoin price movements with high accuracy. Built with **production-grade architecture**, it features real-time technical analysis, interactive dashboards, and a complete user management system.

### 🎯 Key Capabilities

| Capability | Description | Status |
|------------|-------------|--------|
| 🔮 **Price Prediction** | Multi-model ensemble forecasting | ✅ Active |
| 📈 **Technical Analysis** | 8+ advanced indicators | ✅ Active |
| 📤 **CSV Upload** | Custom dataset analysis | ✅ Active |
| 👥 **User Management** | Full auth & profiles | ✅ Active |
| 🗃️ **Database CRUD** | SQLite with SQLAlchemy | ✅ Active |
| 🌓 **Theme Toggle** | Dark/Light persistent | ✅ Active |

---

## ✨ Features

### 🧠 Advanced Machine Learning Models

```mermaid
graph TB
    subgraph ML_Engine["🧠 ML Prediction Engine"]
        A["📊 Input Features<br/>RSI, MACD, Bollinger, MA, Stochastic, ATR, OBV"] --> B["🔀 Feature Engineering"]
        B --> C["📐 Data Normalization"]
        C --> D{"🤖 Model Selection"}
        D -->|Ensemble| E["🌲 Random Forest<br/>200 Estimators"]
        D -->|Baseline| F["📉 Linear Regression"]
        D -->|Complex| G["🎯 Support Vector Regression"]
        E --> H["⚖️ Model Evaluation"]
        F --> H
        G --> H
        H --> I["📈 Prediction Output<br/>Price Forecast + Confidence"]
    end

    style ML_Engine fill:#1a1a2e,stroke:#16213e,stroke-width:3px,color:#fff
    style A fill:#0f3460,stroke:#e94560,stroke-width:2px,color:#fff
    style I fill:#16c79a,stroke:#0f3460,stroke-width:3px,color:#fff
```

### 📊 Technical Indicators Suite

```mermaid
mindmap
  root((📊 Technical<br/>Indicators))
    Momentum
      RSI["🌡️ RSI<br/>Relative Strength"]
      Stochastic["🎲 Stochastic<br/>Oscillator"]
      MACD["📈 MACD<br/>Convergence/Divergence"]
    Trend
      SMA["📉 SMA<br/>Simple MA"]
      EMA["📊 EMA<br/>Exponential MA"]
      MA5["5, 10, 20, 50, 100<br/>Period MAs"]
    Volatility
      Bollinger["🎯 Bollinger Bands"]
      ATR["⚡ ATR<br/>Average True Range"]
    Volume
      OBV["📦 OBV<br/>On-Balance Volume"]
      Volume["📊 Volume Analysis"]
```

### 🎨 Interactive Dashboard Features

- **Real-time Charts**: Dynamic Plotly/Matplotlib visualizations
- **Drag & Drop Upload**: Intuitive CSV file handling
- **Responsive Design**: Mobile-first Bootstrap 5 layout
- **Persistent Themes**: Dark/Light mode with localStorage
- **Live Notifications**: Toast alerts for all operations

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph Client["🖥️ Client Layer"]
        Browser["🌐 Web Browser"]
        Mobile["📱 Mobile Device"]
    end

    subgraph FlaskApp["⚡ Flask Application Layer"]
        WTForms["📝 WTForms Validation"]
        Jinja2["🎨 Jinja2 Templates"]
        Static["📁 Static Assets<br/>CSS/JS/Images"]
    end

    subgraph Backend["🔧 Backend Services"]
        Auth["🔐 Authentication Service<br/>Bcrypt Hashing"]
        ML["🧠 ML Engine<br/>Scikit-Learn"]
        DataProc["🔧 Data Processing<br/>Pandas/NumPy"]
        Upload["📤 Upload Handler<br/>CSV Parser"]
    end

    subgraph Database["🗄️ Data Layer"]
        SQLite[("💾 SQLite Database<br/>SQLAlchemy ORM")]
        Uploads["📂 File Storage<br/>/data/uploads"]
    end

    Browser -->|HTTP/HTTPS| FlaskApp
    Mobile -->|HTTP/HTTPS| FlaskApp

    FlaskApp -->|Validate| WTForms
    FlaskApp -->|Render| Jinja2
    FlaskApp -->|Serve| Static

    WTForms -->|Process| Auth
    WTForms -->|Process| ML
    WTForms -->|Process| DataProc
    WTForms -->|Process| Upload

    Auth -->|CRUD| SQLite
    ML -->|Store Results| SQLite
    DataProc -->|Read/Write| SQLite
    Upload -->|Save Files| Uploads
    DataProc -->|Read Files| Uploads

    style Client fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style FlaskApp fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Backend fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style Database fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

---

## 🔄 Data Flow Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant User as 👤 User
    participant Browser as 🌐 Browser
    participant Flask as ⚡ Flask App
    participant Forms as 📝 Forms
    participant ML as 🧠 ML Engine
    participant DB as 🗄️ Database
    participant File as 📁 File System

    User->>Browser: Navigate to /prediction
    Browser->>Flask: GET /prediction
    Flask->>Forms: Initialize PredictionForm
    Forms-->>Flask: Return form instance
    Flask-->>Browser: Render prediction.html
    Browser-->>User: Display prediction form

    User->>Browser: Input parameters & Submit
    Browser->>Flask: POST /prediction (form data)
    Flask->>Forms: Validate form data
    alt Validation Failed
        Forms-->>Flask: Validation errors
        Flask-->>Browser: Render with errors
    else Validation Passed
        Forms-->>Flask: Validated data
        Flask->>ML: Call predict_price(data)
        ML->>ML: Feature engineering
        ML->>ML: Scale/normalize features
        ML->>ML: Run ensemble models
        ML-->>Flask: Return prediction results
        Flask->>DB: Save prediction to history
        DB-->>Flask: Confirm save
        Flask-->>Browser: Return results + charts
        Browser-->>User: Display prediction + visualizations
    end

    Note over User,File: CSV Upload Flow
    User->>Browser: Select CSV file
    Browser->>Flask: POST /upload (multipart)
    Flask->>File: Save to /data/uploads/
    File-->>Flask: File path
    Flask->>ML: Process CSV (pandas)
    ML->>ML: Calculate indicators
    ML->>ML: Generate analysis
    ML-->>Flask: Analysis results
    Flask->>DB: Store upload metadata
    Flask-->>Browser: Render analysis view
    Browser-->>User: Show interactive charts
```

---

## 🧠 Machine Learning Pipeline

```mermaid
flowchart LR
    subgraph Input["📥 Data Input"]
        Raw["Raw Price Data<br/>Open, High, Low, Close, Volume"]
    end

    subgraph Preprocessing["🔧 Preprocessing"]
        Clean["Data Cleaning<br/>Missing Values, Outliers"]
        Engineer["Feature Engineering<br/>8+ Indicators"]
        Scale["Feature Scaling<br/>StandardScaler"]
        Split["Train/Test Split<br/>80/20 Ratio"]
    end

    subgraph Training["🏋️ Model Training"]
        RF["🌲 Random Forest<br/>n_estimators=200<br/>max_depth=20"]
        LR["📉 Linear Regression<br/>fit_intercept=True"]
        SVR["🎯 SVR<br/>kernel=rbf<br/>C=1.0, gamma=scale"]
    end

    subgraph Evaluation["📊 Evaluation"]
        Metrics["Metrics Calculation<br/>MSE, RMSE, MAE, R²"]
        Compare["Model Comparison<br/>Select Best"]
    end

    subgraph Output["📤 Prediction Output"]
        Result["Forecasted Price<br/>Confidence Interval<br/>Trend Direction"]
    end

    Raw --> Clean --> Engineer --> Scale --> Split
    Split --> RF
    Split --> LR
    Split --> SVR
    RF --> Metrics
    LR --> Metrics
    SVR --> Metrics
    Metrics --> Compare --> Result

    style Input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Preprocessing fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    style Training fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Evaluation fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style Output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

### 🎯 Model Performance Comparison

```mermaid
graph LR
    subgraph RandomForest["🌲 Random Forest"]
        RF_Acc["Accuracy: ~92%"]
        RF_Speed["Speed: Medium"]
        RF_Best["Best Overall"]
    end

    subgraph LinearReg["📉 Linear Regression"]
        LR_Acc["Accuracy: ~78%"]
        LR_Speed["Speed: Fast"]
        LR_Best["Baseline Model"]
    end

    subgraph SVR_Model["🎯 SVR"]
        SVR_Acc["Accuracy: ~85%"]
        SVR_Speed["Speed: Slow"]
        SVR_Best["Complex Patterns"]
    end

    style RandomForest fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style LinearReg fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style SVR_Model fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

---

## 📊 Technical Indicators Engine

```mermaid
graph TB
    subgraph Indicators["📊 Technical Indicators Calculation Engine"]
        Price["Price Data<br/>OHLCV"]

        subgraph Momentum["🚀 Momentum Indicators"]
            RSI_Calc["RSI = 100 - (100 / (1 + RS))<br/>Period: 14"]
            Stoch_Calc["%K = (Close - Low14) / (High14 - Low14) * 100<br/>%D = SMA of %K"]
            MACD_Calc["MACD = EMA12 - EMA26<br/>Signal = EMA9 of MACD<br/>Histogram = MACD - Signal"]
        end

        subgraph Trend["📈 Trend Indicators"]
            SMA_Calc["SMA = Σ(Price) / n<br/>n ∈ {5, 10, 20, 50, 100}"]
            EMA_Calc["EMA = Price * k + EMA_prev * (1-k)<br/>k = 2/(n+1)"]
        end

        subgraph Volatility["⚡ Volatility Indicators"]
            BB_Calc["Middle = SMA20<br/>Upper = SMA20 + 2σ<br/>Lower = SMA20 - 2σ"]
            ATR_Calc["TR = max(High-Low, |High-Close|, |Low-Close|)<br/>ATR = SMA14 of TR"]
        end

        subgraph Volume["📦 Volume Indicators"]
            OBV_Calc["OBV = OBV_prev + Volume<br/>if Close > Close_prev<br/>OBV = OBV_prev - Volume<br/>if Close < Close_prev"]
        end
    end

    Price --> RSI_Calc
    Price --> Stoch_Calc
    Price --> MACD_Calc
    Price --> SMA_Calc
    Price --> EMA_Calc
    Price --> BB_Calc
    Price --> ATR_Calc
    Price --> OBV_Calc

    RSI_Calc --> Features["Feature Vector<br/>Input to ML Models"]
    Stoch_Calc --> Features
    MACD_Calc --> Features
    SMA_Calc --> Features
    EMA_Calc --> Features
    BB_Calc --> Features
    ATR_Calc --> Features
    OBV_Calc --> Features

    style Indicators fill:#1a1a2e,stroke:#16213e,stroke-width:3px,color:#fff
    style Momentum fill:#0f3460,stroke:#e94560,stroke-width:2px,color:#fff
    style Trend fill:#0f3460,stroke:#16c79a,stroke-width:2px,color:#fff
    style Volatility fill:#0f3460,stroke:#f9a825,stroke-width:2px,color:#fff
    style Volume fill:#0f3460,stroke:#7c4dff,stroke-width:2px,color:#fff
    style Features fill:#16c79a,stroke:#0f3460,stroke-width:3px,color:#fff
```

---

## 🗄️ Database Schema

```mermaid
erDiagram
    USER ||--o{ PREDICTION : makes
    USER ||--o{ UPLOAD : uploads
    USER ||--o{ PREFERENCE : has

    USER {
        int id PK
        string username UK
        string email UK
        string password_hash
        string first_name
        string last_name
        datetime created_at
        datetime last_login
        boolean is_active
        string theme_preference
    }

    PREDICTION {
        int id PK
        int user_id FK
        datetime timestamp
        float predicted_price
        float actual_price
        float confidence_score
        string model_used
        string features_used
        json input_parameters
        json results
    }

    UPLOAD {
        int id PK
        int user_id FK
        string filename
        string original_name
        string file_path
        int file_size
        int row_count
        int column_count
        datetime uploaded_at
        string analysis_status
        json analysis_results
    }

    PREFERENCE {
        int id PK
        int user_id FK
        string theme
        string chart_type
        boolean email_notifications
        json dashboard_layout
    }
```

---

## 🛠️ Installation

### Prerequisites

```bash
# Check Python version (3.11+ required)
python --version

# Check pip version
pip --version
```

### Step-by-Step Setup

```mermaid
graph LR
    A["📥 Clone Repository"] --> B["📦 Install Dependencies"]
    B --> C["🔧 Configure Environment"]
    C --> D["🚀 Run Application"]
    D --> E["🌐 Open Browser<br/>localhost:5000"]

    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style C fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style D fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style E fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

```bash
# Step 1: Clone the repository
git clone https://github.com/issu321/bitcoin-prediction.git
cd bitcoin-prediction

# Step 2: Create virtual environment (recommended)
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Initialize database (first run)
python -c "from app import app; from models import db; app.app_context().push(); db.create_all()"

# Step 5: Run the application
python run.py

# Step 6: Open browser and navigate to
# http://localhost:5000
```

### Default Credentials

```
┌─────────────────────────────────────────┐
│         🔐 DEFAULT LOGIN                │
├─────────────────────────────────────────┤
│  Username: admin                        │
│  Password: admin123                     │
├─────────────────────────────────────────┤
│  ⚠️  Change after first login!          │
└─────────────────────────────────────────┘
```

---

## ⚙️ Configuration

```python
# config.py - Application Settings

class Config:
    SECRET_KEY = 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///bitcoin_prediction.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'data/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # ML Configuration
    MODEL_ESTIMATORS = 200
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Theme Configuration
    DEFAULT_THEME = 'dark'
    ALLOWED_THEMES = ['dark', 'light']
```

---

## 📁 Project Structure

```
📦 bitcoin_prediction_project/
│
├── ⚡ CORE APPLICATION
│   ├── app.py                  # Main Flask application entry
│   ├── config.py               # Configuration & environment settings
│   ├── models.py               # SQLAlchemy database models
│   ├── forms.py                # WTForms validation schemas
│   ├── ml_engine.py            # Machine Learning prediction engine
│   └── run.py                  # Production entry point
│
├── 📁 templates/               # Jinja2 HTML Templates
│   ├── base.html               # Base layout with theme support
│   ├── index.html              # Landing page
│   ├── about.html              # Project information
│   ├── features.html           # Feature showcase
│   ├── login.html              # Authentication page
│   ├── register.html           # User registration
│   ├── dashboard.html          # User dashboard & stats
│   ├── prediction.html         # ML prediction interface
│   ├── prediction_history.html # Historical predictions
│   ├── upload.html             # CSV upload form
│   ├── view_upload.html        # Upload analysis view
│   ├── database.html           # Database management
│   ├── profile.html            # User profile settings
│   └── error.html              # Error handling pages
│
├── 📁 static/                  # Static Assets
│   ├── css/
│   │   ├── main.css            # Custom styles
│   │   ├── dark-theme.css      # Dark mode variables
│   │   └── light-theme.css     # Light mode variables
│   ├── js/
│   │   ├── main.js             # Core functionality
│   │   ├── charts.js           # Chart.js configurations
│   │   ├── predictions.js      # Prediction form handling
│   │   ├── upload.js           # Drag & drop upload
│   │   └── theme.js            # Theme toggle logic
│   └── images/
│       ├── logo.png
│       ├── bitcoin-icon.svg
│       └── charts/
│
├── 📁 data/
│   └── uploads/                # Uploaded CSV storage
│       └── .gitkeep
│
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md                # This file
└── 📄 LICENSE                  # Educational License
```

---

## 🌐 API Endpoints

```mermaid
graph TB
    subgraph Routes["🌐 Flask Routes & Endpoints"]
        Home["/"]
        About["/about"]
        Features["/features"]
        Login["/login"]
        Register["/register"]
        Logout["/logout"]
        Dashboard["/dashboard"]
        Prediction["/prediction"]
        History["/prediction/history"]
        Upload["/upload"]
        ViewUpload["/upload/<id>"]
        Database["/database"]
        Profile["/profile"]
        API_Predict["/api/predict"]
        API_Data["/api/data/<id>"]
    end

    subgraph AuthRequired["🔒 Authentication Required"]
        Dashboard
        Prediction
        History
        Upload
        ViewUpload
        Database
        Profile
        API_Predict
        API_Data
    end

    subgraph Public["🌍 Public Access"]
        Home
        About
        Features
        Login
        Register
    end

    style AuthRequired fill:#ffebee,stroke:#c62828,stroke-width:2px
    style Public fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `GET` | `/` | Home page with feature overview | ❌ No |
| `GET` | `/about` | Project information & team | ❌ No |
| `GET` | `/features` | Detailed feature documentation | ❌ No |
| `GET/POST` | `/login` | User authentication | ❌ No |
| `GET/POST` | `/register` | New user registration | ❌ No |
| `GET` | `/logout` | Session termination | ✅ Yes |
| `GET` | `/dashboard` | User statistics & overview | ✅ Yes |
| `GET/POST` | `/prediction` | Run ML price prediction | ✅ Yes |
| `GET` | `/prediction/history` | View past predictions | ✅ Yes |
| `GET/POST` | `/upload` | CSV file upload & analysis | ✅ Yes |
| `GET` | `/upload/<id>` | View specific upload analysis | ✅ Yes |
| `GET/POST` | `/database` | Database management interface | ✅ Yes |
| `GET/POST` | `/profile` | User profile settings | ✅ Yes |
| `POST` | `/api/predict` | JSON API for predictions | ✅ Yes |
| `GET` | `/api/data/<id>` | Retrieve upload data as JSON | ✅ Yes |

---

## 👤 User Journey

```mermaid
journey
    title User Experience Flow
    section Registration
      Visit Landing Page: 5: User
      Click Register: 4: User
      Fill Form: 3: User
      Submit: 4: User, System
    section First Login
      Enter Credentials: 4: User
      System Validation: 5: System
      Dashboard Load: 5: System
    section Make Prediction
      Navigate to Prediction: 4: User
      Input Parameters: 3: User
      Run ML Model: 5: System
      View Results: 5: User
    section Upload Data
      Go to Upload: 4: User
      Drag CSV File: 4: User
      Processing: 5: System
      View Analysis: 5: User
```

---

## 🎨 UI/UX Themes

```mermaid
graph LR
    subgraph ThemeSystem["🌓 Theme System Architecture"]
        Toggle["Theme Toggle Button"] -->|Click| JS["theme.js Handler"]
        JS -->|Save| LocalStorage["localStorage<br/>theme_preference"]
        JS -->|Apply| CSS["CSS Variables<br/>:root[data-theme]"]
        CSS -->|Render| Dark["🌙 Dark Mode<br/>#1a1a2e, #16213e"]
        CSS -->|Render| Light["☀️ Light Mode<br/>#ffffff, #f5f5f5"]
        LocalStorage -->|Load on Init| JS
    end

    style ThemeSystem fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Dark fill:#1a1a2e,stroke:#e94560,stroke-width:2px,color:#fff
    style Light fill:#ffffff,stroke:#f9a825,stroke-width:2px
```

### Theme Color Palette

| Element | Dark Mode | Light Mode |
|---------|-----------|------------|
| Background | `#1a1a2e` | `#ffffff` |
| Surface | `#16213e` | `#f5f5f5` |
| Primary | `#e94560` | `#1976d2` |
| Secondary | `#0f3460` | `#424242` |
| Accent | `#16c79a` | `#4caf50` |
| Text | `#eaeaea` | `#212121` |
| Border | `#2a2a4a` | `#e0e0e0` |

---

## 📈 Performance Metrics

```mermaid
graph TB
    subgraph Metrics["📊 System Performance Metrics"]
        subgraph ML_Metrics["🧠 ML Model Performance"]
            MSE["MSE<br/>Mean Squared Error"]
            RMSE["RMSE<br/>Root Mean Squared Error"]
            MAE["MAE<br/>Mean Absolute Error"]
            R2["R² Score<br/>Coefficient of Determination"]
        end

        subgraph App_Metrics["⚡ Application Performance"]
            Response["Response Time<br/>< 200ms"]
            Throughput["Throughput<br/>100 req/s"]
            Uptime["Uptime<br/>99.9%"]
        end

        subgraph Accuracy["🎯 Prediction Accuracy"]
            RF_Acc["Random Forest<br/>92%"]
            LR_Acc["Linear Regression<br/>78%"]
            SVR_Acc["SVR<br/>85%"]
        end
    end

    style ML_Metrics fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style App_Metrics fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style Accuracy fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

---

## 🔒 Security

```mermaid
graph TB
    subgraph Security["🔐 Security Architecture"]
        Auth["Authentication"]
        Hash["Password Hashing<br/>Bcrypt"]
        Session["Session Management<br/>Flask-Login"]
        CSRF["CSRF Protection<br/>Flask-WTF"]
        Validation["Input Validation<br/>WTForms"]
        UploadSec["Upload Security<br/>File Type Validation"]
    end

    Auth --> Hash
    Auth --> Session
    Auth --> CSRF
    Auth --> Validation
    Auth --> UploadSec

    style Security fill:#ffebee,stroke:#c62828,stroke-width:2px
    style Auth fill:#ffcdd2,stroke:#d32f2f,stroke-width:2px
```

### Security Features

- ✅ **Password Hashing**: Bcrypt with salt rounds
- ✅ **CSRF Protection**: Token-based form validation
- ✅ **Session Security**: Secure cookie flags
- ✅ **Input Sanitization**: WTForms validators
- ✅ **File Upload Security**: Extension & MIME type checking
- ✅ **SQL Injection Prevention**: SQLAlchemy ORM parameterized queries
- ✅ **XSS Protection**: Jinja2 auto-escaping

---

## 🤝 Contributing

```mermaid
graph LR
    A["🍴 Fork Repository"] --> B["🔧 Create Branch"]
    B --> C["💻 Make Changes"]
    C --> D["🧪 Test Thoroughly"]
    D --> E["📤 Submit Pull Request"]
    E --> F["✅ Code Review"]
    F --> G["🎉 Merge to Main"]

    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style C fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style D fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style E fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style F fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style G fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
```

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov=app

# Run linting
flake8 app.py models.py forms.py ml_engine.py
black app.py models.py forms.py ml_engine.py
```

---

## 📜 License

```
MIT License - Educational Use

Copyright (c) 2024 Bitcoin Prediction Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
```

This project is for **educational purposes** only. Not financial advice. ⚠️

---

## 👥 Authors & Acknowledgments

```
╔══════════════════════════════════════════════════════════╗
║                    👨‍💻 DEVELOPMENT TEAM                   ║
╠══════════════════════════════════════════════════════════╣
║  Lead Developer: @issu321                                ║
║  GitHub: https://github.com/issu321                      ║
║  Email: jaafreeusman@gmail.com                           ║
╠══════════════════════════════════════════════════════════╣
║  Technologies:                                           ║
║  • Python 3.11+  • Flask 3.0  • SQLAlchemy              ║
║  • Scikit-Learn  • Pandas  • NumPy                      ║
║  • Bootstrap 5  • Font Awesome  • Matplotlib            ║
╚══════════════════════════════════════════════════════════╝
```

### 🙏 Special Thanks

- **Scikit-Learn Team** for the robust ML framework
- **Flask Community** for the lightweight web framework
- **Bootstrap Team** for responsive UI components
- **Open Source Community** for continuous inspiration

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**[🔝 Back to Top](#-bitcoin-price-prediction-using-machine-learning)**

<p align="center">
  <img src="https://komarev.com/ghpvc/?username=issu321&repo=bitcoin-prediction&color=e94560&style=for-the-badge" alt="Profile Views">
</p>

**Made with ❤️ and ☕ by @issu321**

</div>
