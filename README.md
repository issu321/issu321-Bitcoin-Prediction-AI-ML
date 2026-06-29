# Bitcoin Price Prediction Using Machine Learning

A comprehensive Flask-based web application for Bitcoin price prediction using advanced machine learning algorithms.

## Features

- **Advanced Price Prediction**: Random Forest, Linear Regression, and SVR models
- **Feature Engineering**: RSI, MACD, Bollinger Bands, Moving Averages, Stochastic Oscillator, ATR, OBV
- **CSV Data Upload & Analysis**: Upload custom datasets with automatic analysis
- **Interactive Dashboards**: Real-time charts and visualizations
- **User Management**: Full authentication with secure password hashing
- **Database Management**: Complete CRUD operations with SQLite
- **Dark/Light Theme**: 100% working toggle with persistent preferences

## Installation

### Step 1: Install Python 3.11 or higher

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python run.py
```

### Step 4: Open Browser
Navigate to `http://localhost:5000`

## Default Login
- **Username**: `admin`
- **Password**: `admin123`

## Project Structure
```
bitcoin_prediction_project/
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── models.py               # Database models (SQLAlchemy)
├── forms.py                # WTForms definitions
├── ml_engine.py            # Machine Learning engine
├── run.py                  # Entry point script
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   ├── features.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── prediction.html
│   ├── prediction_history.html
│   ├── upload.html
│   ├── view_upload.html
│   ├── database.html
│   ├── profile.html
│   └── error.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
└── data/
    └── uploads/            # Uploaded CSV files
```

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Home | `/` | Landing page with features overview |
| About | `/about` | Project information |
| Features | `/features` | Detailed feature list |
| Login | `/login` | User authentication |
| Register | `/register` | New user registration |
| Dashboard | `/dashboard` | User dashboard with stats |
| Prediction | `/prediction` | Run ML predictions |
| Upload | `/upload` | CSV upload and analysis |
| Database | `/database` | Database management view |
| Profile | `/profile` | User profile settings |

## Machine Learning Models

1. **Random Forest Regressor**: 200 estimators, best overall accuracy
2. **Linear Regression**: Fast and interpretable baseline
3. **Support Vector Regression**: Effective for complex patterns

## Technical Indicators

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Simple & Exponential Moving Averages (5, 10, 20, 50, 100)
- Stochastic Oscillator
- Average True Range (ATR)
- On-Balance Volume (OBV)
- Volume Analysis

## Technologies Used

- Python 3.11+
- Flask 3.0
- SQLAlchemy + SQLite
- Scikit-Learn
- Pandas & NumPy
- Matplotlib
- Bootstrap 5
- Font Awesome

## License

This project is for educational purposes.
