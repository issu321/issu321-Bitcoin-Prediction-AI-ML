from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash
import os
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from config import Config
from models import db, User, Prediction, CSVUpload, LoginLog, SystemLog
from forms import RegistrationForm, LoginForm, PredictionForm, CSVUploadForm, ProfileUpdateForm
from ml_engine import BitcoinPredictionEngine, get_model_description

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Ensure static/images directory exists
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==================== CONTEXT PROCESSORS ====================

@app.context_processor
def inject_globals():
    return {
        'current_year': datetime.now().year,
        'app_name': 'Bitcoin Price Predictor',
        'app_version': '2.0.0'
    }

# ==================== HOME & STATIC PAGES ====================

@app.route('/')
def index():
    """Landing page with project overview"""
    features = [
        {
            'icon': 'fa-chart-line',
            'title': 'Advanced Price Prediction',
            'description': 'Utilize cutting-edge machine learning algorithms including Random Forest, Linear Regression, and Support Vector Regression to forecast Bitcoin prices with high accuracy.'
        },
        {
            'icon': 'fa-brain',
            'title': 'Feature Engineering',
            'description': 'Comprehensive technical indicators including RSI, MACD, Bollinger Bands, Moving Averages, Stochastic Oscillator, and Volume Analysis for robust predictions.'
        },
        {
            'icon': 'fa-upload',
            'title': 'CSV Data Analysis',
            'description': 'Upload your own Bitcoin historical data CSV files and get instant analysis, correlation matrices, statistical summaries, and custom predictions.'
        },
        {
            'icon': 'fa-chart-area',
            'title': 'Interactive Dashboard',
            'description': 'Real-time visualization with dynamic charts showing price trends, volume analysis, technical indicators, and prediction forecasts in an elegant interface.'
        },
        {
            'icon': 'fa-shield-alt',
            'title': 'Secure User Management',
            'description': 'Full authentication system with password hashing, session management, login tracking, and user activity monitoring for complete data security.'
        },
        {
            'icon': 'fa-database',
            'title': 'Data Management',
            'description': 'Comprehensive database with user profiles, prediction history, CSV uploads, login logs, and system analytics with full CRUD operations.'
        }
    ]

    tools = [
        {'name': 'Python 3.11', 'category': 'Language', 'icon': 'fa-python'},
        {'name': 'Flask 3.0', 'category': 'Web Framework', 'icon': 'fa-flask'},
        {'name': 'Scikit-Learn', 'category': 'ML Library', 'icon': 'fa-cogs'},
        {'name': 'Pandas', 'category': 'Data Processing', 'icon': 'fa-table'},
        {'name': 'NumPy', 'category': 'Numerical Computing', 'icon': 'fa-calculator'},
        {'name': 'Matplotlib', 'category': 'Visualization', 'icon': 'fa-chart-bar'},
        {'name': 'Plotly', 'category': 'Interactive Charts', 'icon': 'fa-chart-pie'},
        {'name': 'SQLAlchemy', 'category': 'ORM', 'icon': 'fa-database'},
        {'name': 'WTForms', 'category': 'Form Handling', 'icon': 'fa-edit'},
        {'name': 'Bootstrap 5', 'category': 'CSS Framework', 'icon': 'fa-css3'},
        {'name': 'Font Awesome', 'category': 'Icons', 'icon': 'fa-icons'},
        {'name': 'SQLite', 'category': 'Database', 'icon': 'fa-server'}
    ]

    stats = {
        'total_users': User.query.count(),
        'total_predictions': Prediction.query.count(),
        'total_uploads': CSVUpload.query.count(),
        'models_available': 3
    }

    return render_template('index.html', features=features, tools=tools, stats=stats)

@app.route('/about')
def about():
    """About the project page"""
    return render_template('about.html')

@app.route('/features')
def features():
    """Detailed features page"""
    return render_template('features.html')

# ==================== AUTHENTICATION ====================

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data
        )
        user.set_password(form.password.data)

        db.session.add(user)
        db.session.commit()

        log = SystemLog(action='User Registration', details=f'New user registered: {user.username}', user_id=user.id)
        db.session.add(log)
        db.session.commit()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter(
            (User.username == form.username.data) | (User.email == form.username.data)
        ).first()

        if user and user.check_password(form.password.data):
            if not user.is_active:
                flash('Your account has been deactivated. Contact admin.', 'danger')
                return redirect(url_for('login'))

            login_user(user, remember=form.remember.data)
            user.last_login = datetime.utcnow()

            login_log = LoginLog(
                user_id=user.id,
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string[:500] if request.user_agent else 'Unknown',
                status='success'
            )
            db.session.add(login_log)
            db.session.commit()

            log = SystemLog(action='User Login', details=f'User {user.username} logged in', user_id=user.id)
            db.session.add(log)
            db.session.commit()

            flash(f'Welcome back, {user.get_full_name()}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            if user:
                login_log = LoginLog(
                    user_id=user.id,
                    ip_address=request.remote_addr,
                    user_agent=request.user_agent.string[:500] if request.user_agent else 'Unknown',
                    status='failed'
                )
                db.session.add(login_log)
                db.session.commit()

            flash('Invalid username/email or password.', 'danger')

    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    login_log = LoginLog(
        user_id=current_user.id,
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string[:500] if request.user_agent else 'Unknown',
        status='logout'
    )
    db.session.add(login_log)

    log = SystemLog(action='User Logout', details=f'User {current_user.username} logged out', user_id=current_user.id)
    db.session.add(log)
    db.session.commit()

    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

# ==================== DASHBOARD ====================

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.prediction_date.desc()).limit(10).all()
    uploads = CSVUpload.query.filter_by(user_id=current_user.id).order_by(CSVUpload.upload_date.desc()).limit(10).all()

    stats = {
        'total_predictions': Prediction.query.filter_by(user_id=current_user.id).count(),
        'total_uploads': CSVUpload.query.filter_by(user_id=current_user.id).count(),
        'total_logins': LoginLog.query.filter_by(user_id=current_user.id, status='success').count(),
        'member_since': current_user.created_at.strftime('%B %Y') if current_user.created_at else 'N/A'
    }

    return render_template('dashboard.html', predictions=predictions, uploads=uploads, stats=stats)

# ==================== PREDICTION ====================

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    """Bitcoin price prediction page"""
    form = PredictionForm()
    result = None
    chart_path = None

    if form.validate_on_submit():
        model_type = form.model_type.data
        days = form.days_to_predict.data

        try:
            engine = BitcoinPredictionEngine(model_type=model_type)
            df = engine.generate_sample_data(days=365)
            metrics = engine.train_model(df)
            predictions = engine.predict_future(df, days=days)

            # FIX: Create images directory if it doesn't exist, use proper path
            chart_filename = f'prediction_{current_user.id}_{int(datetime.now().timestamp())}.png'
            chart_full_path = os.path.join(STATIC_IMAGES_DIR, chart_filename)
            engine.generate_charts(df, predictions, save_path=chart_full_path)
            chart_path = f'images/{chart_filename}'

            pred = Prediction(
                user_id=current_user.id,
                model_type=model_type,
                current_price=predictions['current_price'],
                predicted_price=predictions['predictions'][-1],
                confidence_score=metrics.get('test_r2', 0),
                features_used=json.dumps(engine.feature_columns[:10]),
                chart_path=chart_path
            )
            db.session.add(pred)

            log = SystemLog(
                action='Prediction Run',
                details=f'User {current_user.username} ran {model_type} prediction for {days} days',
                user_id=current_user.id
            )
            db.session.add(log)
            db.session.commit()

            result = {
                'metrics': metrics,
                'predictions': predictions,
                'model_info': get_model_description(model_type),
                'chart_path': chart_path
            }

            flash('Prediction completed successfully!', 'success')

        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'danger')
            app.logger.error(f'Prediction error: {str(e)}')

    return render_template('prediction.html', form=form, result=result)

@app.route('/prediction/history')
@login_required
def prediction_history():
    """View prediction history"""
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.prediction_date.desc()).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template('prediction_history.html', predictions=predictions)

# ==================== CSV UPLOAD & ANALYSIS ====================

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_csv():
    """Upload and analyze CSV files"""
    form = CSVUploadForm()
    analysis_result = None
    uploaded_file = None

    if form.validate_on_submit():
        file = form.file.data
        if file and file.filename.endswith('.csv'):
            try:
                unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
                file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
                file.save(file_path)

                df = pd.read_csv(file_path)

                csv_upload = CSVUpload(
                    user_id=current_user.id,
                    filename=unique_filename,
                    original_filename=file.filename,
                    file_path=file_path,
                    file_size=os.path.getsize(file_path),
                    row_count=len(df),
                    column_count=len(df.columns),
                    columns=','.join(df.columns.tolist()),
                    description=form.description.data
                )
                db.session.add(csv_upload)

                engine = BitcoinPredictionEngine()
                analysis = engine.analyze_csv(file_path)

                csv_upload.analysis_result = json.dumps(analysis)
                csv_upload.analysis_status = 'completed'

                log = SystemLog(
                    action='CSV Upload',
                    details=f'User {current_user.username} uploaded {file.filename}',
                    user_id=current_user.id
                )
                db.session.add(log)
                db.session.commit()

                analysis_result = analysis
                uploaded_file = csv_upload

                flash(f'File uploaded and analyzed successfully! {len(df)} rows processed.', 'success')

            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                app.logger.error(f'Upload error: {str(e)}')
        else:
            flash('Please upload a valid CSV file.', 'warning')

    uploads = CSVUpload.query.filter_by(user_id=current_user.id).order_by(CSVUpload.upload_date.desc()).all()

    return render_template('upload.html', form=form, analysis_result=analysis_result, uploads=uploads, uploaded_file=uploaded_file)

@app.route('/upload/view/<int:upload_id>')
@login_required
def view_upload(upload_id):
    """View uploaded CSV details"""
    upload = CSVUpload.query.get_or_404(upload_id)

    if upload.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('upload_csv'))

    analysis = json.loads(upload.analysis_result) if upload.analysis_result else {}

    try:
        df = pd.read_csv(upload.file_path)
        preview_data = df.head(50).to_dict('records')
        preview_columns = df.columns.tolist()
    except:
        preview_data = []
        preview_columns = []

    return render_template('view_upload.html', upload=upload, analysis=analysis,
                         preview_data=preview_data, preview_columns=preview_columns)

@app.route('/upload/delete/<int:upload_id>', methods=['POST'])
@login_required
def delete_upload(upload_id):
    """Delete uploaded CSV"""
    upload = CSVUpload.query.get_or_404(upload_id)

    if upload.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('upload_csv'))

    try:
        if os.path.exists(upload.file_path):
            os.remove(upload.file_path)

        db.session.delete(upload)

        log = SystemLog(
            action='CSV Delete',
            details=f'User {current_user.username} deleted upload {upload.original_filename}',
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()

        flash('File deleted successfully.', 'success')
    except Exception as e:
        flash(f'Error deleting file: {str(e)}', 'danger')

    return redirect(url_for('upload_csv'))

# ==================== DATABASE ADMIN ====================

@app.route('/database')
@login_required
def database_view():
    """Database management view"""
    users = User.query.all()
    predictions = Prediction.query.order_by(Prediction.prediction_date.desc()).limit(50).all()
    uploads = CSVUpload.query.order_by(CSVUpload.upload_date.desc()).limit(50).all()
    login_logs = LoginLog.query.order_by(LoginLog.login_time.desc()).limit(100).all()
    system_logs = SystemLog.query.order_by(SystemLog.timestamp.desc()).limit(100).all()

    summary = {
        'total_users': User.query.count(),
        'total_predictions': Prediction.query.count(),
        'total_uploads': CSVUpload.query.count(),
        'total_logins': LoginLog.query.filter_by(status='success').count(),
        'failed_logins': LoginLog.query.filter_by(status='failed').count(),
        'system_events': SystemLog.query.count()
    }

    return render_template('database.html', users=users, predictions=predictions,
                         uploads=uploads, login_logs=login_logs, system_logs=system_logs, summary=summary)

# ==================== PROFILE ====================

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management"""
    form = ProfileUpdateForm()

    if request.method == 'GET':
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email

    if form.validate_on_submit():
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data
        current_user.email = form.email.data

        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))

    recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.prediction_date.desc()).limit(5).all()
    recent_uploads = CSVUpload.query.filter_by(user_id=current_user.id).order_by(CSVUpload.upload_date.desc()).limit(5).all()
    recent_logins = LoginLog.query.filter_by(user_id=current_user.id).order_by(LoginLog.login_time.desc()).limit(10).all()

    return render_template('profile.html', form=form,
                         recent_predictions=recent_predictions,
                         recent_uploads=recent_uploads,
                         recent_logins=recent_logins)

@app.route('/profile/theme', methods=['POST'])
@login_required
def update_theme():
    """Update theme preference"""
    theme = request.json.get('theme', 'dark')
    current_user.theme_preference = theme
    db.session.commit()
    return jsonify({'success': True, 'theme': theme})

# ==================== API ENDPOINTS ====================

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for predictions"""
    data = request.get_json()
    model_type = data.get('model_type', 'random_forest')
    days = data.get('days', 7)

    try:
        engine = BitcoinPredictionEngine(model_type=model_type)
        df = engine.generate_sample_data(days=365)
        metrics = engine.train_model(df)
        predictions = engine.predict_future(df, days=days)

        return jsonify({
            'success': True,
            'metrics': metrics,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
@login_required
def api_stats():
    """Get system statistics"""
    stats = {
        'total_users': User.query.count(),
        'total_predictions': Prediction.query.count(),
        'total_uploads': CSVUpload.query.count(),
        'active_today': LoginLog.query.filter(
            LoginLog.login_time >= datetime.utcnow().replace(hour=0, minute=0, second=0)
        ).count()
    }
    return jsonify(stats)

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_code=404, error_message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('error.html', error_code=500, error_message='Internal server error'), 500

# ==================== INITIALIZATION ====================

def init_db():
    """Initialize database with tables"""
    with app.app_context():
        db.create_all()

        # Check and add missing columns (simple migration)
        try:
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('csv_uploads')]
            if 'description' not in columns:
                from sqlalchemy import text
                db.session.execute(text("ALTER TABLE csv_uploads ADD COLUMN description TEXT"))
                db.session.commit()
                print("Added 'description' column to csv_uploads table")
        except Exception as e:
            print(f"Migration check: {e}")

        # Create admin user if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@bitcoinpredictor.local',
                first_name='System',
                last_name='Admin',
                is_active=True
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created: username='admin', password='admin123'")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)