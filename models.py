from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import uuid

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    profile_image = db.Column(db.String(200), default='default.png')
    theme_preference = db.Column(db.String(10), default='dark')

    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    uploads = db.relationship('CSVUpload', backref='user', lazy=True, cascade='all, delete-orphan')
    login_logs = db.relationship('LoginLog', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_full_name(self):
        return f"{self.first_name or ''} {self.last_name or ''}".strip() or self.username

    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    current_price = db.Column(db.Float)
    predicted_price = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    actual_price = db.Column(db.Float, nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    features_used = db.Column(db.Text)
    chart_path = db.Column(db.String(300))

    def __repr__(self):
        return f'<Prediction {self.id}>'

class CSVUpload(db.Model):
    __tablename__ = 'csv_uploads'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)
    file_size = db.Column(db.Integer)
    row_count = db.Column(db.Integer)
    column_count = db.Column(db.Integer)
    columns = db.Column(db.Text)
    description = db.Column(db.Text)  # <-- THIS LINE WAS ADDED
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_status = db.Column(db.String(50), default='pending')
    analysis_result = db.Column(db.Text)

    def __repr__(self):
        return f'<CSVUpload {self.filename}>'

class LoginLog(db.Model):
    __tablename__ = 'login_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    status = db.Column(db.String(20), default='success')

    def __repr__(self):
        return f'<LoginLog {self.id}>'

class SystemLog(db.Model):
    __tablename__ = 'system_logs'

    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    severity = db.Column(db.String(20), default='info')

    def __repr__(self):
        return f'<SystemLog {self.id}>'