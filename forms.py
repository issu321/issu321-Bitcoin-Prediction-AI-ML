from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField, FloatField, FileField, TextAreaField, IntegerField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, NumberRange
from models import User

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=20, message='Username must be between 3 and 20 characters')
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message='Please enter a valid email address')
    ])
    first_name = StringField('First Name', validators=[
        Length(max=50, message='First name too long')
    ])
    last_name = StringField('Last Name', validators=[
        Length(max=50, message='Last name too long')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=6, message='Password must be at least 6 characters long')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Create Account')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered. Please use a different one or login.')

class LoginForm(FlaskForm):
    username = StringField('Username or Email', validators=[
        DataRequired(),
        Length(min=3, max=120)
    ])
    password = PasswordField('Password', validators=[
        DataRequired()
    ])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class PredictionForm(FlaskForm):
    model_type = SelectField('Model Type', choices=[
        ('random_forest', 'Random Forest'),
        ('linear_regression', 'Linear Regression'),
        ('svr', 'Support Vector Regression')
    ], default='random_forest')
    days_to_predict = IntegerField('Days to Predict', validators=[
        DataRequired(),
        NumberRange(min=1, max=30, message='Must be between 1 and 30 days')
    ], default=7)
    submit = SubmitField('Run Prediction')

class CSVUploadForm(FlaskForm):
    file = FileField('Upload CSV File', validators=[
        DataRequired()
    ])
    description = TextAreaField('Description (Optional)', validators=[
        Length(max=500)
    ])
    submit = SubmitField('Upload & Analyze')

class ProfileUpdateForm(FlaskForm):
    first_name = StringField('First Name', validators=[Length(max=50)])
    last_name = StringField('Last Name', validators=[Length(max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Update Profile')
