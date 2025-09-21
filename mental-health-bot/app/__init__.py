from flask import Flask
from db import init_db

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev'

    # Initialize database
    init_db()

    # Register blueprints
    from pattern_analysis import bp as main_bp  # Web interface
    app.register_blueprint(main_bp)
    
    from routes import api_bp  # API endpoints
    app.register_blueprint(api_bp)

    return app