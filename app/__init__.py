from flask import Flask

def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    
    # Configure upload parsing
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload
    
    with app.app_context():
        from . import routes
        
    return app
