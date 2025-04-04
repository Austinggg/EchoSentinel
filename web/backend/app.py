from flask import send_from_directory

from api.auth import auth_bp
from api.test import test_bp
from api.user import user_bp
from utils.database import init_dataset
from utils.extensions import app

init_dataset(app)


@app.route("/")
def index():
    return send_from_directory("dist", "index.html")


app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(test_bp)

if __name__ == "__main__":
    app.run(debug=True, port=8000)  # 开启调试模式（包含热重载）
