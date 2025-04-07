from flask import send_from_directory

from api import auth, menu, test, user, userAnalyse
from utils.database import init_dataset
from utils.extensions import app

init_dataset(app)


@app.route("/")
def index():
    return send_from_directory("dist", "index.html")


app.register_blueprint(auth.bp)
app.register_blueprint(menu.bp)
app.register_blueprint(test.bp)
app.register_blueprint(user.bp)
app.register_blueprint(userAnalyse.bp)
if __name__ == "__main__":
    app.run(debug=True, port=8000)  # 开启调试模式（包含热重载）
