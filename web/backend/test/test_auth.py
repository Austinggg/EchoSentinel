import pytest
from app import create_app
from app.models.user import User  # 只导入 User
from app.utils.extensions import db  # 从正确位置导入 db

# ================================
# 测试环境配置
# ================================

@pytest.fixture(scope='session')
def app():
    """创建测试用的 Flask 应用实例"""
    app = create_app()
    app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
    })

    with app.app_context():
        yield app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()


@pytest.fixture
def db_session(app):
    """数据库会话 fixture"""
    with app.app_context():
        connection = db.engine.connect()
        transaction = connection.begin()
        db.session.configure(bind=connection)

        yield db.session

        transaction.rollback()
        connection.close()
        db.session.remove()


# ================================
# 数据库连通性测试
# ================================

def test_database_connection(app):
    """测试数据库连接"""
    with app.app_context():
        result = db.session.execute(db.text("SELECT 1"))
        assert result.fetchone()[0] == 1


def test_user_table_exists(app):
    """测试用户表是否存在"""
    with app.app_context():
        result = db.session.execute(db.text("""
                                            SELECT COUNT(*)
                                            FROM information_schema.tables
                                            WHERE table_schema = DATABASE()
                                              AND table_name = 'user'
        """))
        assert result.fetchone()[0] == 1


# ================================
# 测试数据管理
# ================================

@pytest.fixture
def test_user(db_session):
    """创建测试用户"""
    import time
    import random

    # 使用时间戳和随机数生成唯一用户名
    unique_username = f"test_user_{int(time.time())}_{random.randint(1000, 9999)}"

    user = User()
    user.username = unique_username
    user.set_password("test_password_456")

    db_session.add(user)
    db_session.commit()

    yield {
        'username': unique_username,
        'password': 'test_password_456',
        'user_object': user
    }

    # 可选：测试完成后清理数据
    db_session.delete(user)
    db_session.commit()

# ================================
# 登录接口测试
# ================================

def test_auth_login_missing_username(client):
    """测试登录 - 缺少用户名"""
    response = client.post("/api/auth/login",
                           json={"password": "some_password"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["code"] != 0
    assert "用户名" in data["message"]


def test_auth_login_missing_password(client):
    """测试登录 - 缺少密码"""
    response = client.post("/api/auth/login",
                           json={"username": "some_user"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["code"] != 0
    assert "密码" in data["message"]


def test_auth_login_user_not_found(client):
    """测试登录 - 用户不存在"""
    response = client.post("/api/auth/login",
                           json={
                               "username": "nonexistent_user",
                               "password": "some_password"
                           })

    assert response.status_code == 200
    data = response.get_json()
    assert data["code"] != 0
    assert "不存在" in data["message"]


def test_auth_login_wrong_password(client, test_user):
    """测试登录 - 密码错误"""
    response = client.post("/api/auth/login",
                           json={
                               "username": test_user['username'],
                               "password": "wrong_password"
                           })

    assert response.status_code == 200
    data = response.get_json()
    assert data["code"] != 0
    assert "错误" in data["message"]


def test_auth_login_success(client, test_user):
    """测试登录 - 成功"""
    response = client.post("/api/auth/login",
                           json={
                               "username": test_user['username'],
                               "password": test_user['password']
                           })

    assert response.status_code == 200
    data = response.get_json()
    assert data["code"] == 0
    assert data["data"]["user"] == test_user['username']
    assert "accessToken" in data["data"]


# ================================
# 注册接口测试
# ================================

def test_auth_register_success(client):
    """测试注册 - 成功"""
    import time
    unique_username = f"new_user_{int(time.time())}"

    response = client.post("/api/auth/register",
                           json={
                               "username": unique_username,
                               "password": "new_password"
                           })

    assert response.status_code == 200
    data = response.get_json()
    assert data["code"] == 0


def test_auth_register_duplicate_username(client, test_user):
    """测试注册 - 用户名重复"""
    response = client.post("/api/auth/register",
                           json={
                               "username": test_user['username'],
                               "password": "some_password"
                           })

    assert response.status_code == 200
    data = response.get_json()
    assert data["code"] != 0


# ================================
# 登出接口测试
# ================================

def test_auth_logout(client):
    """测试登出"""
    response = client.post("/api/auth/logout")
    assert response.status_code == 200
    data = response.get_json()
    assert data["code"] == 0


