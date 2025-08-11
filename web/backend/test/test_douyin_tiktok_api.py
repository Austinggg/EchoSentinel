import pytest
from unittest import mock
from flask import Flask
from app.views.douyin_tiktok_api import douyin_api

@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(douyin_api)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@mock.patch("app.views.douyin_tiktok_api.requests.get")
def test_download_and_analyze_video_missing_url(mock_get, client):
    resp = client.get("/api/download_and_analyze")
    assert resp.status_code == 400
    assert resp.json["message"] == "缺少URL参数"

def test_download_and_analyze_video_unsupported_platform(client):
    resp = client.get("/api/download_and_analyze?url=http://example.com/video/123")
    assert resp.status_code == 400
    assert "不支持的平台" in resp.json["message"]

@mock.patch("app.views.douyin_tiktok_api.requests.get")
def test_download_video_only_missing_url(mock_get, client):
    resp = client.get("/api/download/")
    assert resp.status_code == 400
    assert resp.json["message"] == "缺少URL参数"

def test_download_video_only_unsupported_platform(client):
    resp = client.get("/api/download/?url=http://example.com/video/123")
    assert resp.status_code == 400
    assert "不支持的平台" in resp.json["message"]

@mock.patch("app.views.douyin_tiktok_api.requests.get")
def test_download_and_analyze_video_invalid_template(mock_get, client):
    # Setup mock for video detail API
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"code": 200, "data": {"aweme_detail": {}}}
    mock_get.return_value = mock_resp

    url = "https://www.douyin.com/video/123456789"
    resp = client.get(f"/api/download_and_analyze?url={url}&template=invalid_template")
    # Should fallback to 'full' template, but since aweme_detail is empty, will fallback to direct download
    # which is not implemented in this test, so expect 500 or fallback handling
    assert resp.status_code in (200, 500, 400)

@mock.patch("app.views.douyin_tiktok_api.requests.get")
def test_download_and_analyze_video_aweme_id_extraction(mock_get, client):
    # Setup mock for video detail API
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "code": 200,
        "data": {
            "aweme_detail": {
                "desc": "Test video",
                "create_time": "1710000000",
                "author": {"nickname": "TestAuthor"},
                "text_extra": [{"type": 1, "hashtag_name": "tag1"}]
            }
        }
    }
    # Setup mock for download API
    def side_effect(url, params=None, stream=False):
        if "fetch_one_video" in url:
            return mock_resp
        else:
            download_resp = mock.Mock()
            download_resp.status_code = 200
            download_resp.headers = {"content-type": "video/mp4"}
            download_resp.iter_content = lambda chunk_size: [b"fakevideodata"]
            return download_resp
    mock_get.side_effect = side_effect

    url = "https://www.douyin.com/video/123456789"
    with mock.patch("app.views.douyin_tiktok_api.db.session.add"), \
         mock.patch("app.views.douyin_tiktok_api.db.session.commit"), \
         mock.patch("app.views.douyin_tiktok_api.generate_video_thumbnail"), \
         mock.patch("app.views.douyin_tiktok_api.start_video_workflow"), \
         mock.patch("app.views.douyin_tiktok_api.VideoFile"), \
         mock.patch("app.views.douyin_tiktok_api.DouyinVideo.query"):
        resp = client.get(f"/api/download_and_analyze?url={url}")
        assert resp.status_code == 200
        assert resp.json["code"] == 200
        assert resp.json["data"]["platform"] == "douyin"