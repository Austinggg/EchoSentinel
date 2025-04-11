# services/content_analysis/video_transcribe.py
import os
import ffmpeg
import requests
from pathlib import Path

class VideoTranscriber:
    def __init__(self, model_service_url="http://121.48.227.136:3000/transcribe"):
        self.model_service_url = model_service_url

    def extract_audio(self, video_path):
        """提取音频并返回临时路径"""
        audio_path = f"{Path(video_path).stem}_audio.wav"
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, format='wav', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            print(f"音频提取失败: {e}")
            return None

    def transcribe_video(self, video_path):
        # 提取音频
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        # 调用远程模型服务
        with open(audio_path, "rb") as f:
            try:
                response = requests.post(
                    self.model_service_url,
                    files={"audio": f},
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                # 清理临时音频
                os.remove(audio_path)
                return result
            except Exception as e:
                os.remove(audio_path)
                print(f"模型服务调用失败: {e}")
                return None