# services/content_analysis/video_transcribe.py
import os
import ffmpeg
import requests
from pathlib import Path
import logging

class VideoTranscriber:
    def __init__(self, model_service_url="http://121.48.227.136:3000/transcribe"):
        self.model_service_url = model_service_url

    def extract_audio(self, video_path):
        """提取音频并返回临时路径"""
        # 确保使用完整路径
        video_path_obj = Path(video_path)
        audio_path = str(video_path_obj.parent / f"{video_path_obj.stem}_audio.wav")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, format='wav', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 验证音频文件是否成功创建
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                return audio_path
            else:
                return None
        except ffmpeg.Error as e:
            # 尝试使用子进程直接调用ffmpeg
            try:
                import subprocess
                cmd = f"ffmpeg -i '{video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 '{audio_path}' -y"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    return audio_path
            except Exception:
                pass
            return None

    def transcribe_video(self, video_path):
        """
        转录视频内容，返回带时间戳的文本
        处理无音频的视频文件
        """
        # 提取音频
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            # 检查视频是否有音频轨道
            try:
                import subprocess
                cmd = f"ffprobe -i '{video_path}' -show_streams -select_streams a -loglevel error"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if not result.stdout.strip():
                    # 返回一个空的转录结果而不是None
                    return {
                        "text": "",
                        "chunks": [],
                        "duration": 0,
                        "message": "视频不包含音频轨道"
                    }
            except Exception:
                pass
            return None