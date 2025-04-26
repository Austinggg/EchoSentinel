

# services/content_analysis/video_transcribe.py
import os
import ffmpeg
import requests
from pathlib import Path
import logging
import re

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
            # 记录错误但不使用子进程
            logging.error(f"音频提取失败: {str(e)}")
            return None
    def extract_subtitles(self, video_path):
        """从视频文件中提取内嵌字幕"""
        video_path_obj = Path(video_path)
        subtitle_path = str(video_path_obj.parent / f"{video_path_obj.stem}_subtitles.srt")
        
        try:
            # 检查视频是否包含字幕轨道
            probe = ffmpeg.probe(video_path)
            subtitle_streams = [
                (i, stream) for i, stream in enumerate(probe['streams']) 
                if stream.get('codec_type') == 'subtitle'
            ]
            
            if not subtitle_streams:
                logging.info(f"视频不包含字幕轨道: {video_path}")
                return None
                
            # 提取第一个字幕轨道
            subtitle_index = subtitle_streams[0][0]
            
            # 提取字幕到SRT文件
            (
                ffmpeg
                .input(video_path)
                .output(subtitle_path, map=f"0:{subtitle_index}", c='copy')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 检查字幕文件是否成功创建
            if not os.path.exists(subtitle_path) or os.path.getsize(subtitle_path) == 0:
                logging.warning(f"字幕提取失败或文件为空: {subtitle_path}")
                return None
                
            # 解析SRT文件为转录格式
            chunks = self._parse_srt_file(subtitle_path)
            
            # 合并文本
            full_text = " ".join([chunk["text"] for chunk in chunks])
            
            # 清理临时文件
            os.remove(subtitle_path)
            
            if chunks:
                return {
                    "text": full_text,
                    "chunks": chunks,
                    "source": "embedded_subtitles"
                }
            
            return None
            
        except Exception as e:
            logging.error(f"提取字幕时发生错误: {str(e)}")
            # 清理临时文件
            if os.path.exists(subtitle_path):
                os.remove(subtitle_path)
            return None
    
    def _parse_srt_file(self, srt_path):
        """解析SRT字幕文件为chunks格式"""
        chunks = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                
            # 分割SRT条目
            entries = re.split(r'\n\s*\n', content.strip())
            
            for entry in entries:
                lines = entry.strip().split('\n')
                if len(lines) < 3:
                    continue
                    
                # 跳过序号行，直接处理时间行
                time_line = None
                for line in lines:
                    if '-->' in line:
                        time_line = line
                        break
                        
                if not time_line:
                    continue
                    
                # 解析时间
                time_parts = time_line.split(' --> ')
                start_time = self._time_to_seconds(time_parts[0])
                end_time = self._time_to_seconds(time_parts[1])
                
                # 获取文本内容（可能跨多行）
                text_start_idx = lines.index(time_line) + 1
                text = ' '.join(lines[text_start_idx:]).strip()
                
                chunks.append({
                    "text": text,
                    "timestamp": [start_time, end_time]
                })
                
            return chunks
            
        except Exception as e:
            logging.error(f"解析SRT文件失败: {str(e)}")
            return []
    
    def _time_to_seconds(self, time_str):
        """将SRT时间格式(00:00:00,000)转换为秒"""
        time_str = time_str.replace(',', '.')
        h, m, s = time_str.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)

    # 原有方法保持不变

    def transcribe_video(self, video_path):
        """转录视频内容，优先使用内嵌字幕，否则提取音频进行转录"""
        # 首先尝试提取内嵌字幕
        subtitle_result = self.extract_subtitles(video_path)
        if subtitle_result:
            logging.info(f"成功提取内嵌字幕: {video_path}")
            return subtitle_result
            
        # 如果没有内嵌字幕，继续原有的音频转录流程
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            # 检查视频是否有音频轨道
            try:
                probe = ffmpeg.probe(video_path)
                audio_streams = [stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'audio']
                
                if not audio_streams:
                    return {
                        "text": "",
                        "chunks": [],
                        "duration": 0,
                        "message": "视频不包含音频轨道或字幕"
                    }
            except Exception as e:
                logging.error(f"检查音频轨道失败: {str(e)}")
                return None
            
            return None
        
        # 原有的模型服务调用代码
        try:
            with open(audio_path, "rb") as f:
                files = {"audio": f}
                
                response = requests.post(
                    self.model_service_url,
                    files=files,
                    timeout=300
                )
                
                response.raise_for_status()
                result = response.json()
                
                os.remove(audio_path)
                return result
                
        except Exception as e:
            logging.error(f"调用转录服务失败: {str(e)}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None