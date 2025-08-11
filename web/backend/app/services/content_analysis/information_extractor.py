# info_extractor.py
import json
from pathlib import Path
import re
import time
import requests
from openai import OpenAI

class InformationExtractor:
    def __init__(self, config=None):
        """初始化信息提取器
        
        Args:
            config: 可选的配置参数，如果不提供则从默认位置加载
        """
        # 如果没有提供配置，则从默认位置加载
        if config is None:
            self.config = self._load_config()
        else:
            self.config = config
            
        self.use_local = self.config["local_information_extraction"]
        
        # 初始化客户端
        if not self.use_local:
            self.client = OpenAI(
                api_key=self.config["information_extraction_model"]["remote_openai"]["api_key"],
                base_url=self.config["information_extraction_model"]["remote_openai"]["base_url"]
            )
        
        # 公共参数
        self.max_retries = self.config["retry"]["max_retries"]
        self.retry_delay = self.config["retry"]["retry_delay"]
        self.system_prompt = self._build_system_prompt()
    def _load_config(self):
        config_path = Path(__file__).parent / 'config' / 'config.json'
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在于：{config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                # 处理BOM头（参考网页9编码问题）
                if raw_content.startswith('\ufeff'):
                    raw_content = raw_content[1:]
                return json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误：{e.doc}")
        except Exception as e:
            raise RuntimeError(f"配置加载失败：{str(e)}")
    def _build_system_prompt(self):
        return """
请严格按JSON格式输出，仅分析以下内容，其他字段键值保持不变，按照原来格式返回：

信息提取任务：
1. INTENT (意图)：判断消息的主要目的（从以下选项选择，可多选需用[]标注）
   可选类型：新闻资讯、虚假信息、侵权内容、营销推广、知识科普、个人观点、公共服务信息、社会警示；
   恶意攻击/骚扰、色情/暴力内容、煽动性言论、垃圾广告、欺诈诱导、隐私侵犯、威胁恐吓、其他

2. STATEMENTS (关键陈述)：提取需要验证的1-5个核心事实主张，严格按重要性降序排列

输出结构说明：
- statements数组长度根据实际内容动态生成（1-5条）
- 每个陈述必须独立且互不重复
- 超过5个时只保留最重要的前5个
'''json
{
"intent": [],
"statements": [
    {"id": 1, "content": ""},
    {"id": 2, "content": ""},
    {"id": 3, "content": ""}
    ......  # 最多5个陈述
]
}
"""

    def extract_information(self, text):
        for attempt in range(self.max_retries):
            try:
                if self.use_local:
                    return self._call_ollama_api(text)
                else:
                    return self._call_openai_api(text)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API调用失败: {str(e)}，{attempt+1}/{self.max_retries}次重试中...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"API调用失败: {str(e)}，已达最大重试次数")
                    return None

    def _call_openai_api(self, text):
        response = self.client.chat.completions.create(
            model=self.config["information_extraction_model"]["remote_openai"]["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )
        return self._parse_response(response.choices[0].message.content.strip())

    def _call_ollama_api(self, text):
        payload = {
            "model": self.config["information_extraction_model"]["local_ollama"]["model"],
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            "stream": False,
            "temperature": 0.2,
            "top_k": 40
        }
        
        response = requests.post(
            self.config["information_extraction_model"]["local_ollama"]["base_url"],
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        data = json.loads(response.text)
        return self._parse_response(data["message"]["content"])

    def _parse_response(self, response_text):
        try:
            # 统一处理不同API返回的JSON格式
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            
            result = json.loads(json_str)
            return {
                "intent": result.get("intent", []),
                "statements": result.get("statements", [])
            }
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"响应解析失败: {str(e)}")
            print(f"原始响应内容: {response_text}")
            return None
        
