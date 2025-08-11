import json
from pathlib import Path
import re
import time
import requests
from openai import OpenAI
import logging


class SummaryGenerator:
    def __init__(self, config=None):
        """初始化摘要生成器

        Args:
            config: 可选的配置参数，如果不提供则从默认位置加载
        """
        # 如果没有提供配置，则从默认位置加载
        if config is None:
            self.config = self._load_config()
        else:
            self.config = config

        self.use_local = self.config["local_report"]

        # 初始化客户端
        if not self.use_local:
            self.client = OpenAI(
                api_key=self.config["report_model"]["remote_openai"]["api_key"],
                base_url=self.config["report_model"]["remote_openai"]["base_url"],
            )

        # 公共参数
        self.max_retries = self.config["retry"]["max_retries"]
        self.retry_delay = self.config["retry"]["retry_delay"]
        self.system_prompt = self._build_system_prompt()

    def _load_config(self):
        config_path = Path(__file__).parent / "config" / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在于：{config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
                # 处理BOM头
                if raw_content.startswith("\ufeff"):
                    raw_content = raw_content[1:]
                return json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误：{e.doc}")
        except Exception as e:
            raise RuntimeError(f"配置加载失败：{str(e)}")

    def _build_system_prompt(self):
        return """
你是一个专业的内容摘要生成器。请根据提供的文本内容和已提取的信息，生成一个简洁、全面的内容摘要。

摘要要求：
1. 摘要中必须明确包含已提取的意图和关键陈述
2. 总结内容的主题和核心观点
3. 保持客观、准确的表达
4. 摘要长度控制在500字以内
5. 清晰的格式进行排版
6. 使用以下emoji图标或文字标签来表示不同的内容类型：
正确、强调
	
✅、🌟、📌
警告、注意
	
⚠️、❗
错误、缺点
	
❌、💢
总结、结论
	
📝、🔚、🏁
目标、用途
	
🎯
提示、技巧
	
、🔍
项目分类
	
🧩、📁、📦


内容结构应包含：
1. 主题说明：简述内容的主要话题和目的
2. 意图分析：基于已提取的意图进行表述
3. 核心陈述：基于已提取的关键陈述进行概括
4. 整体评价：对内容进行总体性描述

输出格式示例：
## 📝内容摘要

### 主题
[简要描述内容的主题]

### 意图
[列出内容的主要意图]

### 💡核心内容
- [关键陈述1]
- [关键陈述2]
- [关键陈述3]

### 🔚总结
[对内容的整体评价和总结]
请确保摘要内容完整、准确，并保持与原文的一致性。 """

    def generate_summary(self, transcript_text, extracted_info, max_length=None):
        """
        生成内容摘要

        Args:
            transcript_text: 原始文本内容
            extracted_info: 已提取的信息（包含intent和statements）
            max_length: 可选的最大长度限制

        Returns:
            str: Markdown格式的摘要内容
        """
        # 组合用户输入，包含原始文本和已提取的信息
        user_input = self._build_user_input(transcript_text, extracted_info)

        for attempt in range(self.max_retries):
            try:
                if self.use_local:
                    summary = self._call_ollama_api(user_input)
                else:
                    summary = self._call_openai_api(user_input)

                # 应用长度限制（如果有）
                if max_length and len(summary) > max_length:
                    summary = summary[: max_length - 3] + "..."

                return summary

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logging.warning(
                        f"摘要生成失败: {str(e)}，{attempt+1}/{self.max_retries}次重试中..."
                    )
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logging.error(f"摘要生成失败: {str(e)}，已达最大重试次数")
                    return self._generate_fallback_summary(extracted_info)

    def _build_user_input(self, transcript_text, extracted_info):
        """构建发送给模型的用户输入"""
        # 提取意图和陈述
        intents = extracted_info.get("intent", [])
        statements = extracted_info.get("statements", [])

        # 格式化陈述
        formatted_statements = []
        for statement in statements:
            if isinstance(statement, dict) and "content" in statement:
                formatted_statements.append(statement["content"])

        # 构建提示
        prompt = f"""请根据以下内容生成摘要：
        原始文本
    {transcript_text[:1000]}... (文本已截断)

    已提取的信息
    意图: {", ".join(intents) if intents else "未识别"}

    关键陈述: {chr(10).join([f"- {s}" for s in formatted_statements])}

    请生成一个基于以上信息的Markdown格式摘要。确保摘要中明确包含已提取的意图和关键陈述。 """
        return prompt

    def _call_openai_api(self, text):
        """调用OpenAI API生成摘要"""
        response = self.client.chat.completions.create(
            model=self.config["report_model"]["remote_openai"]["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _call_ollama_api(self, text):
        """调用本地Ollama API生成摘要"""
        payload = {
            "model": self.config["report_model"]["local_ollama"]["model"],
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "temperature": 0.3,
        }

        response = requests.post(
            self.config["report_model"]["local_ollama"]["base_url"],
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = json.loads(response.text)
        return data["message"]["content"]

    def _generate_fallback_summary(self, extracted_info):
        """当API调用失败时生成备用摘要"""
        intents = extracted_info.get("intent", [])
        statements = extracted_info.get("statements", [])

        intent_str = "、".join(intents) if intents else "未识别"

        statement_bullets = []
        for statement in statements:
            if isinstance(statement, dict) and "content" in statement:
                statement_bullets.append(f"- {statement['content']}")

        statement_str = (
            "\n".join(statement_bullets) if statement_bullets else "- 无明确陈述"
        )

        fallback_summary = f"""## 内容摘要
        主题
    该内容的主题无法自动识别。

    意图
    {intent_str}

    核心内容
    {statement_str}

    总结
    由于技术原因，无法生成完整摘要，上述内容仅供参考。 """
        return fallback_summary
