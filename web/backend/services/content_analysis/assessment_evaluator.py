from pathlib import Path
import time
import json          # 新增导入
import requests      # 新增导入
from openai import OpenAI


class AssessmentEvaluator:
    def __init__(self, config=None):
        """初始化评估器
        
        Args:
            config: 可选的配置参数，如果不提供则从默认位置加载
        """
        # 如果没有提供配置，则从默认位置加载
        if config is None:
            self.config = self._load_config()
        else:
            self.config = config
            
        self.use_local = self.config["local_assessment"]
        if not self.use_local:
            self.client = OpenAI(
                api_key=self.config["assessment_model"]["remote_openai"]["api_key"],
                base_url=self.config["assessment_model"]["remote_openai"]["base_url"]
            )
        self.max_retries = self.config["retry"]["max_retries"]
        self.retry_delay = self.config["retry"]["retry_delay"]


    def _load_config(self):
        """从默认位置加载配置文件"""
        config_path = Path(__file__).parent / 'config' / 'config.json'
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在于：{config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                # 处理BOM头
                if raw_content.startswith('\ufeff'):
                    raw_content = raw_content[1:]
                return json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误：{e.doc}")
        except Exception as e:
            raise RuntimeError(f"配置加载失败：{str(e)}")

    def _call_ollama_api(self, system_prompt, user_prompt, temperature, top_k):
        payload = {
            "model": self.config["assessment_model"]["local_ollama"]["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "temperature": temperature,
            "top_k": top_k
        }
        response = requests.post(
            self.config["assessment_model"]["local_ollama"]["base_url"],
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = json.loads(response.text)
        return data["message"]["content"].strip()

    def _call_openai_api(self, system_prompt, user_prompt, temperature, top_k):
        response = self.client.chat.completions.create(
            model=self.config["assessment_model"]["remote_openai"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def _call_api(self, system_prompt, user_prompt, temperature=0, top_k=40):
        for attempt in range(self.max_retries + 1):
            try:
                if self.use_local:
                    output_str = self._call_ollama_api(
                        system_prompt, user_prompt, temperature, top_k)
                else:
                    output_str = self._call_openai_api(
                        system_prompt, user_prompt, temperature, top_k)
                if "</think>" in output_str:
                    output_str = output_str.split("</think>")[-1].strip()
                return float(output_str)
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"API调用失败: {str(e)}")
                    return None
                time.sleep(self.retry_delay * (attempt + 1))
        return None

    # 以下为各评估指标的具体实现
    def p1_assessment(self, message, temperature=0.2, top_k=40):
        """背景信息充分性评估"""
        system_prompt = """你是一个消息背景信息评估助手。你的任务是根据提供的消息判断该消息是否包含足够的背景信息。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间，数值越接近1表示背景信息越充分
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0"""
        user_prompt = f"请对以下消息进行判断，是否包含足够的背景信息：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k)

    def p2_assessment(self, message, temperature=0.2, top_k=40):
        """背景信息准确性评估"""
        system_prompt = """你是一个消息背景信息准确性评估助手。你的任务是根据提供的消息判断其中的背景信息是否准确且客观。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间，数值越接近1表示背景信息越准确客观
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0"""
        user_prompt = f"请对以下消息进行判断，背景信息是否准确且客观：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k)

    def p3_assessment(self, message, temperature=0.2, top_k=40):
        """内容完整性评估"""
        system_prompt = """你是一个消息完整性评估助手。你的任务是根据提供的消息判断是否有内容被故意删除而导致意思被歪曲。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间，数值越接近1表示内容完整且没有故意删除
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0"""
        user_prompt = f"请对以下消息进行判断，是否存在内容被故意删除而导致意思被歪曲：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k)

    def p4_assessment(self, message, intent, temperature=0.2, top_k=40):
        """不当意图评估"""
        system_prompt = """你是一个消息意图评估助手。你的任务是根据提供的消息及其意图判断消息中是否存在不当意图（例如政治动机、商业目的等）。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间，数值越接近1表示不当意图可能性越低
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0"""
        user_prompt = f"请对以下消息进行判断，是否存在不当意图：\n消息: {message}\n意图: {intent}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k)

    def p5_assessment(self, reputation=0.5, temperature=0.2, top_k=40):
        """发布者历史评估"""
        system_prompt = """你是一个发布者历史记录评估助手。你的任务是根据提供的发布者信誉信息判断其是否有发布带有不当意图信息的历史记录。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间，数值越接近1表示发布者历史记录较好
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0"""
        user_prompt = f"请对下面的发布者信誉信息进行判断：\n发布者信誉: {reputation}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k)

    def p6_assessment(self, message, temperature=0.2, top_k=40):
        """情感煽动性评估"""
        system_prompt = """你是一个情感煽动性评估助手。评估消息是否使用煽情语言操纵观众情绪。遵守规则：
        - 返回0-1的小数，1表示情感中立
        - 只返回数字，无其他内容
        - 无法评估返回0"""
        user_prompt = f"评估以下消息的情感煽动性：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k)

    def p7_assessment(self, message, temperature=0.2, top_k=40):
        """诱导行为评估"""
        system_prompt = """你是一个行为诱导评估助手。检测消息是否包含诱导点赞/转发/消费等内容。遵守规则：
        - 返回0-1的小数，1表示无诱导行为
        - 只返回数字
        - 无法评估返回0"""
        user_prompt = f"检测以下消息是否包含诱导行为：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k)

    def p8_assessment(self, statements, temperature=0.2, top_k=40):
        """信息一致性评估"""
        system_prompt = """你是信息一致性评估助手。判断多条声明内容是否存在自相矛盾。规则：
        - 返回0-1的小数，1表示完全一致
        - 只返回数字
        - 无法评估返回0"""
        statements_text = "\n".join([s["content"] for s in statements])
        user_prompt = f"判断以下声明的信息一致性：\n{statements_text}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k)
