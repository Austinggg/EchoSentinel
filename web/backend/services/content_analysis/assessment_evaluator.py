import logging
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

    def _call_openai_api(self, system_prompt, user_prompt, temperature, top_k, include_reasoning=False):
        """调用OpenAI API，可选择包含reasoner字段"""
        try:
            # 添加API参数获取reasoner信息
            response = self.client.chat.completions.create(
                model=self.config["assessment_model"]["remote_openai"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                # 添加可能的参数以获取reasoning内容
                extra_body={
                    "return_reasoning": True
                }
            )
            
            # 提取内容
            content = response.choices[0].message.content.strip()
            
            # 正确检查推理内容位置 - 修改这里
            if include_reasoning and hasattr(response.choices[0].message, "reasoning_content"):
                return {
                    "score": float(content),
                    "reasoning_content": response.choices[0].message.reasoning_content
                }
            
            # 备用检查 - 可能的其他API返回格式
            if include_reasoning and hasattr(response.choices[0], "reasoning_content"):
                return {
                    "score": float(content),
                    "reasoning_content": response.choices[0].reasoning_content
                }
            
            # 另一种可能的格式检查
            if include_reasoning and hasattr(response, "reasoning"):
                return {
                    "score": float(content),
                    "reasoning_content": response.reasoning
                }
                
            # 默认值
            if include_reasoning:
                return {
                    "score": float(content),
                    "reasoning_content": "API未返回推理过程"
                }
            else:
                return content
            
        except Exception as e:
            logging.error(f"OpenAI API调用失败: {str(e)}")
            raise
    def _call_api(self, system_prompt, user_prompt, temperature=0, top_k=40, include_reasoning=False):
        """调用API并处理响应"""
        for attempt in range(self.max_retries + 1):
            try:
                if self.use_local:
                    # 本地API不支持reasoning
                    output_str = self._call_ollama_api(
                        system_prompt, user_prompt, temperature, top_k)
                    
                    # 尝试转换为浮点数
                    try:
                        score = float(output_str)
                        if include_reasoning:
                            return {
                                "score": score,
                                "reasoning_content": "本地模型未提供推理过程"
                            }
                        else:
                            return score
                    except ValueError:
                        # 无法转换，处理特殊输出
                        if "</think>" in output_str:
                            parts = output_str.split("</think>")
                            thinking = parts[0].replace("<think>", "").strip()
                            result = parts[1].strip()
                            
                            try:
                                score = float(result)
                                if include_reasoning:
                                    return {
                                        "score": score,
                                        "reasoning_content": thinking
                                    }
                                else:
                                    return score
                            except:
                                logging.warning(f"无法解析数字结果: {result}")
                                return None
                        else:
                            logging.warning(f"无法解析结果: {output_str}")
                            return None
                else:
                    # 使用OpenAI API
                    result = self._call_openai_api(
                        system_prompt, user_prompt, temperature, top_k, include_reasoning)
                    
                    # 修改这里 - 检查返回结果类型和字段
                    if isinstance(result, dict):
                        # 检查字典中具有哪些字段
                        if "score" in result:
                            # 正确格式，直接返回
                            return result
                        elif "content" in result:
                            # 旧格式，需要转换
                            try:
                                score = float(result["content"])
                                if include_reasoning:
                                    return {
                                        "score": score,
                                        "reasoning_content": result.get("reasoning_content", "未提供推理过程")
                                    }
                                else:
                                    return score
                            except ValueError:
                                logging.warning(f"无法解析数字结果: {result['content']}")
                                if include_reasoning:
                                    return {"score": None, "reasoning_content": f"无法解析的结果: {result['content']}"}
                                else:
                                    return None
                        else:
                            # 未知的字典格式
                            logging.warning(f"未知的结果格式: {result}")
                            if include_reasoning:
                                return {"score": None, "reasoning_content": "API返回了未知格式的结果"}
                            else:
                                return None
                    else:
                        # 直接返回的是字符串或其他类型
                        try:
                            score = float(result) if result is not None else None
                            if include_reasoning:
                                return {
                                    "score": score,
                                    "reasoning_content": "API未返回推理过程"
                                }
                            else:
                                return score
                        except (ValueError, TypeError):
                            logging.warning(f"无法解析数字结果: {result}")
                            if include_reasoning:
                                return {"score": None, "reasoning_content": f"无法解析的结果: {result}"}
                            else:
                                return None
            except Exception as e:
                if attempt == self.max_retries:
                    logging.error(f"API调用失败: {str(e)}")
                    if include_reasoning:
                        return {"score": None, "reasoning_content": f"错误: {str(e)}"}
                    else:
                        return None
                time.sleep(self.retry_delay * (attempt + 1))
        
        if include_reasoning:
            return {"score": None, "reasoning_content": "达到最大重试次数后仍然失败"}
        else:
            return None

    # 以下为各评估指标的具体实现
    def p1_assessment(self, message, temperature=0.2, top_k=40, include_reasoning=False):
        """背景信息充分性评估"""
        system_prompt = """你是一个消息背景信息评估助手。你的任务是根据提供的消息判断该消息是否包含足够的背景信息。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间
        - 1表示背景信息非常充分
        - 0表示背景信息严重不足
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0.5"""
        user_prompt = f"请对以下消息进行判断，是否包含足够的背景信息：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k, include_reasoning=include_reasoning)

    def p2_assessment(self, message, temperature=0.2, top_k=40, include_reasoning=False):
        """背景信息准确性评估"""
        system_prompt = """你是一个消息背景信息准确性评估助手。你的任务是根据提供的消息判断其中的背景信息是否准确且客观。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间
        - 1表示背景信息完全准确客观
        - 0表示背景信息严重不准确或有明显偏见
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0.5"""
        user_prompt = f"请对以下消息进行判断，背景信息是否准确且客观：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k, include_reasoning=include_reasoning)

    def p3_assessment(self, message, temperature=0.2, top_k=40, include_reasoning=False):
        """内容完整性评估"""
        system_prompt = """你是一个消息完整性评估助手。你的任务是根据提供的消息判断是否有内容被故意删除而导致意思被歪曲。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间
        - 1表示内容完全完整，没有任何删减导致的歪曲
        - 0表示内容有严重删减，导致意思被完全歪曲
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0.5"""
        user_prompt = f"请对以下消息进行判断，是否存在内容被故意删除而导致意思被歪曲：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k, include_reasoning=include_reasoning)

    def p4_assessment(self, message, intent, temperature=0.2, top_k=40, include_reasoning=False):
        """意图正当性评估"""
        system_prompt = """你是一个消息意图正当性评估助手。你的任务是根据提供的消息及其意图判断消息中的意图是否正当。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间
        - 1表示意图完全正当合理
        - 0表示意图完全不当（如有害政治动机、不当商业目的等）
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0.5"""
        user_prompt = f"请对以下消息进行判断，其意图是否正当：\n消息: {message}\n意图: {intent}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k, include_reasoning=include_reasoning)

    def p5_assessment(self, reputation=0.5, temperature=0.2, top_k=40, include_reasoning=False):
        """发布者信誉评估"""
        system_prompt = """你是一个发布者信誉评估助手。你的任务是根据提供的发布者信誉信息判断其是否值得信任。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间
        - 1表示发布者信誉极佳，非常值得信任
        - 0表示发布者信誉极差，完全不值得信任
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回输入的信誉值"""
        user_prompt = f"请对下面的发布者信誉信息进行判断：\n发布者信誉: {reputation}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k, include_reasoning=include_reasoning)

    def p6_assessment(self, message, temperature=0.2, top_k=40, include_reasoning=False):
        """情感中立性评估"""
        system_prompt = """你是一个情感中立性评估助手。你的任务是评估消息是否使用煽情语言操纵观众情绪。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间
        - 1表示内容完全情感中立，没有煽动性
        - 0表示内容极度煽动情绪，明显操纵观众
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0.5"""
        user_prompt = f"评估以下消息的情感中立性：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k, include_reasoning=include_reasoning)

    def p7_assessment(self, message, temperature=0.2, top_k=40, include_reasoning=False):
        """行为自主性评估"""
        system_prompt = """你是一个行为自主性评估助手。你的任务是检测消息是否包含诱导点赞/转发/消费等内容。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间
        - 1表示内容完全不含诱导行为，尊重用户自主选择
        - 0表示内容强烈诱导特定行为，严重干扰用户自主判断
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0.5"""
        user_prompt = f"检测以下消息是否尊重用户行为自主性：\n消息: {message}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k, include_reasoning=include_reasoning)

    def p8_assessment(self, statements, temperature=0.2, top_k=40, include_reasoning=False):
        """信息一致性评估"""
        system_prompt = """你是信息一致性评估助手。你的任务是判断多条声明内容是否存在自相矛盾。请严格遵守以下规则：
        - 只返回一个小数，范围在0到1之间
        - 1表示所有信息完全一致，没有任何矛盾
        - 0表示信息严重矛盾，存在明显冲突
        - 不要返回任何其他文字说明或解释
        - 如果无法评估，请返回0.5"""
        statements_text = "\n".join([s["content"] for s in statements])
        user_prompt = f"判断以下声明的信息一致性：\n{statements_text}"
        return self._call_api(system_prompt, user_prompt, temperature=temperature, top_k=top_k, include_reasoning=include_reasoning)