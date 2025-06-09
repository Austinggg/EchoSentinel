import json
import re
import time
import logging
import requests
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

class AnalysisReporter:
    def __init__(self, config=None, feature_map=None):
        """初始化分析报告生成器
        
        Args:
            config: 可选的配置参数，如果不提供则从默认位置加载
            feature_map: 特征映射字典，用于转换特征ID到可读名称
        """
        # 如果没有提供配置，则从默认位置加载
        if config is None:
            self.config = self._load_config()
        else:
            self.config = config
            
        # 如果没有提供特征映射，则使用默认映射
        if feature_map is None:
            self.feature_map = {
                1: "背景信息充分性",
                2: "背景信息准确性",
                3: "内容完整性",
                4: "意图正当性", 
                5: "发布者信誉",
                6: "情感中立性",
                7: "行为自主性",
                8: "信息一致性"
            }
        else:
            self.feature_map = feature_map
        
        self.use_local = self.config.get("local_report", False)
        
        # 初始化客户端
        if not self.use_local:
            try:
                self.client = OpenAI(
                    api_key=self.config["report_model"]["remote_openai"]["api_key"],
                    base_url=self.config["report_model"]["remote_openai"]["base_url"]
                )
            except Exception as e:
                logger.error(f"初始化OpenAI客户端失败: {str(e)}")
                self.use_local = True  # 回退到本地模式
                logger.warning("回退到本地模式")
        
        # 公共参数
        self.max_retries = self.config["retry"]["max_retries"]
        self.retry_delay = self.config["retry"]["retry_delay"]
        self.system_prompt = self._build_system_prompt()
        logger.info(f"分析报告生成器初始化完成，使用{'本地' if self.use_local else '远程'}模型")

    def _load_config(self):
        """从默认位置加载配置文件"""
        config_path = Path(__file__).parent / 'config' / 'config.json'
        
        if not config_path.exists():
            logger.error(f"配置文件不存在于：{config_path}")
            raise FileNotFoundError(f"配置文件不存在于：{config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                # 处理BOM头
                if raw_content.startswith('\ufeff'):
                    raw_content = raw_content[1:]
                return json.loads(raw_content)
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误：{e}")
            raise ValueError(f"配置文件格式错误：{e}")
        except Exception as e:
            logger.error(f"配置加载失败：{str(e)}")
            raise RuntimeError(f"配置加载失败：{str(e)}")

    def _build_system_prompt(self):
        """构建系统提示"""
        return """您是一个专业的信息验证助手，请根据以下要素生成验证报告：
1. 结合技术规则分析结果
2. 解释关键评估指标异常
3. 保持专业但易懂的文风
4. 包含结论和建议

请使用以下Markdown格式：
## 结论
## 主要依据
## 详细分析
## 建议行动

请特别注意：
- 用▲标注重点内容
- 技术术语需括号解释
- 得分低于0.6的指标需特别标注
"""

    def generate_report(self, data_instance, dnf_rules):
        """
        生成可验证性报告
        :param data_instance: 数据实例(dict)
        :param dnf_rules: DNF规则输出(dict)
        :return: 自然语言报告(str)
        """
        for attempt in range(self.max_retries):
            try:
                user_prompt = self._build_user_prompt(data_instance, dnf_rules)
                logger.info(f"尝试生成报告，尝试次数: {attempt+1}/{self.max_retries}")
                
                if self.use_local:
                    return self._call_local_api(user_prompt)
                else:
                    return self._call_remote_api(user_prompt)
                    
            except Exception as e:
                logger.warning(f"报告生成失败: {str(e)}，尝试次数: {attempt+1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"报告生成失败: {str(e)}，已达最大重试次数")
                    return None

    def _build_user_prompt(self, data_instance, dnf_rules):
        """构建用户提示"""
        # 构建关键指标分析
        key_factors = []
        for p in range(1, 9):
            score = data_instance['general'][f'P{p}']
            if score < 0.6:
                key_factors.append(f"{self.feature_map[p]}不足（{score:.2f}）")
            elif score > 0.9:
                key_factors.append(f"{self.feature_map[p]}异常（{score:.2f}）")

        # 构建DNF规则摘要
        try:
            rule_summary = "\n".join([
                f"- {self._explain_rule(rule)}" 
                for rule in dnf_rules.get('disjuncts', {}).values()
                if "∅" not in rule
            ][:3])
        except Exception as e:
            logger.warning(f"解析DNF规则失败: {str(e)}")
            rule_summary = "无法解析规则"

        return f"""
## 待验证信息
{data_instance.get('message', '')[:2000]}

## 技术分析结果
{'\n'.join(key_factors)}

## DNF决策规则
{rule_summary}

## 生成要求
1. 用▲标注关键风险点
2. 得分低于0.6的指标需优先分析
3. 结合决策规则说明判定依据
"""

    def _explain_rule(self, rule: str) -> str:
        """规则解释器"""
        explanations = []
        try:
            for term in re.split(r'[∨∧]', rule):
                term = term.strip()
                if "P" in term:
                    sign = "非" if "¬" in term else ""
                    p_num = term.split("P")[-1]
                    explanations.append(f"{sign}{self.feature_map.get(int(p_num), '未知指标')}")
                elif "conj" in term:
                    explanations.append(f"组合规则{term.replace('conj','')}")
            return " + ".join(explanations)
        except Exception as e:
            logger.warning(f"解释规则失败: {str(e)}")
            return "规则解释失败"

    def _call_remote_api(self, user_prompt):
        """调用远程API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config["report_model"]["remote_openai"]["model"],
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"调用远程API失败: {str(e)}")
            raise

    def _call_local_api(self, user_prompt):
        """调用本地API"""
        try:
            payload = {
                "model": self.config["report_model"]["local_ollama"]["model"],
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "temperature": 0.3
            }
            
            api_url = self.config["report_model"]["local_ollama"]["base_url"]
            logger.info(f"调用本地API: {api_url}")
            
            response = requests.post(
                api_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return self._parse_response(response.json()["message"]["content"])
        except requests.exceptions.RequestException as e:
            logger.error(f"本地API请求异常: {str(e)}")
            raise
        except KeyError as e:
            logger.error(f"解析本地API响应失败, 缺少键值: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"调用本地API时发生未知错误: {str(e)}")
            raise

    def _parse_response(self, response_text):
        """统一响应解析"""
        try:
            # 提取Markdown格式内容
            cleaned = re.sub(r"```markdown\s*", "", response_text)
            cleaned = re.sub(r"```\s*", "", cleaned)
            return cleaned.strip()
        except Exception as e:
            logger.error(f"响应解析异常: {str(e)}")
            return "报告生成失败，请查看原始分析数据"