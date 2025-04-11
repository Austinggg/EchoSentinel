import json
import re
import time
import requests
from openai import OpenAI

class VeracityReporter:
    def __init__(self, config, feature_map):
        self.config = config
        self.feature_map = feature_map
        self.use_local = config["local_report"]
        
        # 初始化客户端
        if not self.use_local:
            self.client = OpenAI(
                api_key=config["report_model"]["remote_openai"]["api_key"],
                base_url=config["report_model"]["remote_openai"]["base_url"]
            )
        
        # 公共参数
        self.max_retries = config["retry"]["max_retries"]
        self.retry_delay = config["retry"]["retry_delay"]
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
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
                
                if self.use_local:
                    return self._call_local_api(user_prompt)
                else:
                    return self._call_remote_api(user_prompt)
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"报告生成失败: {str(e)}，{attempt+1}/{self.max_retries}次重试...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"报告生成失败: {str(e)}，已达最大重试次数")
                    return None

    def _build_user_prompt(self, data_instance, dnf_rules):
        # 构建关键指标分析
        key_factors = []
        for p in range(1, 9):
            score = data_instance['general'][f'P{p}']
            if score < 0.6:
                key_factors.append(f"{self.feature_map[p]}不足（{score:.2f}）")
            elif score > 0.9:
                key_factors.append(f"{self.feature_map[p]}异常（{score:.2f}）")

        # 构建DNF规则摘要
        rule_summary = "\n".join([
            f"- {self._explain_rule(rule)}" 
            for rule in dnf_rules['disjuncts'].values()
            if "∅" not in rule
        ][:3])

        return f"""
## 待验证信息
{data_instance['message'][:2000]}

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
        for term in re.split(r'[∨∧]', rule):
            term = term.strip()
            if "P" in term:
                sign = "非" if "¬" in term else ""
                p_num = term.split("P")[-1]
                explanations.append(f"{sign}{self.feature_map.get(int(p_num), '未知指标')}")
            elif "conj" in term:
                explanations.append(f"组合规则{term.replace('conj','')}")
        return " + ".join(explanations)

    def _call_remote_api(self, user_prompt):
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

    def _call_local_api(self, user_prompt):
        payload = {
            "model": self.config["report_model"]["local_ollama"]["model"],
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "temperature": 0.3
        }
        
        response = requests.post(
            self.config["report_model"]["local_ollama"]["base_url"],
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return self._parse_response(response.json()["message"]["content"])

    def _parse_response(self, response_text):
        """统一响应解析"""
        try:
            # 提取Markdown格式内容
            cleaned = re.sub(r"```markdown\s*", "", response_text)
            cleaned = re.sub(r"```\s*", "", cleaned)
            return cleaned.strip()
        except Exception as e:
            print(f"响应解析异常: {str(e)}")
            return "报告生成失败，请查看原始分析数据"