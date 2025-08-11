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
        return """您是一个专业的内容威胁分析专家，需要基于多维度检测数据生成威胁分析报告。

报告要求：
1. 使用▲标注所有高风险点
2. 采用客观、专业的分析语言
3. 提供具体的威胁评估和建议
4. 结合内容分析、数字人检测、事实核查等多维度数据

请严格按照以下Markdown格式输出：

## 威胁等级评估
[基于综合分析的威胁等级判定]

## 关键发现
[列出主要威胁点，用▲标注高风险项]

## 详细分析
### 内容可信度分析
[基于八项指标的内容分析]

### 真实性检测
[如有数字人检测结果，分析视频真实性]

### 事实准确性
[如有事实核查结果，分析内容准确性]

## 风险评分
[各维度评分说明]

## 处置建议
[针对性的处理建议]

注意事项：
- 所有评分低于0.6的指标必须用▲标注
- 数字人检测AI概率>0.7时用▲标注
- 事实核查发现虚假信息时用▲标注
- 保持专业客观的分析语调"""

    def generate_comprehensive_report(self, video_id, data_instance, dnf_rules, digital_human_data=None, fact_check_data=None):
        """
        生成综合威胁分析报告
        :param video_id: 视频ID
        :param data_instance: 内容分析数据
        :param dnf_rules: DNF规则
        :param digital_human_data: 数字人检测数据（可选）
        :param fact_check_data: 事实核查数据（可选）
        :return: 综合分析报告
        """
        for attempt in range(self.max_retries):
            try:
                user_prompt = self._build_comprehensive_prompt(
                    video_id, data_instance, dnf_rules, digital_human_data, fact_check_data
                )
                logger.info(f"生成综合威胁分析报告，视频ID: {video_id}，尝试次数: {attempt+1}/{self.max_retries}")
                
                if self.use_local:
                    return self._call_local_api(user_prompt)
                else:
                    return self._call_remote_api(user_prompt)
                    
            except Exception as e:
                logger.warning(f"综合报告生成失败: {str(e)}，尝试次数: {attempt+1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"综合报告生成失败: {str(e)}，已达最大重试次数")
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

    def _build_comprehensive_prompt(self, video_id, data_instance, dnf_rules, digital_human_data=None, fact_check_data=None):
        """构建综合分析提示"""
        
        # 内容分析部分
        content_analysis = self._analyze_content_risks(data_instance)
        
        # 数字人检测分析
        digital_human_analysis = ""
        if digital_human_data and digital_human_data.get('status') == 'completed':
            digital_human_analysis = self._analyze_digital_human_risks(digital_human_data)
        
        # 事实核查分析
        fact_check_analysis = ""
        if fact_check_data and fact_check_data.get('status') == 'completed':
            fact_check_analysis = self._analyze_fact_check_risks(fact_check_data)
        
        # DNF规则摘要
        rule_summary = self._build_rule_summary(dnf_rules)
        
        return f"""## 视频威胁分析任务
**视频ID**: {video_id}
**分析内容**: {data_instance.get('message', '')[:1500]}

## 内容可信度分析
{content_analysis}

## 真实性检测结果
{digital_human_analysis if digital_human_analysis else "未进行数字人检测"}

## 事实准确性检查
{fact_check_analysis if fact_check_analysis else "未进行事实核查"}

## 决策规则依据
{rule_summary}

## 分析要求
1. 综合以上所有维度数据进行威胁评估
2. 用▲明确标注所有高风险点
3. 为每个维度的异常情况提供具体解释
4. 给出明确的威胁等级和处置建议
5. 如某些检测未进行，在报告中说明其对整体评估的影响"""

    def _analyze_content_risks(self, data_instance):
        """分析内容风险"""
        risk_items = []
        scores = data_instance['general']
        
        for p in range(1, 9):
            score = scores[f'P{p}']
            feature_name = self.feature_map[p]
            
            if score < 0.4:
                risk_items.append(f"▲ {feature_name}严重不足（{score:.2f}）")
            elif score < 0.6:
                risk_items.append(f"▲ {feature_name}不足（{score:.2f}）")
            elif score > 0.9:
                risk_items.append(f"• {feature_name}异常偏高（{score:.2f}）")
            else:
                risk_items.append(f"• {feature_name}正常（{score:.2f}）")
        
        return "\n".join(risk_items)

    def _analyze_digital_human_risks(self, digital_human_data):
        """分析数字人检测风险"""
        detection = digital_human_data.get('detection', {})
        summary = digital_human_data.get('summary', {})
        
        analysis_parts = []
        
        # 综合评估
        if summary.get('final_prediction'):
            prediction = summary['final_prediction']
            ai_prob = summary.get('ai_probability', 0)
            confidence = summary.get('confidence', 0)
            
            if prediction == "AI-Generated" and ai_prob > 0.7:
                analysis_parts.append(f"▲ 检测为AI生成内容，置信度{confidence:.1%}（AI概率{ai_prob:.1%}）")
            elif prediction == "AI-Generated":
                analysis_parts.append(f"• 疑似AI生成内容，置信度{confidence:.1%}（AI概率{ai_prob:.1%}）")
            else:
                analysis_parts.append(f"• 检测为真实人类内容，置信度{confidence:.1%}")
        
        # 各模块结果
        modules = ['face', 'body', 'overall']
        for module in modules:
            if detection.get(f'{module}_ai_probability') is not None:
                ai_prob = detection[f'{module}_ai_probability']
                prediction = detection[f'{module}_prediction']
                if ai_prob > 0.7:
                    analysis_parts.append(f"▲ {module.title()}检测: {prediction}（AI概率{ai_prob:.1%}）")
                else:
                    analysis_parts.append(f"• {module.title()}检测: {prediction}（AI概率{ai_prob:.1%}）")
        
        return "\n".join(analysis_parts) if analysis_parts else "数字人检测数据不完整"

    def _analyze_fact_check_risks(self, fact_check_data):
        """分析事实核查风险"""
        if not fact_check_data.get('worth_checking'):
            return "• 内容不需要进行事实核查"
        
        results = fact_check_data.get('fact_check_results', [])
        summary = fact_check_data.get('search_summary', {})
        
        analysis_parts = []
        
        # 统计分析
        total_claims = summary.get('total_claims', 0)
        false_claims = summary.get('false_claims', 0)
        true_claims = summary.get('true_claims', 0)
        uncertain_claims = summary.get('uncertain_claims', 0)
        
        if false_claims > 0:
            analysis_parts.append(f"▲ 发现{false_claims}条虚假信息（共{total_claims}条断言）")
        
        if uncertain_claims > total_claims * 0.5:
            analysis_parts.append(f"▲ {uncertain_claims}条信息无法验证（占{uncertain_claims/total_claims:.1%}）")
        
        if true_claims == total_claims:
            analysis_parts.append(f"• 所有{total_claims}条断言均为真实")
        elif true_claims > 0:
            analysis_parts.append(f"• {true_claims}条断言为真实")
        
        # 详细结果（最多显示3条重要的）
        for i, result in enumerate(results[:3]):
            claim = result.get('claim', '')[:50] + '...' if len(result.get('claim', '')) > 50 else result.get('claim', '')
            is_true = result.get('is_true', '未确定')
            
            if is_true == "否":
                analysis_parts.append(f"▲ 虚假断言: {claim}")
            elif is_true == "未确定":
                analysis_parts.append(f"• 无法验证: {claim}")
        
        return "\n".join(analysis_parts) if analysis_parts else "事实核查数据不完整"

    def _build_rule_summary(self, dnf_rules):
        """构建规则摘要"""
        try:
            rule_items = []
            for rule in list(dnf_rules.get('disjuncts', {}).values())[:3]:
                if "∅" not in rule:
                    explained = self._explain_rule(rule)
                    rule_items.append(f"• {explained}")
            return "\n".join(rule_items) if rule_items else "无有效决策规则"
        except Exception as e:
            logger.warning(f"解析DNF规则失败: {str(e)}")
            return "规则解析失败"

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