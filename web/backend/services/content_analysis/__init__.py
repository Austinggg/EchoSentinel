from .assessment_evaluator import AssessmentEvaluator
# from .file_processor import WeiboProcessor
from .information_extractor import InformationExtractor
from .reporter import VeracityReporter

# 提供模块功能概述
__doc__ = """
内容分析服务模块
===========================

提供对社交媒体文本的全面分析功能：
1. 信息提取 - 从原始文本中提取关键信息点和意图
2. 可验证性评估 - 评估信息的可信度和可验证程度
3. 信息报告 - 生成专业分析报告
4. 微博数据处理 - 对微博数据进行处理和初步分析

主要类:
- AssessmentEvaluator: 信息可验证性评估器
- InformationExtractor: 信息要素提取器
- VeracityReporter: 可验证性报告生成器
- WeiboProcessor: 微博数据处理器
"""