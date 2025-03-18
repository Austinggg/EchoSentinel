import json
from utils.information_extractor import InformationExtractor
from assessment import AssessmentEvaluator
from utils.file_processor import SocialMediaProcessor  # 修改导入

# 加载配置
with open('./config/config.json', encoding='utf-8') as f:
    config = json.load(f)

# 初始化提取器和评估器
extractor = InformationExtractor(config)
evaluator = AssessmentEvaluator(config)

# 指定要处理的文件夹为 "./weibo"
processor = SocialMediaProcessor("C:\Users\Administrator\Desktop\work\test")

# 批量处理文件夹内的 JSON 文件：先提取信息，再评估
processor.process_files(extractor)
processor.process_assessments(evaluator)

print("测试完成")

