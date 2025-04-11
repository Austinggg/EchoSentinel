import json
import os
from tqdm import tqdm

class SocialMediaProcessor:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def process_files(self, extractor):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        for filename in tqdm(files, desc="处理微博文件"):
            filepath = os.path.join(self.input_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                message = data.get("message", "")
                if not message:
                    continue

                result = extractor.extract_information(message)
                if result:
                    data.update(result)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    def process_assessments(self, evaluator):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        for filename in tqdm(files, desc="评估微博文件"):
            filepath = os.path.join(self.input_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                assessments = {
                    "P1": lambda: evaluator.p1_assessment(data.get("message", "")),
                    "P2": lambda: evaluator.p2_assessment(data.get("message", "")),
                    "P3": lambda: evaluator.p3_assessment(data.get("message", "")),
                    "P4": lambda: evaluator.p4_assessment(
                        data.get("message", ""),
                        data.get("intent", [])
                    ),
                    "P5": lambda: evaluator.p5_assessment(data.get("reputation", 0.5)),
                    "P6": lambda: evaluator.p6_assessment(data.get("message", "")),
                    "P7": lambda: evaluator.p7_assessment(data.get("message", "")),
                    "P8": lambda: evaluator.p8_assessment(data.get("statements", []))
                }

                results = {}
                for key in assessments:
                    try:
                        results[key] = assessments[key]()
                    except Exception as e:
                        print(f"评估 {key} 失败: {str(e)}")
                        results[key] = None

                data["general"] = results
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")