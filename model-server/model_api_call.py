import json
import os
from datetime import datetime

import torch
from modelscope import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info


class DigitalHumanEvaluator:
    def __init__(
        self, model_path="/root/EchoSentinel/model-server/mod/Qwen2-VL-2B-Instruct"
    ):
        """初始化数字人评估器

        Args:
            model_path (str): 模型路径
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.model_path = model_path
        self._init_model()

        # 评估标准
        self.evaluation_criteria = [
            {
                "name": "关节位置",
                "prompt": "请仔细分析图片中人物的关节位置是否自然合理。特别关注颈部、肩部、肘部、手腕、髋部、膝盖和脚踝的弯曲角度和相对位置。如果所有关节位置都符合人体解剖学原理且看起来自然，请回答'是'；如果发现任何关节位置不自然、扭曲或变形，请回答'否'并简要说明原因。",
                "weight": 1.5,
            },
            {
                "name": "手指数量",
                "prompt": "请详细检查图片中人物的手部，确认每只手是否都有正确数量的手指（通常每只手应有5根手指）。请注意观察手指是否完整可见，是否存在多余或缺失的手指。如果手指数量正确且形态自然，请回答'是'；如果发现任何异常（如手指数量不正确、手指畸形或融合等），请回答'否'并简要描述问题。",
                "weight": 1.0,
            },
            {
                "name": "面部表情",
                "prompt": "请分析图片中人物的面部表情是否自然协调。观察眼睛、眉毛、嘴巴和面部肌肉的状态，判断表情是否符合正常人类情感表达。特别注意面部各部分是否协调一致（如微笑时眼角是否上扬）。如果面部表情看起来自然、协调且符合情境，请回答'是'；如果表情僵硬、不协调或存在扭曲变形，请回答'否'并简要说明问题所在。",
                "weight": 1.2,
            },
            {
                "name": "身体比例",
                "prompt": "请评估图片中人物的身体比例是否协调合理。关注头部与身体的比例、四肢与躯干的长度关系、上半身与下半身的平衡等。参考正常人体比例（如成人头长约为身高的1/8）进行判断。如果身体各部分比例协调且符合人体美学标准，请回答'是'；如果发现任何明显不协调（如手臂过长、腿部过短等），请回答'否'并简要指出不协调之处。",
                "weight": 1.3,
            },
            {
                "name": "动作流畅度",
                "prompt": "请评价图片中人物的动作姿态是否流畅自然。考虑人物当前动作的平衡性、重心分布、肌肉张力以及与物理规律的符合程度。判断该动作是否是人类能够自然完成的，以及是否符合人体运动力学原理。如果动作看起来流畅、自然且符合物理规律，请回答'是'；如果动作看起来僵硬、不自然或违反物理规律（如不可能的扭曲或平衡状态），请回答'否'并简要解释原因。",
                "weight": 1.4,
            },
        ]

    def _init_model(self):
        """初始化模型和处理器"""
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            print(f"模型加载成功，使用设备: {self.device}")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False

    def _generate_response(self, text_prompt, image):
        """生成模型响应"""
        if not self.model or not self.processor:
            return "模型未正确初始化"

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0] if output_text else None

        except Exception as e:
            print(f"生成响应失败: {str(e)}")
            return None

    def evaluate_image(self, image_path, output_dir="evaluation_results"):
        """评估图片

        Args:
            image_path (str): 图片路径
            output_dir (str): 输出目录

        Returns:
            dict: 评估结果
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 加载图片
        image = Image.open(image_path)

        # 准备评估结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "timestamp": timestamp,
            "image_path": image_path,
            "criteria": [],
            "total_score": 0,
        }

        # 对每个标准进行评估
        total_weight = 0
        for criterion in self.evaluation_criteria:
            response = self._generate_response(criterion["prompt"], image)

            # 解析响应
            is_positive = "是" in response if response else False
            score = 1 if is_positive else 0

            criterion_result = {
                "name": criterion["name"],
                "prompt": criterion["prompt"],
                "response": response,
                "score": score,
                "weight": criterion["weight"],
                "weighted_score": score * criterion["weight"],
            }

            result["criteria"].append(criterion_result)
            total_weight += criterion["weight"]

        # 计算总分
        total_score = sum(c["weighted_score"] for c in result["criteria"])
        result["total_score"] = total_score / total_weight if total_weight > 0 else 0

        # 保存结果
        output_file = os.path.join(output_dir, f"evaluation_{timestamp}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result


# 使用示例
if __name__ == "__main__":
    # 创建评估器实例
    evaluator = DigitalHumanEvaluator()

    # 评估图片
    result = evaluator.evaluate_image(
        "/root/EchoSentinel/model-server/aigc_detection/5e8444ee-cd8e-43ee-b6a7-2b8ce3efed9c.jpg",
        output_dir="whole_evaluation_results",
    )

    # 打印结果
    print(f"评估完成，总分: {result['total_score']:.2f}")
    print(f"详细结果已保存到: evaluation_results/evaluation_{result['timestamp']}.json")
