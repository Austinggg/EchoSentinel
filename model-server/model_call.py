import torch
from modelscope import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info


class QwenVLModel:
    def __init__(self, model_path="/root/EchoSentinel/model-server/mod/Qwen2-VL-2B-Instruct"):
        """初始化Qwen2-VL模型
        
        Args:
            model_path (str): 模型路径
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.model_path = model_path
        self._init_model()
    
    def _init_model(self):
        """初始化模型和处理器"""
        try:
            # 使用modelscope加载模型
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            print(f"模型加载成功，使用设备: {self.device}")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
    
    def generate(self, text_prompt, image_path=None):
        """生成模型响应
        
        Args:
            text_prompt (str): 文本提示
            image_path (str, optional): 图片路径
            
        Returns:
            str: 模型生成的响应
        """
        if not self.model or not self.processor:
            return "模型未正确初始化"
            
        try:
            # 准备消息格式
            messages = []
            if image_path:
                # 图文混合输入
                image = Image.open(image_path)
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
            else:
                # 纯文本输入
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                        ],
                    }
                ]
            
            # 准备输入
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
            
            # 生成响应
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else None
            
        except Exception as e:
            print(f"生成响应失败: {str(e)}")
            return None

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = QwenVLModel()
    
    # 纯文本生成示例
    text_response = model.generate("你好，请介绍一下你自己")
    print("文本响应:", text_response)
    
    # 图文混合生成示例
    image_response = model.generate(
        "这张图片里有什么？",
        image_path="path/to/your/image.jpg"
    )
    print("图文响应:", image_response)