from flask import request, jsonify
import os
import tempfile
import uuid
import subprocess
import json
import cv2
import shutil
import logging
from pathlib import Path
from model_api_call import DigitalHumanEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIGCDetectionService:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.models_loaded = True  # 使用现有的外部模型，不需要加载transformers
        
    def extract_middle_frame(self, video_path, output_image_path):
        """提取视频中间的一帧 - 复用aigc_detection.py的实现"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(output_image_path, frame)

        cap.release()
        return ret
    
    def detect_face_aigc(self, temp_dir, video_id):
        """面部AIGC检测 - 使用DeepfakeBench，复用aigc_detection.py的逻辑"""
        try:
            # 准备DeepfakeBench需要的目录结构
            deepfake_base_path = Path(__file__).parent / "mod" / "DeepfakeBench" / "datasets" / "rgb" / "UADFV" / "fake"
            deepfake_path = deepfake_base_path / video_id
            deepfake_path.mkdir(parents=True, exist_ok=True)
            
            # 复制视频文件到DeepfakeBench目录
            video_files = list(Path(temp_dir).glob("*.mp4")) + list(Path(temp_dir).glob("*.avi")) + list(Path(temp_dir).glob("*.mov"))
            if not video_files:
                raise Exception("未找到视频文件")
            
            video_file = video_files[0]
            target_video = deepfake_path / f"{video_id}.mp4"
            shutil.copy2(video_file, target_video)
            
            logger.info(f"视频文件已复制到: {target_video}")
            
            # 执行DeepfakeBench检测 - 复用原有逻辑
            log_file = os.path.join(temp_dir, "face_detection.log")
            
            logger.info("开始执行DeepfakeBench检测...")
            with open(log_file, "w") as f:
                subprocess.run(
                    [
                        "/root/EchoSentinel/model-server/.venv/bin/python",
                        "/root/EchoSentinel/model-server/mod/DeepfakeBench/run_pipeline.py",
                    ],
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            
            logger.info("DeepfakeBench检测完成，开始查找结果文件...")
            
            # 读取结果 - 修复匹配逻辑
            result_path = Path(__file__).parent / "mod" / "DeepfakeBench" / "results" / "xception"
            result_data = None
            found_file = None
            
            logger.info(f"在路径中查找结果文件: {result_path}")
            
            if not result_path.exists():
                logger.error(f"结果路径不存在: {result_path}")
                raise Exception(f"结果路径不存在: {result_path}")
            
            # 列出所有JSON文件
            json_files = list(result_path.glob("*.json"))
            logger.info(f"找到的JSON文件: {[str(f.name) for f in json_files]}")
            
            for result_file in json_files:
                try:
                    logger.info(f"读取结果文件: {result_file}")
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    logger.info(f"结果文件内容: {data}")
                    
                    # 检查UADFV字段
                    if "UADFV" in data:
                        video_name = data["UADFV"].get("video_name", "")
                        logger.info(f"检查视频名称匹配: '{video_name}' vs '{video_id}'")
                        
                        # 更宽松的匹配逻辑
                        if (video_name == video_id or 
                            video_name.startswith(video_id) or 
                            video_id in video_name or
                            video_name.replace(".mp4", "") == video_id):
                            
                            logger.info(f"找到匹配的结果文件: {result_file}")
                            result_data = data
                            found_file = result_file
                            break
                    else:
                        logger.warning(f"结果文件中没有UADFV字段: {data.keys()}")
                        
                except Exception as e:
                    logger.error(f"读取结果文件失败 {result_file}: {e}")
                    continue
            
            # 如果没有找到匹配的结果，使用最新的结果文件
            if not result_data and json_files:
                logger.warning("没有找到完全匹配的结果，使用最新的结果文件")
                # 按修改时间排序，获取最新的文件
                latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"使用最新的结果文件: {latest_file}")
                
                try:
                    with open(latest_file, "r", encoding="utf-8") as f:
                        result_data = json.load(f)
                    found_file = latest_file
                    logger.info(f"成功读取最新结果文件: {result_data}")
                except Exception as e:
                    logger.error(f"读取最新结果文件失败: {e}")
            
            if result_data:
                # 提取预测结果
                if "UADFV" in result_data:
                    pred_mean = result_data["UADFV"].get("pred_mean", 0)
                    logger.info(f"提取pred_mean: {pred_mean}")
                else:
                    # 如果没有UADFV字段，查找其他可能的字段
                    pred_mean = 0
                    for key, value in result_data.items():
                        if isinstance(value, dict) and "pred_mean" in value:
                            pred_mean = value["pred_mean"]
                            break
                    logger.info(f"从其他字段提取pred_mean: {pred_mean}")
                
                prediction = "AI-Generated" if pred_mean > 0.5 else "Human"
                
                result = {
                    "region": "face",
                    "ai_probability": float(pred_mean),
                    "human_probability": 1.0 - float(pred_mean),
                    "prediction": prediction,
                    "confidence": float(max(pred_mean, 1.0 - pred_mean)),
                    "raw_results": result_data
                }
                
                logger.info(f"面部检测结果: {result}")
                
                # 清理结果文件
                if found_file:
                    try:
                        found_file.unlink()
                        logger.info(f"已清理结果文件: {found_file}")
                    except Exception as e:
                        logger.warning(f"清理结果文件失败: {e}")
                
                # 清理DeepfakeBench临时文件
                self.cleanup_deepfake_files(video_id)
                
                return result
            else:
                # 打印调试信息
                logger.error("未找到任何可用的结果文件")
                
                # 打印日志文件内容来诊断问题
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        log_content = f.read()
                        logger.error(f"DeepfakeBench日志内容: {log_content}")
                
                # 列出结果目录中的所有文件
                if result_path.exists():
                    all_files = list(result_path.glob("*"))
                    logger.error(f"结果目录中的所有文件: {[str(f.name) for f in all_files]}")
                
                raise Exception(f"未找到视频 {video_id} 的面部检测结果文件")
                
        except subprocess.CalledProcessError as e:
            # 处理 subprocess 错误
            error_msg = f"DeepfakeBench执行失败: {e}"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_content = f.read()
                    error_msg += f"\n日志内容: {log_content}"
            
            logger.error(error_msg)
            self.cleanup_deepfake_files(video_id)
            raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"面部AIGC检测失败: {str(e)}")
            self.cleanup_deepfake_files(video_id)
            raise Exception(f"面部AIGC检测失败: {str(e)}")
    
    def detect_body_aigc(self, temp_dir, video_id):
        """躯干AIGC检测 - 使用DigitalHumanEvaluator，复用aigc_detection.py的逻辑"""
        try:
            # 提取视频中间帧
            video_files = list(Path(temp_dir).glob("*.mp4")) + list(Path(temp_dir).glob("*.avi")) + list(Path(temp_dir).glob("*.mov"))
            if not video_files:
                raise Exception("未找到视频文件")
            
            video_file = video_files[0]
            image_path = os.path.join(temp_dir, f"{video_id}.jpg")
            self.extract_middle_frame(str(video_file), image_path)
            
            # 使用DigitalHumanEvaluator进行检测 - 复用start_body的逻辑
            evaluator = DigitalHumanEvaluator()
            output_dir = os.path.join(temp_dir, "body_results")
            os.makedirs(output_dir, exist_ok=True)
            
            evaluator.evaluate_image(image_path, output_dir=output_dir)
            
            # 读取结果 - 复用write_body_data的逻辑
            result_files = list(Path(output_dir).glob("*.json"))
            if result_files:
                with open(result_files[0], "r") as f:
                    data = json.load(f)
                    
                total_score = data.get("total_score", 0.0)
                prediction = "AI-Generated" if total_score > 0.5 else "Human"
                
                result = {
                    "region": "body",
                    "ai_probability": float(total_score),
                    "human_probability": 1.0 - float(total_score),
                    "prediction": prediction,
                    "confidence": float(max(total_score, 1.0 - total_score)),
                    "raw_results": data
                }
                
                return result
            else:
                raise Exception("未找到躯干检测结果")
                
        except Exception as e:
            raise Exception(f"躯干AIGC检测失败: {str(e)}")
    
    def detect_overall_aigc(self, temp_dir, video_id):
        """整体图像AIGC检测 - 使用DIRE，复用aigc_detection.py的逻辑"""
        try:
            # 提取视频中间帧
            video_files = list(Path(temp_dir).glob("*.mp4")) + list(Path(temp_dir).glob("*.avi")) + list(Path(temp_dir).glob("*.mov"))
            if not video_files:
                raise Exception("未找到视频文件")
            
            video_file = video_files[0]
            image_path = os.path.join(temp_dir, f"{video_id}.jpg")
            
            logger.info(f"开始提取视频帧: {video_file} -> {image_path}")
            frame_extracted = self.extract_middle_frame(str(video_file), image_path)
            
            if not frame_extracted:
                raise Exception("视频帧提取失败")
            
            if not os.path.exists(image_path):
                raise Exception(f"提取的图像文件不存在: {image_path}")
            
            logger.info(f"视频帧提取成功: {image_path}")
            
            # 检查DIRE相关文件是否存在
            dire_demo_path = "/root/EchoSentinel/model-server/DIRE/demo.py"
            dire_model_path = "/root/EchoSentinel/model-server/DIRE/lsun_adm.pth"
            dire_python_path = "/opt/conda/envs/dire/bin/python"
            
            if not os.path.exists(dire_demo_path):
                raise Exception(f"DIRE demo.py 不存在: {dire_demo_path}")
            
            if not os.path.exists(dire_model_path):
                raise Exception(f"DIRE 模型文件不存在: {dire_model_path}")
            
            if not os.path.exists(dire_python_path):
                logger.warning(f"DIRE Python环境不存在: {dire_python_path}，尝试使用默认Python")
                dire_python_path = "python"
            
            # 创建DIRE期望的目录结构 - 这是关键修复
            aigc_detection_dir = "/root/EchoSentinel/model-server/aigc_detection"
            video_specific_dir = os.path.join(aigc_detection_dir, video_id)
            os.makedirs(video_specific_dir, exist_ok=True)
            
            logger.info(f"创建DIRE输出目录: {video_specific_dir}")
            
            # 执行DIRE检测
            log_file = os.path.join(temp_dir, f"whole-{video_id}.log")
            
            logger.info(f"开始执行DIRE检测...")
            logger.info(f"命令: {dire_python_path} {dire_demo_path} -f {image_path} -m {dire_model_path}")
            
            try:
                with open(log_file, "w") as f:
                    result = subprocess.run(
                        [
                            dire_python_path,
                            dire_demo_path,
                            "-f", image_path,
                            "-m", dire_model_path,
                        ],
                        check=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd="/root/EchoSentinel/model-server/DIRE",
                    )
                
                logger.info("DIRE检测执行完成")
                
            except subprocess.CalledProcessError as e:
                # 即使退出码为1，如果是因为文件写入失败但检测成功，我们可以从日志中提取结果
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        log_content = f.read()
                    
                    # 检查是否是因为目录不存在导致的写入失败，但检测成功
                    if "Prob of being synthetic:" in log_content and "No such file or directory" in log_content:
                        logger.info("DIRE检测成功，但写入文件失败，从日志中提取结果")
                        
                        # 从日志中提取概率值
                        import re
                        prob_match = re.search(r"Prob of being synthetic:\s+([0-9.]+)", log_content)
                        if prob_match:
                            prob = float(prob_match.group(1))
                            
                            # 手动创建结果
                            result_data = {
                                "prob": prob,
                                "image_path": image_path,
                                "source": "extracted_from_log"
                            }
                            
                            prediction = "AI-Generated" if prob > 0.5 else "Human"
                            
                            result = {
                                "region": "overall",
                                "ai_probability": float(prob),
                                "human_probability": 1.0 - float(prob),
                                "prediction": prediction,
                                "confidence": float(max(prob, 1.0 - prob)),
                                "raw_results": result_data
                            }
                            
                            logger.info(f"从日志提取的整体检测结果: {result}")
                            
                            # 清理创建的目录
                            try:
                                if os.path.exists(video_specific_dir):
                                    shutil.rmtree(video_specific_dir)
                            except Exception as cleanup_e:
                                logger.warning(f"清理目录失败: {cleanup_e}")
                            
                            return result
                    
                    # 如果不是这种情况，抛出原始错误
                    error_msg = f"DIRE执行失败，退出码: {e.returncode}\n错误日志:\n{log_content}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                else:
                    raise Exception(f"DIRE执行失败，退出码: {e.returncode}")
            
            # 正常情况下读取结果文件
            result_file = os.path.join(video_specific_dir, "whole.json")
            
            if os.path.exists(result_file):
                logger.info(f"找到结果文件: {result_file}")
                try:
                    with open(result_file, "r") as f:
                        result_data = json.load(f)
                    
                    logger.info(f"成功读取结果文件: {result_data}")
                    
                    # 提取概率值
                    prob = result_data.get("prob", 0.0)
                    prediction = "AI-Generated" if prob > 0.5 else "Human"
                    
                    result = {
                        "region": "overall",
                        "ai_probability": float(prob),
                        "human_probability": 1.0 - float(prob),
                        "prediction": prediction,
                        "confidence": float(max(prob, 1.0 - prob)),
                        "raw_results": result_data
                    }
                    
                    logger.info(f"整体检测结果: {result}")
                    
                    # 清理结果文件和目录
                    try:
                        os.remove(result_file)
                        if os.path.exists(video_specific_dir):
                            shutil.rmtree(video_specific_dir)
                        logger.info(f"已清理结果文件和目录")
                    except Exception as e:
                        logger.warning(f"清理文件失败: {e}")
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"读取结果文件失败: {e}")
            
            # 如果没有找到结果文件，从日志中提取
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_content = f.read()
                
                # 尝试从日志中提取概率
                import re
                prob_match = re.search(r"Prob of being synthetic:\s+([0-9.]+)", log_content)
                if prob_match:
                    prob = float(prob_match.group(1))
                    
                    result_data = {
                        "prob": prob,
                        "image_path": image_path,
                        "source": "extracted_from_log"
                    }
                    
                    prediction = "AI-Generated" if prob > 0.5 else "Human"
                    
                    result = {
                        "region": "overall",
                        "ai_probability": float(prob),
                        "human_probability": 1.0 - float(prob),
                        "prediction": prediction,
                        "confidence": float(max(prob, 1.0 - prob)),
                        "raw_results": result_data
                    }
                    
                    logger.info(f"从日志提取的整体检测结果: {result}")
                    
                    # 清理创建的目录
                    try:
                        if os.path.exists(video_specific_dir):
                            shutil.rmtree(video_specific_dir)
                    except Exception as cleanup_e:
                        logger.warning(f"清理目录失败: {cleanup_e}")
                    
                    return result
            
            # 如果都失败了，抛出错误
            error_msg = "未找到整体检测结果"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_content = f.read()
                    error_msg += f"\nDIRE执行日志:\n{log_content}"
            
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            # 确保清理临时创建的目录
            try:
                aigc_detection_dir = "/root/EchoSentinel/model-server/aigc_detection"
                video_specific_dir = os.path.join(aigc_detection_dir, video_id)
                if os.path.exists(video_specific_dir):
                    shutil.rmtree(video_specific_dir)
            except Exception as cleanup_e:
                logger.warning(f"清理目录失败: {cleanup_e}")
            
            logger.error(f"整体AIGC检测失败: {str(e)}")
            raise Exception(f"整体AIGC检测失败: {str(e)}")
    def comprehensive_detection(self, temp_dir, video_id):
        """综合AIGC检测工作流"""
        try:
            # 执行三种检测
            face_result = self.detect_face_aigc(temp_dir, video_id)
            body_result = self.detect_body_aigc(temp_dir, video_id)
            overall_result = self.detect_overall_aigc(temp_dir, video_id)
            
            # 计算综合评分
            results = [face_result, body_result, overall_result]
            
            # 加权平均（整体检测权重更高）
            weights = {"face": 0.3, "body": 0.2, "overall": 0.5}
            
            weighted_ai_score = sum(result["ai_probability"] * weights[result["region"]] for result in results)
            weighted_human_score = sum(result["human_probability"] * weights[result["region"]] for result in results)
            
            # 一致性检查
            predictions = [result["prediction"] for result in results]
            ai_votes = predictions.count("AI-Generated")
            human_votes = predictions.count("Human")
            
            consensus = ai_votes >= 2
            confidence_penalty = 0.1 if ai_votes == 1 or human_votes == 1 else 0
            
            final_confidence = max(weighted_ai_score, weighted_human_score) - confidence_penalty
            
            return {
                "comprehensive_result": {
                    "prediction": "AI-Generated" if consensus else "Human",
                    "ai_probability": float(weighted_ai_score),
                    "human_probability": float(weighted_human_score),
                    "confidence": float(max(0, min(1, final_confidence))),
                    "consensus": consensus,
                    "votes": {"ai": ai_votes, "human": human_votes}
                },
                "detailed_results": {
                    "face": face_result,
                    "body": body_result,
                    "overall": overall_result
                }
            }
            
        except Exception as e:
            raise Exception(f"综合AIGC检测失败: {str(e)}")
    
    def cleanup_deepfake_files(self, video_id):
        """清理DeepfakeBench临时文件 - 复用clean_face_data的逻辑"""
        try:
            dataset_path = Path(__file__).parent / "mod" / "DeepfakeBench" / "datasets" / "rgb" / "UADFV" / "fake"
            frames_path = dataset_path / "frames"
            landmarks = dataset_path / "landmarks"
            
            paths_to_clean = [
                dataset_path / str(video_id),
                landmarks / str(video_id),
                frames_path / str(video_id)
            ]
            
            for path in paths_to_clean:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                        
        except Exception as e:
            logger.warning(f"清理DeepfakeBench文件失败: {e}")
    
    def save_temp_file(self, file):
        """保存临时文件"""
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        temp_dir = os.path.join(self.temp_dir, file_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, f"{file_id}{file_extension}")
        file.save(temp_path)
        return temp_dir, file_id
    
    def cleanup_temp_file(self, temp_dir):
        """清理临时文件"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    
    def __del__(self):
        """清理临时目录"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")

# 全局服务实例
aigc_service = AIGCDetectionService()

def init_digital_human(app):
    """初始化数字人AIGC检测服务"""
    try:
        # 检查必要的模型文件是否存在
        required_paths = [
            Path(__file__).parent / "mod" / "DeepfakeBench" / "run_pipeline.py",
            Path(__file__).parent / "DIRE" / "demo.py",
            Path(__file__).parent / "DIRE" / "lsun_adm.pth"
        ]
        
        missing_files = [str(path) for path in required_paths if not path.exists()]
        if missing_files:
            logger.warning(f"缺少必要文件: {missing_files}")
            app.aigc_ready = False
            return False
        
        app.aigc_ready = True
        app.aigc_service = aigc_service
        logger.info("数字人AIGC检测服务初始化成功")
        return True
    except Exception as e:
        logger.error(f"数字人AIGC检测服务初始化失败: {e}")
        app.aigc_ready = False
        return False

def register_digital_human_routes(app):
    """注册数字人AIGC检测相关的API路由"""
    
    @app.route('/aigc/detect/face', methods=['POST'])
    def detect_face_aigc():
        """面部AIGC检测API"""
        if not getattr(app, "aigc_ready", False):
            return jsonify({"error": "AIGC检测服务未就绪"}), 503
        
        if 'video' not in request.files:
            return jsonify({"error": "未上传视频文件"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
        
        temp_dir = None
        try:
            # 保存临时文件
            temp_dir, file_id = app.aigc_service.save_temp_file(video_file)
            
            # 执行面部检测
            result = app.aigc_service.detect_face_aigc(temp_dir, file_id)
            
            return jsonify({
                "success": True,
                "file_id": file_id,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"面部AIGC检测失败: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
        finally:
            # 清理临时文件
            if temp_dir:
                app.aigc_service.cleanup_temp_file(temp_dir)
    
    @app.route('/aigc/detect/body', methods=['POST'])
    def detect_body_aigc():
        """躯干AIGC检测API"""
        if not getattr(app, "aigc_ready", False):
            return jsonify({"error": "AIGC检测服务未就绪"}), 503
        
        if 'video' not in request.files:
            return jsonify({"error": "未上传视频文件"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
        
        temp_dir = None
        try:
            temp_dir, file_id = app.aigc_service.save_temp_file(video_file)
            result = app.aigc_service.detect_body_aigc(temp_dir, file_id)
            
            return jsonify({
                "success": True,
                "file_id": file_id,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"躯干AIGC检测失败: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
        finally:
            if temp_dir:
                app.aigc_service.cleanup_temp_file(temp_dir)
    
    @app.route('/aigc/detect/overall', methods=['POST'])
    def detect_overall_aigc():
        """整体AIGC检测API"""
        if not getattr(app, "aigc_ready", False):
            return jsonify({"error": "AIGC检测服务未就绪"}), 503
        
        if 'video' not in request.files:
            return jsonify({"error": "未上传视频文件"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
        
        temp_dir = None
        try:
            temp_dir, file_id = app.aigc_service.save_temp_file(video_file)
            result = app.aigc_service.detect_overall_aigc(temp_dir, file_id)
            
            return jsonify({
                "success": True,
                "file_id": file_id,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"整体AIGC检测失败: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
        finally:
            if temp_dir:
                app.aigc_service.cleanup_temp_file(temp_dir)
    
    @app.route('/aigc/detect/comprehensive', methods=['POST'])
    def detect_comprehensive_aigc():
        """综合AIGC检测工作流API"""
        if not getattr(app, "aigc_ready", False):
            return jsonify({"error": "AIGC检测服务未就绪"}), 503
        
        if 'video' not in request.files:
            return jsonify({"error": "未上传视频文件"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
        
        temp_dir = None
        try:
            temp_dir, file_id = app.aigc_service.save_temp_file(video_file)
            result = app.aigc_service.comprehensive_detection(temp_dir, file_id)
            
            return jsonify({
                "success": True,
                "file_id": file_id,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"综合AIGC检测失败: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
        finally:
            if temp_dir:
                app.aigc_service.cleanup_temp_file(temp_dir)
    
    @app.route('/aigc/status', methods=['GET'])
    def aigc_status():
        """获取AIGC检测服务状态"""
        try:
            models_ready = getattr(app, "aigc_ready", False)
            
            return jsonify({
                "service_ready": models_ready,
                "available_detections": ["face", "body", "overall", "comprehensive"],
                "models": {
                    "face": "DeepfakeBench",
                    "body": "DigitalHumanEvaluator", 
                    "overall": "DIRE"
                }
            })
        except Exception as e:
            return jsonify({
                "service_ready": False,
                "error": str(e)
            }), 500