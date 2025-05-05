# 在文件开头添加导入
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
import numpy as np
from pymongo import MongoClient
import threading
from flask import request, jsonify
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import logging

# 导入配置
from config import MONGODB, MODEL, FEATURE_MAP
# 导入数据集和模型
from models.weibo_21 import WeiboDataset
from models.classifiers import DNFClassifier
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入DNF模型相关组件
try:
    from models.dnf import DNF, DeltaDelayedExponentialDecayScheduler
except ImportError:
    logger.error("无法导入DNF模型组件，请确保models目录中包含dnf.py文件")
    raise

# 修改TrainingService类初始化方法，使用配置文件中的路径
class TrainingService:
    def __init__(self):
        self.model = None
        self.training_thread = None
        self.training_status = {
            "is_training": False,
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "best_accuracy": 0,
            "start_time": None,
            "end_time": None,
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "error": None
        }
        self.training_config = {}
        self.model_save_path = os.path.join(MODEL["save_dir"], "best_model.pth")
        self.logs_save_path = os.path.join(MODEL["save_dir"], "training_logs.json")
        
        # 创建模型保存目录
        os.makedirs(MODEL["save_dir"], exist_ok=True)
    def _train_process(self):
        """执行实际的训练过程"""
        try:
            # 重置训练状态并初始化
            self.training_status = {
                "is_training": True,
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": self.training_config["num_epochs"],
                "best_accuracy": 0,
                "start_time": time.time(),
                "end_time": None,
                "train_loss": [],
                "val_loss": [],
                "accuracy": [],
                "error": None
            }
            
            # 加载数据集
            logger.info("加载数据集...")
            dataset, error = self.load_dataset(
                mongo_uri=self.training_config["mongo_uri"],
                db_name=self.training_config["db_name"],
                collection_name=self.training_config["collection_name"],
                selected_features=self.training_config["selected_features"]
            )
            
            if error:
                self.training_status["error"] = error
                self.training_status["is_training"] = False
                logger.error(f"训练失败: {error}")
                return
                
            # 划分训练集和验证集
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # 创建模型
            num_features = len(self.training_config["selected_features"])
            self.model, error = self.create_model(
                num_features=num_features,
                num_conjuncts=self.training_config["num_conjuncts"],
                delta=self.training_config["delta"]
            )
            
            if error:
                self.training_status["error"] = error
                self.training_status["is_training"] = False
                logger.error(f"训练失败: {error}")
                return
                
            # 设置优化器
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config["learning_rate"])
            criterion = nn.CrossEntropyLoss()
            
            # 设置delta衰减调度器
            delta_scheduler = DeltaDelayedExponentialDecayScheduler(
            initial_delta=self.training_config["delta"],
            delta_decay_delay=self.training_config["delta_decay_delay"],
            delta_decay_steps=self.training_config["delta_decay_steps"],
            delta_decay_rate=self.training_config["delta_decay_rate"]
        )
            
            
            # 初始化早停参数
            best_val_loss = float('inf')
            patience_counter = 0
            best_accuracy = 0
            
            # 训练循环
            logger.info(f"开始训练，共{self.training_config['num_epochs']}个epochs...")
            for epoch in range(self.training_config["num_epochs"]):
                # 更新当前epoch
                self.training_status["current_epoch"] = epoch + 1
                
                # 训练阶段
                self.model.train()
                train_loss = 0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    
                avg_train_loss = train_loss / len(train_loader)
                self.training_status["train_loss"].append(avg_train_loss)
                
                # 验证阶段
                self.model.eval()
                val_loss = 0
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        # 修复：获取真实标签的类别索引，而不是直接使用one-hot编码
                        _, target_classes = torch.max(targets, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(target_classes.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                accuracy = accuracy_score(all_targets, all_preds)
                
                # 更新训练状态
                self.training_status["val_loss"].append(avg_val_loss)
                self.training_status["accuracy"].append(float(accuracy))
                self.training_status["progress"] = (epoch + 1) / self.training_config["num_epochs"] * 100
                
                # 检查是否为最佳模型
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.training_status["best_accuracy"] = float(best_accuracy)
                    torch.save(self.model.state_dict(), self.model_save_path)
                    patience_counter = 0
                    logger.info(f"Epoch {epoch+1}/{self.training_config['num_epochs']}, "
                              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                              f"Accuracy: {accuracy:.4f} - 保存最佳模型")
                else:
                    patience_counter += 1
                    logger.info(f"Epoch {epoch+1}/{self.training_config['num_epochs']}, "
                              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                              f"Accuracy: {accuracy:.4f} - 未改进")
                
                # 更新delta值
                current_step = epoch  # 使用当前epoch作为step计数
                delta_scheduler.step(self.model.dnf, current_step)                
                # 早停检查
                if patience_counter >= self.training_config["patience"]:
                    logger.info(f"早停触发，{self.training_config['patience']}个epoch未改进")
                    break
                    
            # 训练完成，更新状态
            self.training_status["is_training"] = False
            self.training_status["end_time"] = time.time()
            
            # 保存训练日志
            with open(self.logs_save_path, 'w') as f:
                json.dump({
                    "config": self.training_config,
                    "results": {
                        "train_loss": self.training_status["train_loss"],
                        "val_loss": self.training_status["val_loss"],
                        "accuracy": self.training_status["accuracy"],
                        "best_accuracy": self.training_status["best_accuracy"],
                        "training_time": self.training_status["end_time"] - self.training_status["start_time"]
                    }
                }, f, indent=4)
            
            logger.info(f"训练完成，最佳准确率: {best_accuracy:.4f}")
            
        except Exception as e:
            self.training_status["is_training"] = False
            self.training_status["error"] = str(e)
            logger.error(f"训练过程中出错: {str(e)}")
    def get_status(self):
        """获取当前训练状态"""
        return self.training_status
    def plot_training_history(self, save_path=None):
        """绘制训练历史并保存为图片"""
        try:
            if not self.training_status["train_loss"]:
                return False, "没有可用的训练历史"
            
            plt.figure(figsize=(12, 10))
            
            # 绘制损失曲线
            plt.subplot(2, 1, 1)
            plt.plot(self.training_status["train_loss"], label='训练损失')
            plt.plot(self.training_status["val_loss"], label='验证损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('训练与验证损失')
            plt.legend()
            
            # 绘制准确率曲线
            plt.subplot(2, 1, 2)
            plt.plot(self.training_status["accuracy"], label='验证准确率')
            plt.axhline(y=self.training_status["best_accuracy"], color='r', linestyle='-', label=f'最佳准确率: {self.training_status["best_accuracy"]:.4f}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('模型准确率')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                return True, save_path
            else:
                return True, "图表已生成但未保存"
        except Exception as e:
            return False, str(e)
    # 修改load_dataset方法，添加默认值
    def load_dataset(self, mongo_uri=None, db_name=None, collection_name=None, selected_features=None):
        """加载数据集"""
        # 使用配置中的默认值
        mongo_uri = mongo_uri or MONGODB["uri"]
        db_name = db_name or MONGODB["db_name"] 
        collection_name = collection_name or MONGODB["collection_name"]
        
        try:
            dataset = WeiboDataset(
                mongo_uri=mongo_uri,
                db_name=db_name,
                collection_name=collection_name,
                selected_features=selected_features
            )
            return dataset, None
        except Exception as e:
            error_msg = f"加载数据集时出错: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    # 修改create_model方法，使用配置文件默认值
    def create_model(self, num_features, num_conjuncts=None, n_out=2, delta=None):
        """创建模型实例"""
        # 使用配置中的默认值
        num_conjuncts = num_conjuncts or MODEL["training"]["num_conjuncts"]
        delta = delta or MODEL["training"]["delta"]
        
        try:
            model = DNFClassifier(num_features, num_conjuncts, n_out, delta)
            return model, None
        except Exception as e:
            error_msg = f"创建模型时出错: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    # 修改train方法，合并默认配置
    def train(self, config):
        """启动训练过程"""
        if self.training_status["is_training"]:
            return False, "已有训练任务正在进行中"
        
        # 合并默认配置和用户提供的配置
        default_config = {
            "mongo_uri": MONGODB["uri"],
            "db_name": MONGODB["db_name"],
            "collection_name": MONGODB["collection_name"],
            "selected_features": [f"P{i}" for i in range(1, 9)],
            "num_conjuncts": MODEL["training"]["num_conjuncts"],
            "num_epochs": MODEL["training"]["num_epochs"],
            "patience": MODEL["training"]["patience"],
            "delta": MODEL["training"]["delta"],
            "delta_decay_delay": MODEL["training"]["delta_decay_delay"],
            "delta_decay_steps": MODEL["training"]["delta_decay_steps"],
            "delta_decay_rate": MODEL["training"]["delta_decay_rate"],
            "learning_rate": MODEL["training"]["learning_rate"]
        }
        
        # 更新配置（用户提供的优先）
        for key, value in config.items():
            default_config[key] = value
        
        # 保存训练配置
        self.training_config = default_config
        
        # 异步启动训练
        self.training_thread = threading.Thread(target=self._train_process)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        return True, "训练任务已启动"
    
    # 修改explain_rule方法，使用全局特征映射
    def explain_rule(self, rule, feature_map=None):
        """解释规则（自然语言形式）"""
        if not feature_map:
            feature_map = FEATURE_MAP
        
        # 其余代码保持不变
        explanation = []
        for term in rule.split():
            if "P" in term:
                # 提取特征和符号
                sign = "非" if "¬" in term else ""
                p_num = term.split("P")[-1]
                try:
                    explanation.append(f"{sign}{feature_map[int(p_num)]}")
                except (ValueError, KeyError):
                    explanation.append(term)
            elif "conj" in term:
                explanation.append(f"({term})")
        return " 或 ".join(explanation).replace("∧", " 且 ").replace("∨", " 或 ")
    
    # 修改generate_semantic_report方法，使用全局特征映射
    def generate_semantic_report(self, rules, feature_map=None):
        """生成完整语义报告"""
        if not feature_map:
            feature_map = FEATURE_MAP
        
        # 其余代码保持不变
        report = []
        # 合取规则解释
        report.append("## 基础特征组合规则")
        for conj in rules["conjuncts"]:
            if "∅" not in conj:
                try:
                    _, expr = conj.split("=")
                    report.append(f"- 当 {self.explain_rule(expr.strip(), feature_map)} 时触发该规则")
                except ValueError:
                    continue
        
        # 最终决策规则解释
        report.append("\n## 最终决策逻辑")
        for cls, rule in rules["disjuncts"].items():
            if "∅" not in rule:
                cls_name = "虚假信息" if cls == 0 else "真实信息"
                report.append(f"### {cls_name}判定条件")
                report.append(f"满足以下任一条件即判定为{cls_name}：")
                for term in rule.split("∨"):
                    report.append(f"  - {self.explain_rule(term.strip(), feature_map)}")
        return "\n".join(report)

# 修改API路由部分，让其使用配置中的特征映射
def register_training_routes(app):
    """注册训练服务相关的API路由"""
    
    # 路径前缀改为与其他服务一致的格式
    @app.route('/train/start', methods=['POST'])
    def start_training():
        """启动新的训练任务"""
        if not hasattr(app, 'training_service'):
            return jsonify({"error": "训练服务未初始化"}), 503
        
        try:
            data = request.get_json() or {}
            
            # 启动训练（不再强制要求参数，使用默认配置）
            success, message = app.training_service.train(data)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": message
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": message
                }), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # 其他路由路径前缀统一修改
    @app.route('/train/status', methods=['GET'])
    def get_training_status():
        """获取当前训练任务的状态"""
        if not hasattr(app, 'training_service'):
            return jsonify({"error": "训练服务未初始化"}), 503
        
        try:
            status = app.training_service.get_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/train/logs', methods=['GET'])
    def get_training_logs():
        """获取完整的训练日志"""
        if not hasattr(app, 'training_service'):
            return jsonify({"error": "训练服务未初始化"}), 503
        
        try:
            logs_path = app.training_service.logs_save_path
            if not os.path.exists(logs_path):
                return jsonify({"error": "训练日志不存在"}), 404
            
            with open(logs_path, 'r') as f:
                logs = json.load(f)
            
            return jsonify(logs)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/train/rules', methods=['GET'])
    def get_model_rules():
        """获取模型规则"""
        if not hasattr(app, 'training_service'):
            return jsonify({"error": "训练服务未初始化"}), 503
        
        try:
            threshold = float(request.args.get('threshold', 0.5))
            format_type = request.args.get('format', 'raw')  # raw, text, semantic
            
            rules, error = app.training_service.get_model_rules(threshold)
            if error:
                return jsonify({"error": error}), 500
            
            if format_type == 'raw':
                # 原始格式
                return jsonify(rules)
            elif format_type == 'text':
                # 文本格式
                semantic_report = app.training_service.generate_semantic_report(rules, FEATURE_MAP)
                return jsonify({"explanation": semantic_report})
            elif format_type == 'semantic':
                # 结构化语义格式
                result = {
                    "conjuncts": [],
                    "decision_logic": {}
                }
                
                # 处理合取规则
                for conj in rules["conjuncts"]:
                    if "∅" not in conj:
                        try:
                            conj_id, expr = conj.split("=")
                            explanation = app.training_service.explain_rule(expr.strip())
                            result["conjuncts"].append({
                                "id": conj_id.strip(),
                                "expression": expr.strip(),
                                "explanation": explanation
                            })
                        except ValueError:
                            continue
                
                # 处理析取规则
                for cls, rule in rules["disjuncts"].items():
                    if "∅" not in rule:
                        cls_name = "虚假信息" if cls == 0 else "真实信息"
                        result["decision_logic"][cls_name] = []
                        for term in rule.split("∨"):
                            term = term.strip()
                            explanation = app.training_service.explain_rule(term)
                            result["decision_logic"][cls_name].append({
                                "condition": term,
                                "explanation": explanation
                            })
                
                return jsonify(result)
            else:
                return jsonify({"error": "不支持的格式类型"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/train/plot', methods=['GET'])
    def get_training_plot():
        """获取训练历史图表"""
        if not hasattr(app, 'training_service'):
            return jsonify({"error": "训练服务未初始化"}), 503
        
        try:
            plot_path = os.path.join(MODEL["save_dir"], "training_plot.png")
            success, result = app.training_service.plot_training_history(save_path=plot_path)
            if success:
                # 返回图片路径或图片本身
                return jsonify({"status": "success", "plot_path": result})
            else:
                return jsonify({"error": result}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
# 现在添加init_training_service函数
def init_training_service(app):
    """初始化训练服务并设置应用上下文"""
    try:
        app.training_service = TrainingService()
        logger.info("训练服务已初始化")
        return True
    except Exception as e:
        logger.error(f"训练服务初始化失败: {e}")
        return False