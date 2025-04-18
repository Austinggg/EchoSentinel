from flask import request, jsonify
import torch
import torch.nn as nn
import numpy as np
import os
from modules.dnf import DNF

class DNFClassifier(nn.Module):
    def __init__(self, num_preds, num_conjuncts, n_out, delta=0.01, weight_init_type="normal"):
        super(DNFClassifier, self).__init__()
        self.dnf = DNF(num_preds, num_conjuncts, n_out, delta, weight_init_type=weight_init_type)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.dnf(x))

class ContentClassifier:
    def __init__(self, model_path=None, num_features=8, num_conjuncts=10, feature_map=None):
        self.num_features = num_features
        self.num_conjuncts = num_conjuncts
        
        # 初始化模型
        self.model = DNFClassifier(num_features, num_conjuncts, 1)
        self.model_loaded = False
        
        # 设置默认特征映射
        if feature_map is None:
            self.feature_map = {
                1: "信息充分性", 2: "信息准确性", 
                3: "内容完整性", 4: "意图正当性",
                5: "发布者信誉", 6: "情感中立性",
                7: "无诱导行为", 8: "信息一致性"
            }
        else:
            self.feature_map = feature_map
        
        # 如果提供了模型路径，则加载模型
        if model_path:
            self.model_loaded = self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            # 使用strict=False忽略额外参数
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
            self.model.eval()
            print(f"微博分类模型成功加载自: {model_path}")
            return True
        except Exception as e:
            print(f"微博分类模型加载失败: {str(e)}")
            return False
    
    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            if isinstance(features, list):
                features = torch.FloatTensor([features])
            elif isinstance(features, np.ndarray):
                features = torch.FloatTensor(features).reshape(1, -1)
            else:
                features = torch.FloatTensor(features).reshape(1, -1)
            
            output = self.model(features)
            prob = output.item()
            pred_label = 1 if prob > 0.5 else 0
            
            return {
                "label": pred_label,
                "probability": prob,
                "class_name": "真实信息" if pred_label == 1 else "虚假信息"
            }
    
    def get_rules(self, threshold=0.5):
        return self.model.dnf.get_rules(threshold=threshold)
    
    def explain_rule(self, rule: str) -> str:
        """更精确地解释单条规则"""
        # 处理特殊情况
        if "conj" in rule and "¬" not in rule and "∧" not in rule and "∨" not in rule:
            # 这是一个简单的合取引用，如 "conj1"
            return rule
        
        # 处理析取规则（通常包含多个合取引用）
        if "∨" in rule:
            parts = rule.split("∨")
            explanations = [self.explain_rule(part.strip()) for part in parts]
            return " 或 ".join(explanations)
        
        # 处理合取规则
        if "∧" in rule:
            parts = rule.split("∧")
            explanations = [self.explain_rule(part.strip()) for part in parts]
            return " 且 ".join(explanations)
        
        # 处理简单谓词（如 P2 或 ¬P2）
        if "P" in rule:
            sign = "非" if "¬" in rule else ""
            p_num = rule.split("P")[-1]
            try:
                feature_name = self.feature_map[int(p_num)]
                return f"{sign}{feature_name}"
            except (ValueError, KeyError):
                return rule
        
        # 处理不能识别的规则
        return rule
    
    def explain_rules(self, threshold=0.5):
        """生成规则解释报告"""
        rules = self.get_rules(threshold)
        
        report = []
        # 合取规则解释
        report.append("## 基础特征组合规则")
        for i, conj in enumerate(rules["conjuncts"]):
            if "∅" not in conj:
                try:
                    conj_id, expr = conj.split("=")
                    conj_id = conj_id.strip()
                    expr = expr.strip()
                    explanation = self.explain_rule(expr)
                    report.append(f"- {conj_id}: 当 {explanation} 成立时激活")
                except ValueError:
                    # 处理无法分割的情况
                    report.append(f"- 规则 {i}: {conj}")
        
        # 最终决策规则解释
        report.append("\n## 最终决策逻辑")
        for cls, rule in rules["disjuncts"].items():
            if "∅" not in rule:
                cls_name = "虚假信息" if cls == 0 else "真实信息"
                report.append(f"### {cls_name}判定条件")
                report.append(f"满足以下任一条件即判定为{cls_name}：")
                
                # 将析取规则分解为单独的条件
                terms = rule.split("∨")
                for term in terms:
                    term = term.strip()
                    if "conj" in term:
                        # 这是对合取的引用，可能有否定符
                        has_negation = "¬" in term
                        conj_id = term.replace("¬", "").strip()
                        negation_text = "不" if has_negation else ""
                        report.append(f"  - 当{negation_text}满足 {conj_id} 时")
                    else:
                        # 直接的谓词表达式
                        explanation = self.explain_rule(term)
                        report.append(f"  - 当 {explanation} 时")
        
        return "\n".join(report)
    def get_formatted_rules(self, threshold=0.5):
        """获取格式化的规则解释，去除重复并提高可读性"""
        rules = self.get_rules(threshold)
        
        # 第一步：整理并去重合取规则
        conj_mapping = {}  # 存储规则内容到ID的映射
        conj_expressions = {}  # 存储ID到规则内容的映射
        conj_explanations = {}  # 存储ID到自然语言解释的映射
        
        for conj in rules["conjuncts"]:
            if "∅" not in conj:
                try:
                    conj_id, expr = conj.split("=")
                    conj_id = conj_id.strip()
                    expr = expr.strip()
                    
                    # 如果这个表达式已经存在，记录同义规则
                    if expr in conj_mapping:
                        conj_mapping[expr].append(conj_id)
                    else:
                        conj_mapping[expr] = [conj_id]
                        conj_expressions[conj_id] = expr
                        conj_explanations[conj_id] = self.explain_rule(expr)
                except ValueError:
                    # 处理无法分割的情况
                    pass
        
        # 第二步：整理析取规则（最终决策逻辑）
        decision_rules = {}
        for cls, rule in rules["disjuncts"].items():
            if "∅" not in rule:
                cls_name = "虚假信息" if cls == 0 else "真实信息"
                decision_rules[cls_name] = []
                
                # 将析取规则分解为单独的条件
                terms = rule.split("∨")
                for term in terms:
                    term = term.strip()
                    if "conj" in term:
                        # 这是对合取的引用
                        has_negation = "¬" in term
                        conj_id = term.replace("¬", "").strip()
                        
                        # 如果是有效的合取ID
                        if conj_id in conj_expressions:
                            expr = conj_expressions[conj_id]
                            explanation = conj_explanations[conj_id]
                            negation_text = "不" if has_negation else ""
                            
                            # 记录所有同义ID
                            synonyms = conj_mapping.get(expr, [conj_id])
                            synonyms_text = ", ".join(synonyms)
                            
                            decision_rules[cls_name].append({
                                "condition": f"{negation_text}满足规则 {conj_id}",
                                "detail": f"{explanation}",
                                "synonyms": synonyms_text if len(synonyms) > 1 else None
                            })
        
        # 构建结构化结果
        result = {
            "unique_rules": [],
            "decision_logic": decision_rules
        }
        
        # 添加去重后的合取规则
        for expr, ids in conj_mapping.items():
            primary_id = ids[0]
            explanation = conj_explanations[primary_id]
            result["unique_rules"].append({
                "id": primary_id,
                "synonyms": ids[1:] if len(ids) > 1 else [],
                "expression": expr,
                "explanation": explanation
            })
        
        return result

    def example_predict(self, features_dict):
        reverse_map = {v: k for k, v in self.feature_map.items()}
        
        feature_vector = [0.0] * self.num_features
        for feature_name, value in features_dict.items():
            if feature_name in reverse_map:
                idx = reverse_map[feature_name] - 1
                feature_vector[idx] = value
        
        return self.predict(feature_vector)

def init_decision(app):
    """初始化决策模型并设置应用上下文"""
    model_path = "mod/Desision-mod/best_model.pth"
    
    try:
        app.content_classifier = ContentClassifier(model_path=model_path)
        app.decision_ready = app.content_classifier.model_loaded
        return app.decision_ready
    except Exception as e:
        print(f"决策模型初始化失败: {e}")
        app.decision_ready = False
        return False



def register_decision_routes(app):
    """注册微博分类服务相关的API路由"""
    
    @app.route('/classify', methods=['POST'])
    def classify():
        if not getattr(app, "decision_ready", False):
            return jsonify({"error": "决策模型服务未就绪"}), 503
        
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "无效的请求数据"}), 400
            
            # 支持特征列表或特征词典
            if "features" in data:
                features = data["features"]
                result = app.content_classifier.predict(features)
            elif "features_dict" in data:
                features_dict = data["features_dict"]
                result = app.content_classifier.example_predict(features_dict)
            else:
                return jsonify({"error": "缺少features或features_dict参数"}), 400
                
            # 可选返回解释
            if data.get("explain", False):
                result["explanation"] = app.content_classifier.explain_rules()
                
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/explain', methods=['GET'])
    def explain():
        if not getattr(app, "decision_ready", False):
            return jsonify({"error": "决策模型服务未就绪"}), 503
        
        try:
            threshold = float(request.args.get('threshold', 0.5))
            format_type = request.args.get('format', 'enhanced')  # 默认使用增强格式
            
            if format_type == 'raw':
                # 原始格式，直接返回DNF规则
                rules = app.content_classifier.model.dnf.get_rules(threshold=threshold)
                return jsonify(rules)
            
            elif format_type == 'text':
                # 文本格式，返回传统的解释文本
                explanation = app.content_classifier.explain_rules(threshold)
                return jsonify({"explanation": explanation})
            
            else:  # 'enhanced' 格式
                # 结构化的增强格式，去重并提高可读性
                formatted_rules = app.content_classifier.get_formatted_rules(threshold)
                return jsonify(formatted_rules)
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500