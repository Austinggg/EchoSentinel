import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os
import json
import io
import base64
from models.dnf import DNF
from config import FEATURE_MAP

class RuleEditor:
    def __init__(self, model_path=None, num_features=None, num_conjuncts=None, n_out=2, feature_map=None):
        """规则编辑器，用于手动编辑和可视化DNF规则"""
        self.num_features = num_features
        self.num_conjuncts = num_conjuncts
        self.n_out = n_out
        self.feature_map = feature_map or FEATURE_MAP
        self.model_path = model_path
        
        # 保存编辑后的规则
        self.edited_rules = None
        self.original_rules = None
        
        # 加载已有模型的规则
        if model_path and os.path.exists(model_path):
            self._load_model_rules()
    
    def _load_model_rules(self, threshold=0.5):
        """从现有模型中加载规则"""
        try:
            # 创建一个临时DNF模型来加载权重
            from models.dnf import DNF
            from decision_service import DNFClassifier
            
            model = DNFClassifier(
                self.num_features,
                self.num_conjuncts,
                self.n_out
            )
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"), strict=False)
            
            # 获取规则
            self.original_rules = model.dnf.get_rules(threshold=threshold)
            if not self.edited_rules:
                self.edited_rules = self._deep_copy_rules(self.original_rules)
            
            return True
        except Exception as e:
            print(f"加载模型规则失败: {str(e)}")
            return False
    
    def _deep_copy_rules(self, rules):
        """深拷贝规则字典"""
        if not rules:
            return None
        
        return {
            "conjuncts": rules["conjuncts"].copy(),
            "disjuncts": {k: v for k, v in rules["disjuncts"].items()}
        }
    
    def get_rules(self):
        """获取当前规则"""
        return self.edited_rules or self.original_rules
        
    def edit_conjunct(self, conj_id, new_rule_expr):
        """编辑特定的合取规则
        
        参数:
            conj_id: 合取规则ID (如 'conj0')
            new_rule_expr: 新规则表达式 (如 'P1 ∧ ¬P3')
        
        返回:
            成功与否
        """
        if not self.edited_rules:
            if not self.original_rules:
                return False, "没有可编辑的规则"
            self.edited_rules = self._deep_copy_rules(self.original_rules)
        
        # 查找并替换规则
        for i, conj in enumerate(self.edited_rules["conjuncts"]):
            if conj.startswith(conj_id + " "):
                self.edited_rules["conjuncts"][i] = f"{conj_id} = {new_rule_expr}"
                return True, f"成功更新规则 {conj_id}"
        
        return False, f"找不到规则 {conj_id}"
    
    def edit_disjunct(self, class_id, new_rule_expr):
        """编辑特定类别的析取规则
        
        参数:
            class_id: 类别ID (如 '0' 或 '1')
            new_rule_expr: 新规则表达式 (如 'conj0 ∨ ¬conj2 ∨ conj5')
        
        返回:
            成功与否
        """
        if not self.edited_rules:
            if not self.original_rules:
                return False, "没有可编辑的规则"
            self.edited_rules = self._deep_copy_rules(self.original_rules)
        
        # 更新类别规则
        if str(class_id) in self.edited_rules["disjuncts"]:
            self.edited_rules["disjuncts"][str(class_id)] = new_rule_expr
            return True, f"成功更新类别 {class_id} 的规则"
        else:
            return False, f"找不到类别 {class_id} 的规则"
    
    def save_rules(self, output_path):
        """将编辑后的规则保存到JSON文件"""
        if not self.edited_rules:
            return False, "没有可保存的编辑规则"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.edited_rules, f, indent=2)
            return True, f"规则已保存到 {output_path}"
        except Exception as e:
            return False, f"保存规则失败: {str(e)}"
    
    def convert_rules_to_weights(self):
        """将编辑后的规则转换回DNF模型权重
        
        注意: 这是一个复杂的功能，需要将符号规则映射回模型权重
        """
        if not self.edited_rules:
            return False, "没有编辑过的规则可转换"
        
        try:
            # 创建一个新的DNF模型
            from models.dnf import DNF
            from decision_service import DNFClassifier
            
            model = DNFClassifier(
                self.num_features,
                self.num_conjuncts,
                self.n_out
            )
            
            # TODO: 将规则转换为权重矩阵
            # 这是一个复杂的转换过程，需要:
            # 1. 解析每个合取规则，确定正负权重
            # 2. 设置合取层的权重
            # 3. 解析析取规则，设置析取层的权重
            
            # 示例实现 (仅供参考，实际逻辑需要根据DNF模型的确切结构调整)
            # 初始化权重矩阵
            conj_weights = torch.zeros(self.num_conjuncts, self.num_features)
            disj_weights = torch.zeros(self.n_out, self.num_conjuncts)
            
            # 解析合取规则
            for i, conj in enumerate(self.edited_rules["conjuncts"]):
                if i >= self.num_conjuncts:
                    break
                    
                try:
                    _, expr = conj.split("=")
                    expr = expr.strip()
                    
                    if "∧" in expr:
                        predicates = expr.split("∧")
                    else:
                        predicates = [expr]
                    
                    for pred in predicates:
                        pred = pred.strip()
                        if "¬P" in pred:
                            # 否定谓词
                            p_idx = int(pred.split("P")[1]) - 1
                            conj_weights[i, p_idx] = -6.0  # 强否定权重
                        elif "P" in pred:
                            # 肯定谓词
                            p_idx = int(pred.split("P")[1]) - 1
                            conj_weights[i, p_idx] = 6.0   # 强肯定权重
                except Exception:
                    continue
            
            # 解析析取规则
            for cls, rule in self.edited_rules["disjuncts"].items():
                cls_idx = int(cls)
                if cls_idx >= self.n_out:
                    continue
                    
                terms = rule.split("∨")
                for term in terms:
                    term = term.strip()
                    if "¬conj" in term:
                        # 否定合取
                        conj_idx = int(term.split("conj")[1])
                        disj_weights[cls_idx, conj_idx] = -6.0  # 强否定权重
                    elif "conj" in term:
                        # 肯定合取
                        conj_idx = int(term.split("conj")[1])
                        disj_weights[cls_idx, conj_idx] = 6.0   # 强肯定权重
            
            # 应用权重到模型
            with torch.no_grad():
                model.dnf.conjunctions.weights.data = conj_weights
                model.dnf.disjunctions.weights.data = disj_weights
            
            # 保存转换后的模型
            new_model_path = self.model_path.replace(".pth", "_edited.pth")
            torch.save(model.state_dict(), new_model_path)
            
            return True, {"message": f"规则已转换为模型并保存到 {new_model_path}"}
        except Exception as e:
            return False, f"转换规则失败: {str(e)}"
    
    def visualize_rules(self, fmt='png'):
        """可视化当前规则集
        
        返回:
            Base64编码的图像数据
        """
        if not self.edited_rules and not self.original_rules:
            return None, "没有规则可视化"
        
        rules = self.edited_rules or self.original_rules
        
        # 配置matplotlib支持中文
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        # 设置中文字体
        mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'SimSun', 'WenQuanYi Micro Hei']
        mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        fig = Figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1)
        
        # 为规则创建矩阵表示
        feature_names = [f"{i}:{self.feature_map[i]}" for i in range(1, self.num_features+1)]
        conj_names = [f"规则{i}" for i in range(self.num_conjuncts)]
        
        # 创建合取规则矩阵
        conj_matrix = np.zeros((self.num_conjuncts, self.num_features))
        
        # 解析合取规则
        for i, conj in enumerate(rules["conjuncts"]):
            if i >= self.num_conjuncts:
                break
                
            try:
                _, expr = conj.split("=")
                expr = expr.strip()
                
                if "∧" in expr:
                    predicates = expr.split("∧")
                else:
                    predicates = [expr]
                
                for pred in predicates:
                    pred = pred.strip()
                    if "¬P" in pred:
                        # 否定谓词
                        p_idx = int(pred.split("P")[1]) - 1
                        conj_matrix[i, p_idx] = -1  # 否定
                    elif "P" in pred:
                        # 肯定谓词
                        p_idx = int(pred.split("P")[1]) - 1
                        conj_matrix[i, p_idx] = 1   # 肯定
            except Exception:
                continue
        
        # 绘制热力图
        im = ax.imshow(conj_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        
        # 设置轴标签
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(conj_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(conj_names)
        
        # 添加文本注释
        for i in range(len(conj_names)):
            for j in range(len(feature_names)):
                text = ""
                if conj_matrix[i, j] == 1:
                    text = "+"
                elif conj_matrix[i, j] == -1:
                    text = "-"
                    
                ax.text(j, i, text, ha="center", va="center", color="black")
        
        ax.set_title("DNF规则可视化")
        fig.tight_layout()
        
        # 转换为base64编码的图像
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return f"data:image/{fmt};base64,{img_data}", None