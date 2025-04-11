# assessment_api.py
from flask import Blueprint, request
from services.content_analysis.assessment_evaluator import AssessmentEvaluator
from utils.HttpResponse import success_response, error_response
import logging

assessment_api = Blueprint('assessment', __name__)

# 初始化评估器
evaluator = AssessmentEvaluator()
@assessment_api.route('/api/assess', methods=['POST'])
def assess_information():
    """
    执行多维信息评估
    ---
    tags:
      - 信息评估
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - text
          properties:
            text:
              type: string
              description: 需要评估的文本内容
            intent:
              type: array
              items:
                type: string
              description: 意图列表（来自信息提取结果）
            statements:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: integer
                  content:
                    type: string
              description: 关键陈述列表（来自信息提取结果）
            reputation:
              type: number
              format: float
              example: 0.75
              description: 发布者信誉评分（可选）
    responses:
      200:
        description: 评估结果
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            data:
              type: object
              properties:
                p1:
                  type: number
                  description: 背景信息充分性
                p2:
                  type: number
                  description: 背景信息准确性
                p3:
                  type: number
                  description: 内容完整性
                p4:
                  type: number
                  description: 不当意图风险
                p5:
                  type: number
                  description: 发布者信誉
                p6:
                  type: number
                  description: 情感煽动性
                p7:
                  type: number
                  description: 诱导行为风险
                p8:
                  type: number
                  description: 信息一致性
            message:
              type: string
      400:
        description: 参数错误
      500:
        description: 服务器内部错误
    """
    try:
        # 校验请求格式
        if not request.is_json:
            logging.warning(f"收到非JSON请求: Content-Type={request.headers.get('Content-Type', '未指定')}")
            return error_response(415, "请求必须是JSON格式，请设置Content-Type为application/json")
        
        data = request.json
        if not data or 'text' not in data:
            return error_response(400, "缺少必要参数：text")
        
        # 解析参数
        text = data['text']
        intent = data.get('intent', [])
        statements = data.get('statements', [])
        reputation = data.get('reputation', 0.5)
        
        # 执行评估流程
        results = {}
        try:
            results['p1'] = _safe_evaluate(evaluator.p1_assessment, text)
            results['p2'] = _safe_evaluate(evaluator.p2_assessment, text)
            results['p3'] = _safe_evaluate(evaluator.p3_assessment, text)
            results['p4'] = _safe_evaluate(evaluator.p4_assessment, text, intent)
            results['p5'] = _safe_evaluate(evaluator.p5_assessment, reputation)
            results['p6'] = _safe_evaluate(evaluator.p6_assessment, text)
            results['p7'] = _safe_evaluate(evaluator.p7_assessment, text)
            results['p8'] = _safe_evaluate(evaluator.p8_assessment, statements)
        except Exception as e:
            logging.exception("评估流程执行异常")
            return error_response(500, f"评估执行失败: {str(e)}")

        # 校验评估结果有效性
        for k, v in results.items():
            if not (0 <= v <= 1):
                logging.warning(f"异常评估值 {k}={v}")
                results[k] = max(0.0, min(v, 1.0))  # 强制归入[0,1]范围

        return success_response(results)
    except Exception as e:
        logging.exception("处理评估请求时发生未知错误")
        return error_response(500, f"服务器内部错误: {str(e)}")

def _safe_evaluate(method, *args):
    """安全执行评估方法"""
    try:
        result = method(*args)
        return float(result) if result is not None else 0.0
    except Exception as e:
        logging.error(f"评估方法{method.__name__}执行失败: {str(e)}")
        return 0.0