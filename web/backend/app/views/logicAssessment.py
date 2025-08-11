# assessment_api.py
from flask import Blueprint, request
from app.services.content_analysis.assessment_evaluator import AssessmentEvaluator
from app.models.analysis import ContentAnalysis,VideoTranscript
from app.utils.extensions import db
from app.utils.HttpResponse import success_response, error_response
import logging

assessment_api = Blueprint('assessment', __name__)

# 初始化评估器
evaluator = AssessmentEvaluator()

@assessment_api.route('/api/assess-all', methods=['POST'])
def assess_all_information():
    try:
        # 校验请求格式
        if not request.is_json:
            logging.warning(f"收到非JSON请求: Content-Type={request.headers.get('Content-Type', '未指定')}")
            return error_response(415, "请求必须是JSON格式，请设置Content-Type为application/json")
        
        data = request.json
        if not data:
            return error_response(400, "请求体不能为空")
        
        # 解析参数
        text = data.get('text')
        intent = data.get('intent', [])
        statements = data.get('statements', [])
        reputation = data.get('reputation', 0.5)
        
        # 新增参数
        include_reasoning = data.get('include_reasoning', False)
        assessment_item = data.get('assessment_item', None)  # 单项评估编号，如"p1"
        assess_all = data.get('assess_all', assessment_item is None)  # 默认全评估
        
        # 验证必要参数
        if text is None and not (assessment_item == "p5" and reputation is not None):
            return error_response(400, "缺少必要参数：text（除非只评估p5）")
            
        # 执行评估流程
        results = {}
        
        try:
            # 单项评估
            if assessment_item:
                if assessment_item == "p1":
                    results[assessment_item] = _safe_evaluate(evaluator.p1_assessment, text, include_reasoning=include_reasoning)
                elif assessment_item == "p2":
                    results[assessment_item] = _safe_evaluate(evaluator.p2_assessment, text, include_reasoning=include_reasoning)
                elif assessment_item == "p3":
                    results[assessment_item] = _safe_evaluate(evaluator.p3_assessment, text, include_reasoning=include_reasoning)
                elif assessment_item == "p4":
                    results[assessment_item] = _safe_evaluate(evaluator.p4_assessment, text, intent, include_reasoning=include_reasoning)
                elif assessment_item == "p5":
                    results[assessment_item] = _safe_evaluate(evaluator.p5_assessment, reputation, include_reasoning=include_reasoning)
                elif assessment_item == "p6":
                    results[assessment_item] = _safe_evaluate(evaluator.p6_assessment, text, include_reasoning=include_reasoning)
                elif assessment_item == "p7":
                    results[assessment_item] = _safe_evaluate(evaluator.p7_assessment, text, include_reasoning=include_reasoning)
                elif assessment_item == "p8":
                    results[assessment_item] = _safe_evaluate(evaluator.p8_assessment, statements, include_reasoning=include_reasoning)
                else:
                    return error_response(400, f"未知的评估项：{assessment_item}")
            
            # 全部评估
            elif assess_all:
                # 手动调用每个评估方法，只是简单地组合结果
                results["p1"] = _safe_evaluate(evaluator.p1_assessment, text, include_reasoning=include_reasoning)
                results["p2"] = _safe_evaluate(evaluator.p2_assessment, text, include_reasoning=include_reasoning)
                results["p3"] = _safe_evaluate(evaluator.p3_assessment, text, include_reasoning=include_reasoning)
                results["p4"] = _safe_evaluate(evaluator.p4_assessment, text, intent, include_reasoning=include_reasoning)
                results["p5"] = _safe_evaluate(evaluator.p5_assessment, reputation, include_reasoning=include_reasoning)
                results["p6"] = _safe_evaluate(evaluator.p6_assessment, text, include_reasoning=include_reasoning)
                results["p7"] = _safe_evaluate(evaluator.p7_assessment, text, include_reasoning=include_reasoning)
                
                # p8需要陈述列表参数
                if statements and isinstance(statements, list) and len(statements) > 0:
                    results["p8"] = _safe_evaluate(evaluator.p8_assessment, statements, include_reasoning=include_reasoning)
                else:
                    if include_reasoning:
                        results["p8"] = {"score": None, "reasoning_content": "未提供有效的陈述列表"}
                    else:
                        results["p8"] = None
            
            # 校验评估结果有效性
            if not include_reasoning:
                for k, v in results.items():
                    if k != "average" and not (0 <= v <= 1):
                        logging.warning(f"异常评估值 {k}={v}")
                        results[k] = max(0.0, min(float(v or 0), 1.0))  # 强制归入[0,1]范围
            else:
                # 处理带有reasoning的结果
                for k, v in results.items():
                    if k != "average" and isinstance(v, dict) and "score" in v:
                        if not (0 <= (v["score"] or 0) <= 1):
                            logging.warning(f"异常评估值 {k}={v['score']}")
                            results[k]["score"] = max(0.0, min(float(v["score"] or 0), 1.0))
            
            return success_response(results)
        except Exception as e:
            logging.exception("评估流程执行异常")
            return error_response(500, f"评估执行失败: {str(e)}")
    except Exception as e:
        logging.exception("处理评估请求时发生未知错误")
        return error_response(500, f"服务器内部错误: {str(e)}")

def _safe_evaluate(method, *args, include_reasoning=True, **kwargs):
    """安全执行评估方法，支持返回评估理由
    
    Args:
        method: 评估方法
        include_reasoning: 是否包含理由
        *args: 传递给评估方法的位置参数
        **kwargs: 传递给评估方法的关键字参数
    
    Returns:
        如果include_reasoning为True，返回包含score和reasoning_content的字典
        否则只返回评分
    """
    try:
        if include_reasoning:
            kwargs["include_reasoning"] = True
            result = method(*args, **kwargs)
            return result  # 已经是字典格式包含score和reasoning_content
        else:
            result = method(*args, **kwargs)
            return float(result) if result is not None else 0.0
    except Exception as e:
        logging.error(f"评估方法{method.__name__}执行失败: {str(e)}")
        if include_reasoning:
            return {"score": 0.0, "reasoning_content": f"评估失败: {str(e)}"}
        else:
            return 0.0

# 添加指定项评估的路由
@assessment_api.route('/api/assess/item/<item>', methods=['POST'])
def assess_single_item_path(item):
    """通过路径参数进行单项评估"""
    try:
        if not request.is_json:
            return error_response(415, "请求必须是JSON格式")
        
        data = request.json or {}
        data['assessment_item'] = item
        # 单项评估默认包含理由(关键修改)
        if 'include_reasoning' not in data:
            data['include_reasoning'] = True
        
        return assess_all_information()  # 调用全部评估函数
    except Exception as e:
        logging.exception(f"处理单项评估请求时发生错误: {item}")
        return error_response(500, f"服务器内部错误: {str(e)}")
# 添加视频评估路由 - 全量评估
@assessment_api.route('/api/videos/<video_id>/assess', methods=['POST'])
def assess_video(video_id):
    """根据视频ID进行全量评估并存储结果"""
    try:
        # 查询数据库获取视频内容分析
        content_analysis = ContentAnalysis.query.filter_by(video_id=video_id).first()
        if not content_analysis:
            return error_response(404, f"未找到视频ID为 {video_id} 的内容分析")
        
        include_reasoning = True  # 默认值

        # 查询视频转录文本 
        transcript = VideoTranscript.query.filter_by(video_id=video_id).first()
        if not transcript or not transcript.transcript:
            # 如果没有转录文本，回退到使用摘要
            text = content_analysis.summary 
            if not text:
                return error_response(400, "视频转录文本和内容摘要均为空，无法进行评估")
        else:
            # 使用完整转录文本
            text = transcript.transcript
        
        # 获取评估所需参数
        intent = content_analysis.intent or []
        statements = content_analysis.statements or []
        reputation = 0.5  # 默认值，实际应从用户信誉系统获取
        
        # 执行全部评估
        try:
            # 依次评估每个项目
            results = {}
            
            # P1: 背景信息充分性评估
            p1_result = _safe_evaluate(evaluator.p1_assessment, text, include_reasoning=include_reasoning)
            results["p1"] = p1_result
            content_analysis.p1_score = p1_result["score"] if include_reasoning else p1_result
            if include_reasoning and "reasoning_content" in p1_result:
                content_analysis.p1_reasoning = p1_result["reasoning_content"]
            
            # P2: 背景信息准确性评估
            p2_result = _safe_evaluate(evaluator.p2_assessment, text, include_reasoning=include_reasoning)
            results["p2"] = p2_result
            content_analysis.p2_score = p2_result["score"] if include_reasoning else p2_result
            if include_reasoning and "reasoning_content" in p2_result:
                content_analysis.p2_reasoning = p2_result["reasoning_content"]
            
            # P3: 内容完整性评估
            p3_result = _safe_evaluate(evaluator.p3_assessment, text, include_reasoning=include_reasoning)
            results["p3"] = p3_result
            content_analysis.p3_score = p3_result["score"] if include_reasoning else p3_result
            if include_reasoning and "reasoning_content" in p3_result:
                content_analysis.p3_reasoning = p3_result["reasoning_content"]
            
            # P4: 不当意图评估
            p4_result = _safe_evaluate(evaluator.p4_assessment, text, intent, include_reasoning=include_reasoning)
            results["p4"] = p4_result
            content_analysis.p4_score = p4_result["score"] if include_reasoning else p4_result
            if include_reasoning and "reasoning_content" in p4_result:
                content_analysis.p4_reasoning = p4_result["reasoning_content"]
            
            # P5: 发布者历史评估
            p5_result = _safe_evaluate(evaluator.p5_assessment, reputation, include_reasoning=include_reasoning)
            results["p5"] = p5_result
            content_analysis.p5_score = p5_result["score"] if include_reasoning else p5_result
            if include_reasoning and "reasoning_content" in p5_result:
                content_analysis.p5_reasoning = p5_result["reasoning_content"]
            
            # P6: 情感煽动性评估
            p6_result = _safe_evaluate(evaluator.p6_assessment, text, include_reasoning=include_reasoning)
            results["p6"] = p6_result
            content_analysis.p6_score = p6_result["score"] if include_reasoning else p6_result
            if include_reasoning and "reasoning_content" in p6_result:
                content_analysis.p6_reasoning = p6_result["reasoning_content"]
            
            # P7: 诱导行为评估
            p7_result = _safe_evaluate(evaluator.p7_assessment, text, include_reasoning=include_reasoning)
            results["p7"] = p7_result
            content_analysis.p7_score = p7_result["score"] if include_reasoning else p7_result
            if include_reasoning and "reasoning_content" in p7_result:
                content_analysis.p7_reasoning = p7_result["reasoning_content"]
            
            # P8: 信息一致性评估(只有当有多个陈述时才评估)
            if statements and isinstance(statements, list) and len(statements) > 0:
                p8_result = _safe_evaluate(evaluator.p8_assessment, statements, include_reasoning=include_reasoning)
                results["p8"] = p8_result
                content_analysis.p8_score = p8_result["score"] if include_reasoning else p8_result
                if include_reasoning and "reasoning_content" in p8_result:
                    content_analysis.p8_reasoning = p8_result["reasoning_content"]
            else:
                if include_reasoning:
                    results["p8"] = {"score": None, "reasoning_content": "未提供有效的陈述列表"}
                else:
                    results["p8"] = None
            
            # 提交数据库更改
            db.session.commit()
            
            # 返回结果
            return success_response({
                "results": results,
                "message": "评估已完成并保存到数据库"
            })
            
        except Exception as e:
            db.session.rollback()
            logging.exception(f"视频评估执行异常: {str(e)}")
            return error_response(500, f"评估执行失败: {str(e)}")
            
    except Exception as e:
        logging.exception(f"处理视频评估请求时发生错误: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")

# 添加视频评估路由 - 单项评估
@assessment_api.route('/api/videos/<video_id>/assess/<item>', methods=['POST'])
def assess_video_item(video_id, item):
    """根据视频ID进行单项评估并存储结果"""
    try:
        # 检查评估项是否有效
        if item not in ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]:
            return error_response(400, f"无效的评估项: {item}")
        include_reasoning = True  # 默认值

        # 查询数据库获取视频内容分析
        content_analysis = ContentAnalysis.query.filter_by(video_id=video_id).first()
        if not content_analysis:
            return error_response(404, f"未找到视频ID为 {video_id} 的内容分析")
                
        transcript = VideoTranscript.query.filter_by(video_id=video_id).first()
        if not transcript or not transcript.transcript:
            # 如果没有转录文本，回退到使用摘要
            text = content_analysis.summary
            if not text and item != "p5" and item != "p8":
                return error_response(400, "视频转录文本和内容摘要均为空，无法进行评估")
        else:
            # 使用完整转录文本
            text = transcript.transcript
        
        # 获取评估所需参数
        intent = content_analysis.intent or []
        statements = content_analysis.statements or []
        reputation = 0.5  # 默认值，实际应从用户信誉系统获取
        
        # 执行单项评估
        try:
            result = None
            
            # 根据评估项执行对应评估
            if item == "p1":
                result = _safe_evaluate(evaluator.p1_assessment, text, include_reasoning=include_reasoning)
                content_analysis.p1_score = result["score"] if include_reasoning else result
                if include_reasoning and "reasoning_content" in result:
                    content_analysis.p1_reasoning = result["reasoning_content"]
            
            elif item == "p2":
                result = _safe_evaluate(evaluator.p2_assessment, text, include_reasoning=include_reasoning)
                content_analysis.p2_score = result["score"] if include_reasoning else result
                if include_reasoning and "reasoning_content" in result:
                    content_analysis.p2_reasoning = result["reasoning_content"]
            
            elif item == "p3":
                result = _safe_evaluate(evaluator.p3_assessment, text, include_reasoning=include_reasoning)
                content_analysis.p3_score = result["score"] if include_reasoning else result
                if include_reasoning and "reasoning_content" in result:
                    content_analysis.p3_reasoning = result["reasoning_content"]
            
            elif item == "p4":
                result = _safe_evaluate(evaluator.p4_assessment, text, intent, include_reasoning=include_reasoning)
                content_analysis.p4_score = result["score"] if include_reasoning else result
                if include_reasoning and "reasoning_content" in result:
                    content_analysis.p4_reasoning = result["reasoning_content"]
            
            elif item == "p5":
                result = _safe_evaluate(evaluator.p5_assessment, reputation, include_reasoning=include_reasoning)
                content_analysis.p5_score = result["score"] if include_reasoning else result
                if include_reasoning and "reasoning_content" in result:
                    content_analysis.p5_reasoning = result["reasoning_content"]
            
            elif item == "p6":
                result = _safe_evaluate(evaluator.p6_assessment, text, include_reasoning=include_reasoning)
                content_analysis.p6_score = result["score"] if include_reasoning else result
                if include_reasoning and "reasoning_content" in result:
                    content_analysis.p6_reasoning = result["reasoning_content"]
            
            elif item == "p7":
                result = _safe_evaluate(evaluator.p7_assessment, text, include_reasoning=include_reasoning)
                content_analysis.p7_score = result["score"] if include_reasoning else result
                if include_reasoning and "reasoning_content" in result:
                    content_analysis.p7_reasoning = result["reasoning_content"]
            
            elif item == "p8":
                if statements and isinstance(statements, list) and len(statements) > 0:
                    result = _safe_evaluate(evaluator.p8_assessment, statements, include_reasoning=include_reasoning)
                    content_analysis.p8_score = result["score"] if include_reasoning else result
                    if include_reasoning and "reasoning_content" in result:
                        content_analysis.p8_reasoning = result["reasoning_content"]
                else:
                    if include_reasoning:
                        result = {"score": None, "reasoning_content": "未提供有效的陈述列表"}
                    else:
                        result = None
            
            # 提交数据库更改
            db.session.commit()
            
            # 返回结果
            return success_response({
                "result": result,
                "message": f"评估项 {item} 已完成并保存到数据库"
            })
            
        except Exception as e:
            db.session.rollback()
            logging.exception(f"视频评估执行异常: {str(e)}")
            return error_response(500, f"评估执行失败: {str(e)}")
            
    except Exception as e:
        logging.exception(f"处理视频评估请求时发生错误: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")
    
# 添加查询单个评估项的路由
@assessment_api.route('/api/videos/<video_id>/assessment/<item>', methods=['GET'])
def get_video_assessment_item(video_id, item):
    """获取视频特定评估项的详细信息
    
    Args:
        video_id: 视频ID
        item: 评估项代码，如p1, p2等
        
    Returns:
        评估项的详细信息，包括分数和评估理由
    """
    try:
        # 检查评估项是否有效
        if item not in ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]:
            return error_response(400, f"无效的评估项: {item}")
            
        # 查询数据库获取视频内容分析
        content_analysis = ContentAnalysis.query.filter_by(video_id=video_id).first()
        if not content_analysis:
            return error_response(404, f"未找到视频ID为 {video_id} 的内容分析")
            
        # 定义评估项名称映射
        item_names = {
            "p1": "背景信息充分性",
            "p2": "背景信息准确性",
            "p3": "内容完整性",
            "p4": "意图正当性",
            "p5": "发布者信誉",
            "p6": "情感中立性",
            "p7": "行为自主性",
            "p8": "信息一致性"
        }
        
        # 获取评估分数和理由
        score = None
        reasoning = None
        
        if item == "p1":
            score = content_analysis.p1_score
            reasoning = content_analysis.p1_reasoning
        elif item == "p2":
            score = content_analysis.p2_score
            reasoning = content_analysis.p2_reasoning
        elif item == "p3":
            score = content_analysis.p3_score
            reasoning = content_analysis.p3_reasoning
        elif item == "p4":
            score = content_analysis.p4_score
            reasoning = content_analysis.p4_reasoning
        elif item == "p5":
            score = content_analysis.p5_score
            reasoning = content_analysis.p5_reasoning
        elif item == "p6":
            score = content_analysis.p6_score
            reasoning = content_analysis.p6_reasoning
        elif item == "p7":
            score = content_analysis.p7_score
            reasoning = content_analysis.p7_reasoning
        elif item == "p8":
            score = content_analysis.p8_score
            reasoning = content_analysis.p8_reasoning
            
        # 构造响应
        result = {
            "item": item,
            "name": item_names.get(item, item),
            "score": score,
            "reasoning": reasoning
        }
        
        return success_response(result)
        
    except Exception as e:
        logging.exception(f"获取视频评估项时发生错误: {str(e)}")
        return error_response(500, f"服务器内部错误: {str(e)}")