# extract_api.py
from flask import Blueprint, request, jsonify
from services.content_analysis.information_extractor import InformationExtractor
from flask import current_app
from utils.HttpResponse import success_response, error_response
import logging

extract_api = Blueprint('extract', __name__)

# 初始化提取器
extractor = InformationExtractor()

@extract_api.route('/api/extract', methods=['POST'])
def extract_information():
    """
    从文本中提取关键信息和意图
    ---
    tags:
      - 信息提取
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
              description: 需要分析的文本内容
    responses:
      200:
        description: 成功提取的信息
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            data:
              type: object
              properties:
                intent:
                  type: array
                  items:
                    type: string
                statements:
                  type: array
                  items:
                    type: object
            message:
              type: string
      400:
        description: 参数错误
      500:
        description: 服务器内部错误
    """
    try:
        # 检查Content-Type
        if not request.is_json:
            logging.warning(f"收到非JSON请求: Content-Type={request.headers.get('Content-Type', '未指定')}")
            return error_response(415, "请求必须是JSON格式，请设置Content-Type为application/json")
        
        data = request.json
        if not data or 'text' not in data:
            return error_response(400, "缺少必要参数：text")
        
        result = extractor.extract_information(data['text'])
        
        if result is None:
            return error_response(500, "信息提取失败")
            
        return success_response(result)
    except Exception as e:
        logging.exception("处理提取请求时出错")
        return error_response(500, f"处理请求时发生错误: {str(e)}")