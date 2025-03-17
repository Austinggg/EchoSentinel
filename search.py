import os
import json
import sys
import time
from tqdm import tqdm
from utils.search_web import Search
from utils.response import search_generate, generate
from utils.crawl_web import Crawl
from utils.rrf import Retrieval

def process_query(query, date):
    """
    处理单个查询，返回结果。

    Args:
        query (str): 用户查询的问题
        date (str): 信息发布日期。

    Returns:
        str: 处理结果字符串。

    """
    # 加载配置文件
    with open(r'./config/config.json', encoding='utf-8') as f:
        config = json.load(f)

    # 是否使用本地大语言模型
    debug = config['debug']['value']
    if not debug:
        print('Warning: 当前不是debug模式，请将配置文件中的debug设置为true')
        sys.exit(0)

    start_time = time.time()
    query = f'搜索和这条消息相关的信息，信息发布日期{date}:"{query}"'

    for attempt in range(5):
        try:
            search_engine = Search()
            search_result, grade, key_words = search_engine.search(query)
            if not search_result:
                return generate(query)

            crawl = Crawl()
            web_pages = crawl.crawl(search_result)
            print(f'爬取网页条数{len(web_pages)}')

            if grade == 1 and len(web_pages) > 5:
                web_pages = web_pages[:5]

            queries = [query, key_words]
            retrieval = Retrieval()
            retrieve_results = retrieval.retrieve(web_pages, queries)

            end_time = time.time()
            print(f'搜索召回用时{end_time - start_time}')
            print('-----回答------')
            final_result = search_generate(query, retrieve_results)  # 使用云端大模型回答

            return final_result
        except Exception as e:
            if attempt == 4:
                return f"生成回答时出现错误：{str(e)}"
    return "生成回答时出现未知错误"

def update_json_files(directory):
    """
    遍历目录中的所有 JSON 文件，根据 message 和 post_time 进行搜索，并将结果写入 message_evidence 中。

    Args:
        directory (str): 目录路径。
    """
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    files = sorted(files)  # 对文件列表进行排序
    for filename in tqdm(files, desc="Processing files"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        message = data.get('message', '')
        post_time = data.get('post_time', '')

        if message and post_time:
            result = process_query(message, post_time)
            print(result)
            if result and "生成回答时出现错误" in result:
                error_directory = './data/error_file'
                os.makedirs(error_directory, exist_ok=True)
                error_filepath = os.path.join(error_directory, filename)
                os.rename(filepath, error_filepath)
            elif result:
                data['message_evidence'] = result
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                ok_directory = './data/ok'
                os.makedirs(ok_directory, exist_ok=True)
                ok_filepath = os.path.join(ok_directory, filename)
                os.rename(filepath, ok_filepath)

if __name__ == '__main__':
    directory = './data/part2'
    update_json_files(directory)
