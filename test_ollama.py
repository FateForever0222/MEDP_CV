# test_ollama.py
import logging
import time
import os
import ollama
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ollama_call():
    model_name = "llama2:13B"  # 使用和您配置中相同的模型
    
    # 测试参数
    gen_params = {
        'temperature': 0.0,
        'top_p': 0.9,
        'num_predict': 1024,
    }
    
    # 简单提示词
    prompt = "What is 2+2?"
    
    logger.info("开始调用ollama.generate...")
    start_time = time.time()
    
    # 1. 使用普通调用
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options=gen_params
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"ollama.generate调用完成，耗时: {elapsed_time:.2f}秒")
    
    # 输出结果的一部分
    text = response.get('response', '')
    logger.info(f"回答前20个字符: {text[:20]}")

def test_with_env_variable():
    # 使用环境变量尝试禁用tqdm
    os.environ['TQDM_DISABLE'] = 'true'
    test_ollama_call()

def test_embeddings():
    # 测试获取嵌入向量是否也会显示进度条
    logger.info("测试获取嵌入向量...")
    
    # 如果ollama库支持获取嵌入的API
    # 这里假设有类似的API，您需要根据实际情况调整
    try:
        if hasattr(ollama, 'embeddings'):
            start_time = time.time()
            text = "This is a test sentence for embeddings."
            
            embedding = ollama.embeddings(
                model="llama2:13B",
                prompt=text
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"获取嵌入完成，耗时: {elapsed_time:.2f}秒")
    except Exception as e:
        logger.error(f"获取嵌入时出错: {e}")

if __name__ == "__main__":
    logger.info("=== 测试1: 普通调用 ===")
    test_ollama_call()
    
    logger.info("\n=== 测试2: 使用环境变量禁用tqdm ===")
    test_with_env_variable()
    
    logger.info("\n=== 测试3: 测试获取嵌入 ===")
    test_embeddings()