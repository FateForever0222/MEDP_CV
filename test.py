import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 导入LLMInterface类
sys.path.append(str(Path(__file__).parent))  # 添加当前目录到路径
from src.llm.llm_interface import LLMInterface

def test_llm_interface():
    """测试LLM接口功能"""
    # 初始化接口
    llm = LLMInterface("config/config.yaml")
    
    # 测试文本生成
    print("\n=== 测试基本文本生成 ===")
    prompt = "用一句话解释为什么天空是蓝色的?"
    try:
        response = llm.generate_text(prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        print("文本生成测试成功!")
    except Exception as e:
        print(f"文本生成测试失败: {e}")
    
    # 测试带置信度的生成
    print("\n=== 测试带置信度的生成 ===")
    try:
        response, confidence = llm.generate_with_confidence(prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        print(f"置信度: {confidence}")
        print("带置信度的生成测试成功!")
    except Exception as e:
        print(f"带置信度的生成测试失败: {e}")
    
    # 测试嵌入功能
    print("\n=== 测试嵌入功能 ===")
    texts = [
        "天空为什么是蓝色的?",
        "蓝天的成因是什么?"
    ]
    try:
        embeddings = [llm.get_embedding(text) for text in texts]
        print(f"文本1: {texts[0]}")
        print(f"嵌入1维度: {len(embeddings[0])}")
        print(f"嵌入1前5个值: {embeddings[0][:5]}")
        
        print(f"文本2: {texts[1]}")
        print(f"嵌入2维度: {len(embeddings[1])}")
        print(f"嵌入2前5个值: {embeddings[1][:5]}")
        
        # 计算相似度
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity = cosine_similarity(
            [embeddings[0]], 
            [embeddings[1]]
        )[0][0]
        
        print(f"两个文本的余弦相似度: {similarity}")
        print("嵌入功能测试成功!")
    except Exception as e:
        print(f"嵌入功能测试失败: {e}")
    
    # 测试批量生成
    print("\n=== 测试批量生成 ===")
    prompts = [
        "1+1等于几?",
        "列出三个水果名称。"
    ]
    try:
        responses = llm.batch_generate(prompts)
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            print(f"提示{i+1}: {prompt}")
            print(f"响应{i+1}: {response}")
        print("批量生成测试成功!")
    except Exception as e:
        print(f"批量生成测试失败: {e}")

if __name__ == "__main__":
    test_llm_interface()