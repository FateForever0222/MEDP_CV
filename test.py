# test_ollama.py
import requests
import json

def test_ollama_call():
    """测试 Ollama API 调用"""
    url = "http://localhost:11434/api/generate"
    
    headers = {'Content-Type': 'application/json'}
    
    data = {
        'model': 'llama3',  # 或使用 'llama2:13b'
        'prompt': '如果一个袋子里有3个红球和2个蓝球，随机抽取一个球，抽到红球的概率是多少？请一步步计算。',
        'options': {
            'temperature': 0.0,
            'num_predict': 512
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        
        print("Ollama 响应:")
        print(result.get('response', '无响应'))
        return True
    except Exception as e:
        print(f"调用 Ollama 失败: {e}")
        return False

if __name__ == "__main__":
    test_ollama_call()