import re

def normalize_answer(text: str) -> str:
    """
    标准化答案文本，用于比较
    
    Args:
        text: 输入文本
        
    Returns:
        标准化后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 转换为小写
    text = text.lower()
    
    # 移除标点符号和多余空格
    text = re.sub(r'[^\w\s]', '', text)
    
    # 去除首尾空格并压缩内部空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text