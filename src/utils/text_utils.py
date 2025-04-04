import re
from typing import Tuple, List

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

def format_for_csv(text: str) -> str:
    """
    格式化文本用于CSV存储，处理换行符
    
    Args:
        text: 输入文本
        
    Returns:
        格式化后的文本
    """
    if text is None:
        return ""
    
    # 将换行符替换为空格
    text = re.sub(r'\n+', ' ', text)
    
    # 将多个空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
def extract_cot_and_answer(response: str, dataset_type: str) -> Tuple[str, str]:
    """
    从LLM响应中提取思维链和最终答案，根据数据集类型采用不同的提取策略
    
    Args:
        response: LLM响应文本
        dataset_type: 数据集类型，必须是以下之一：'CSQA', 'StrategyQA', 'Letter', 'Coin', 'MultiArith', 'AQuA'
        
    Returns:
        (思维链, 最终答案)元组
        
    Raises:
        ValueError: 如果数据集类型不受支持
    """
    if not response:
        return "", ""
    
    # 验证数据集类型
    valid_datasets = ["CSQA", "StrategyQA", "Letter", "Coin", "MultiArith", "AQuA"]
    if dataset_type not in valid_datasets:
        raise ValueError(f"不支持的数据集类型: {dataset_type}。支持的类型: {', '.join(valid_datasets)}")
    
    # 首先拆分思维链和答案部分
    chain_of_thought, answer_text = _split_cot_and_answer_text(response)
    
    # 根据数据集类型使用不同的答案提取策略
    dataset_type = dataset_type.upper()
    
    # 选择题类型数据集 (CSQA, AQUA)
    if dataset_type in ["CSQA", "AQUA"]:
        return chain_of_thought, _extract_multiple_choice_answer(answer_text)
        
    # 数学问题类型数据集 (MultiArith)
    elif dataset_type == "MULTIARITH":
        return chain_of_thought, _extract_numerical_answer(answer_text)
        
    # 是非题类型数据集 (StrategyQA, Coin)
    elif dataset_type in ["STRATEGYQA", "COIN"]:
        return chain_of_thought, _extract_yes_no_answer(answer_text)
        
    # 字符串答案类型数据集 (Letter)
    elif dataset_type == "LETTER":
        return chain_of_thought, _extract_string_answer(answer_text)
    
def _split_cot_and_answer_text(response: str) -> Tuple[str, str]:
    """
    拆分思维链和答案文本，但保留答案句在思维链中
    
    Args:
        response: LLM响应文本
        
    Returns:
        (思维链, 答案文本)元组，其中思维链包含完整文本
    """
    # 设置直接答案触发词
    answer_markers = [
        "Answer:", "Therefore,", "So the answer is", "The answer is", "Hence,", 
        "In conclusion,", "Thus,", "So,", "We get", "Finally,"
    ]
    
    # 标记匹配的位置
    marker_pos = -1
    matched_marker = ""
    
    for marker in answer_markers:
        pos = response.lower().find(marker.lower())
        if pos > marker_pos:
            marker_pos = pos
            matched_marker = marker
    
    if marker_pos > 0:
        # 找到了标记，提取答案部分但保留完整文本作为思维链
        answer_part = response[marker_pos + len(matched_marker):].strip()
        # 返回完整文本作为思维链
        return response.strip(), answer_part
    
    # 如果没有找到标记，尝试使用最后一段落
    paragraphs = [p for p in re.split(r'\n\s*\n', response) if p.strip()]
    if len(paragraphs) > 1:
        answer_part = paragraphs[-1].strip()
        # 返回完整文本作为思维链
        return response.strip(), answer_part
    
    # 如果只有一个段落，尝试最后一行
    lines = response.strip().split('\n')
    if len(lines) > 1:
        answer_part = lines[-1].strip()
        # 返回完整文本作为思维链
        return response.strip(), answer_part
    
    # 如果都失败了，返回整个文本
    return response.strip(), response.strip()

def _extract_multiple_choice_answer(answer_text: str) -> str:
    """提取选择题答案 (A/B/C/D/E)"""
    # 1. 直接匹配"答案是X"模式
    option_match = re.search(r'(?:answer is|选择|option)\s*(?:is|:|：)?\s*([A-E])', answer_text, re.IGNORECASE)
    if option_match:
        return option_match.group(1).upper()
    
    # 2. 匹配答案的最后提到的单个选项字母
    option_matches = re.findall(r'\b([A-E])\b', answer_text.upper())
    if option_matches:
        return option_matches[-1]  # 使用最后一个匹配，通常是最终答案
    
    # 3. 寻找括号中的选项
    bracket_match = re.search(r'\(([A-E])\)', answer_text, re.IGNORECASE)
    if bracket_match:
        return bracket_match.group(1).upper()
    
    # 如果以上都失败，尝试从整个文本中提取任何可能的选项标识符
    all_options = ''.join(re.findall(r'[A-E]', answer_text.upper()))
    if all_options:
        return all_options[-1]  # 返回最后一个选项标识符
    
    return "A"  # 如果无法提取，返回默认选项

def _extract_numerical_answer(answer_text: str) -> str:
    """提取数值型答案 (MultiArith数据集)"""
    # 首先检查是否有"答案是"之类的短语后跟数字
    num_phrase_match = re.search(r'(?:answer is|=|equals?)[:\s]*(-?\d[\d\.,]*)', answer_text, re.IGNORECASE)
    if num_phrase_match:
        # 去除逗号并处理小数点
        return num_phrase_match.group(1).replace(",", "")
    
    # 否则提取所有数字，选择最后一个（通常是最终答案）
    number_matches = re.findall(r'(-?\d[\d\.,]*)', answer_text.replace(",", ""))
    if number_matches:
        result = number_matches[-1]
        # 去除尾部的.0
        if result.endswith('.0'):
            result = result[:-2]
        return result
    
    return "0"  # 如果无法提取，返回默认值

def _extract_yes_no_answer(answer_text: str) -> str:
    """提取是非题答案 (StrategyQA和Coin数据集)"""
    # 首先查找明确的YES或NO大写答案
    yes_no_upper = re.search(r'\b(YES|NO)\b', answer_text)
    if yes_no_upper:
        return yes_no_upper.group(1).lower()
    
    # 搜索常规yes或no
    yes_no_match = re.search(r'\b(yes|no)\b', answer_text.lower())
    if yes_no_match:
        return yes_no_match.group(1).lower()
    
    # 搜索"答案是yes/no"模式
    yn_phrase_match = re.search(r'(?:answer is|:|：)\s*(yes|no)', answer_text, re.IGNORECASE)
    if yn_phrase_match:
        return yn_phrase_match.group(1).lower()
    
    # 检查文本是否包含肯定/否定的指示词
    if re.search(r'\b(yes|correct|right|true|确实|是的|still|heads up)\b', answer_text.lower()):
        return "yes"
    if re.search(r'\b(no|incorrect|wrong|false|不是|否|not)\b', answer_text.lower()):
        return "no"
    
    return "no"  # 默认返回no

def _extract_string_answer(answer_text: str) -> str:
    """提取字符串类型答案 (Letter数据集)"""
    # 尝试提取引号中的内容
    quote_match = re.search(r'[\'"]([^\'"]+)[\'"]', answer_text)
    if quote_match:
        return quote_match.group(1).strip()
    
    # 尝试找到"is"后面的字母序列
    is_match = re.search(r'(?:answer is|:|：)\s*"?([a-zA-Z]+)"?', answer_text, re.IGNORECASE)
    if is_match:
        return is_match.group(1).strip()
    
    # 根据看到的样例，答案通常是大写字母组合，尝试提取
    uppercase_seq = re.search(r'\b([A-Z]{2,})\b', answer_text)
    if uppercase_seq:
        return uppercase_seq.group(1)
    
    # 去除所有标点和空格，保留字母
    clean_text = re.sub(r'[^a-zA-Z]', '', answer_text)
    if clean_text:
        return clean_text
    
    return answer_text
def check_answer_correctness(generated: str, correct: str) -> bool:
    """
    检查生成的答案是否正确
    
    Args:
        generated: 生成的答案
        correct: 正确答案
        
    Returns:
        是否正确
    """
    if not correct or not generated:
        return False
    
    # 标准化答案
    norm_generated = normalize_answer(generated)
    norm_correct = normalize_answer(correct)
    
    # 直接比较清理后的答案
    if norm_generated == norm_correct:
        return True
    
    # 处理多选题的情况（如ABCDE）
    if len(norm_correct) <= 3 and re.match(r'^[a-e]$', norm_correct):
        option_matches = re.findall(r'\b([a-e])\b', norm_generated)
        return any(match.lower() == norm_correct for match in option_matches)
    
    # 处理数值型答案
    if re.match(r'^-?\d+\.?\d*$', norm_correct):
        # 提取生成答案中的数字
        number_matches = re.findall(r'-?\d+\.?\d*', norm_generated)
        return any(match == norm_correct for match in number_matches)
    
    # 处理是非题
    if norm_correct in ["yes", "no"]:
        return norm_correct in norm_generated
    
    # 对于其他类型的答案，检查正确答案是否包含在生成答案中
    return norm_correct in norm_generated

def count_reasoning_steps(cot: str) -> int:
    """
    计算思维链中的推理步骤数
    
    Args:
        cot: 思维链文本
        
    Returns:
        步骤数
    """
    if not cot:
        return 0
        
    # 更全面的步骤标记正则表达式
    step_markers = [
        r"Step\s+\d+[\.:]?", 
        r"\d+\.\s+", 
        r"\(\d+\)\s+", 
        r"First[,:]?\s+", "Second[,:]?\s+", "Third[,:]?\s+", "Fourth[,:]?\s+", "Fifth[,:]?\s+",
        r"Next[,:]?\s+", r"Then[,:]?\s+", r"Finally[,:]?\s+",
        r"To start[,:]?\s+", r"Initially[,:]?\s+", r"Lastly[,:]?\s+",
        r"Let's\s+first", r"Let's\s+now", r"We\s+can\s+now"
    ]
    
    # 尝试通过步骤标记计数
    lines = cot.split('\n')
    step_count = 0
    prev_step_line = -1
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        for marker in step_markers:
            if re.match(marker, line, re.IGNORECASE):
                # 确保这是一个新步骤（不是同一步骤的延续）
                if i > prev_step_line + 1 or prev_step_line == -1:
                    step_count += 1
                    prev_step_line = i
                    break
    
    # 如果通过标记找到的步骤数少于2，尝试通过段落计数
    if step_count < 2:
        # 更智能地分割段落，避免误判
        paragraphs = []
        current_para = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
            else:
                current_para.append(line)
        
        if current_para:
            paragraphs.append('\n'.join(current_para))
        
        # 过滤掉可能只是问题陈述或者非推理步骤的段落
        filtered_paras = []
        for para in paragraphs:
            # 跳过非推理的段落，如问题重述
            if re.search(r'question|asked|problem states', para, re.IGNORECASE):
                continue
            # 跳过答案总结
            if any(re.search(marker, para, re.IGNORECASE) for marker in ["answer is", "therefore", "thus", "hence"]):
                continue
            filtered_paras.append(para)
        
        # 使用过滤后的段落计数
        step_count = len(filtered_paras)
    
    # 确保步骤数至少为1
    return max(1, step_count)