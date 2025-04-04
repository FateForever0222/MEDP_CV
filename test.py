from sentence_transformers import SentenceTransformer

# 指定本地模型路径
model_path = './model/all-MiniLM-L6-v2/'

# 加载本地模型
model = SentenceTransformer(model_path)

# 测试句子
sentences = [
    "This is an example sentence.",
    "Sentence embeddings are useful for many NLP tasks.",
    "How does sentence-transformers work?"
]

# 生成嵌入
embeddings = model.encode(sentences)

# 打印每个句子的嵌入结果
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding[:10]}...")  # 只打印前10个值，防止输出过多
    print()
