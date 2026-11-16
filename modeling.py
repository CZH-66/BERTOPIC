"""
BERTopic主题建模
"""
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

SEED = 42  # 随机种子
random.seed(SEED)
np.random.seed(SEED)


# 路径
BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "arxiv_processed.csv"
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
MODEL_DIR = BASE_DIR / "model"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 参数
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 15
TOP_N_WORDS = 10
NGRAM_RANGE = (1, 2)

# 读取数据

def load_dataset():
    print("=" * 60)
    print("读取CSV")
    print("=" * 60)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"找不到文件:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)
    required_cols = [
        "title",
        "processed_text",
        "published_year"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"缺少字段: {col}"
            )
    df = df.dropna(
        subset=["processed_text"]
    )
    df = df.reset_index(drop=True)
    print(f"论文数量: {len(df):,}")
    return df

# 创建BERTopic
def build_model():
    print()
    print("加载Embedding模型...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )
    print("初始化UMAP...")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED
    )
    print("初始化HDBSCAN...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        prediction_data=True
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=NGRAM_RANGE
    )
    print("创建BERTopic...")
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=MIN_TOPIC_SIZE,
        top_n_words=TOP_N_WORDS,
        calculate_probabilities=True,
        verbose=True
    )
    return model


# 模型训练
def train_model(model, df):
    print()
    print("=" * 60)
    print("开始训练")
    print("=" * 60)
    docs = df["processed_text"].tolist()
    start_time = time.time()
    topics, probs = model.fit_transform(
        docs
    )
    elapsed = time.time() - start_time
    print()
    print(f"训练完成，耗时 {elapsed:.2f} 秒")
    return topics, probs

# 保存模型
def save_model(model):
    print()
    print("保存BERTopic模型...")
    save_path = MODEL_DIR / "bertopic_model"
    try:
        model.save(
            str(save_path),
            serialization="safetensors",
            save_ctfidf=True
        )
    except Exception:
        model.save(str(save_path))
    print("模型保存完成")


# 保存CSV

def save_results(model, df):
    print()
    print("保存CSV结果...")
    topic_info = model.get_topic_info()
    topic_info.to_csv(
        OUTPUT_DIR / "topic_info.csv",
        index=False,
        encoding="utf-8-sig"
    )
    document_info = model.get_document_info(
        df["processed_text"].tolist()
    )
    document_info["title"] = df["title"]
    document_info["published_year"] = df["published_year"]
    document_info.to_csv(
        OUTPUT_DIR / "document_info.csv",
        index=False,
        encoding="utf-8-sig"
    )
    freq = model.get_topic_freq()
    freq.to_csv(
        OUTPUT_DIR / "topic_frequency.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("CSV保存完成")
    return topic_info

def compute_bertopic_cv(model, docs, dictionary=None):
    print()
    print("计算BERTopic C_v一致性...")
    # 分词（按空格切分）
    tokenized_docs = [doc.split() for doc in docs]
    
    # 建立词典（与LDA保持一致：过滤 extremes）
    if dictionary is None:
        dictionary = Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=10, no_above=0.4)   # 与lda_model.py参数一致
    
    # 获取所有有效主题ID（排除 -1 离群点）
    topic_info = model.get_topic_info()
    valid_topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    
    # 获取每个主题的前10个词（与 TOP_N_WORDS 一致）
    topic_words = []
    for topic_id in valid_topic_ids:
        # get_topic() 返回 [(word, score), ...]，直接切片取前10
        words = [word for word, _ in model.get_topic(topic_id)[:TOP_N_WORDS]]
        topic_words.append(words)
    
    # 计算C_v
    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    cv_score = cm.get_coherence()
    print(f"BERTopic C_v = {cv_score:.4f}")
    return cv_score

# Topic Over Time
def save_topics_over_time(model, df, topics):
    print()
    print("生成Topic Over Time...")
    try:
        topics_over_time = model.topics_over_time(
            docs=df["processed_text"].tolist(),
            topics=topics,
            timestamps=df["published_year"].astype(str)
        )
        topics_over_time.to_csv(
            OUTPUT_DIR / "topics_over_time.csv",
            index=False,
            encoding="utf-8-sig"
        )
        fig = model.visualize_topics_over_time(
            topics_over_time
        )
        fig.write_html(
            FIGURE_DIR / "topics_over_time.html"
        )
        print("Topic Over Time完成")
    except Exception as e:
        print("Topic Over Time生成失败：", e)

# HTML可视化
def save_visualizations(model):
    print()
    print("生成HTML图...")
    figures = [
        ("topic_map.html", model.visualize_topics),
        ("hierarchy.html", model.visualize_hierarchy),
        ("barchart.html", model.visualize_barchart),
        ("heatmap.html", model.visualize_heatmap),
    ]

    for filename, func in figures:
        try:
            fig = func()
            fig.write_html(
                FIGURE_DIR / filename
            )
            print(f"{filename} 已保存")
        except Exception as e:
            print(f"{filename} 生成失败：{e}")

# 输出统计
def print_summary(topic_info):
    print()
    print("=" * 60)
    print("统计信息")
    print("=" * 60)
    valid = topic_info[topic_info.Topic != -1]
    print(f"主题数量：{len(valid)}")
    print()
    print(valid[
        ["Topic", "Count", "Name"]
    ].head(10))
    print()
    print("结果目录：")
    print(OUTPUT_DIR)
    print(FIGURE_DIR)
    print(MODEL_DIR)

# 主程序
def main():
    total_start = time.time()
    try:
        df = load_dataset()
        model = build_model()
        topics, probs = train_model(
            model,
            df
        )
        save_model(model)
        # 计算C_v并保存
        cv_score = compute_bertopic_cv(model, df["processed_text"].tolist())
        cv_df = pd.DataFrame({"Model": ["BERTopic"], "C_v": [cv_score]})
        cv_df.to_csv(OUTPUT_DIR / "bertopic_cv.csv", index=False, encoding="utf-8-sig")
        print(f"BERTopic C_v 已保存至 {OUTPUT_DIR / 'bertopic_cv.csv'}")
        
        topic_info = save_results(
            model,
            df
        )
        save_topics_over_time(
            model,
            df,
            topics
        )
        save_visualizations(model)
        print_summary(topic_info)
        elapsed = time.time() - total_start
        print()
        print("=" * 60)
        print("全部完成")
        print("=" * 60)
        print(f"总耗时：{elapsed:.2f} 秒")

    except Exception as e:
        print()
        print("=" * 60)
        print("程序运行失败")
        print("=" * 60)
        print(e)
        raise

if __name__ == "__main__":

    main()