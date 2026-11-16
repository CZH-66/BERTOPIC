"""
LDA主题建模 + C_v评价
"""

import time
import random
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings("ignore")


# 随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 路径
BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "arxiv_processed.csv"
OUTPUT_DIR = BASE_DIR / "output_lda"
MODEL_DIR = BASE_DIR / "lda_model"
FIGURE_DIR = BASE_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# 参数
NUM_TOPICS = 15
TOP_N_WORDS = 10
PASSES = 20
RANDOM_STATE = 42
NO_BELOW = 10
NO_ABOVE = 0.4

# 读取数据
def load_dataset():
    print("="*60)
    print("读取CSV")
    print("="*60)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    if "processed_text" not in df.columns:
        raise ValueError("CSV缺少processed_text字段")
    docs = (
        df["processed_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    docs = [
        doc for doc in docs
        if len(doc.strip()) > 0
    ]
    print(f"论文数量：{len(docs):,}")
    return docs

# 分词
def tokenize(docs):
    print()
    print("文本分词...")
    tokenized_docs = [
        doc.split()
        for doc in docs
    ]
    return tokenized_docs

# 建立Dictionary
def build_dictionary(tokenized_docs):
    print()
    print("建立Dictionary...")
    dictionary = corpora.Dictionary(
        tokenized_docs
    )
    dictionary.filter_extremes(
        no_below=NO_BELOW,
        no_above=NO_ABOVE
    )
    print(f"词典大小：{len(dictionary):,}")
    return dictionary

# 建立Corpus
def build_corpus(
    tokenized_docs,
    dictionary
):
    print()
    print("建立Corpus...")
    corpus = [
        dictionary.doc2bow(doc)
        for doc in tokenized_docs
    ]
    return corpus

# LDA训练
def train_lda(
    corpus,
    dictionary
):
    print()
    print("="*60)
    print("开始训练LDA")
    print("="*60)
    start = time.time()
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=RANDOM_STATE,
        passes=PASSES,
        alpha="auto",
        eta="auto"
    )
    elapsed = time.time() - start
    print(f"LDA训练完成，用时 {elapsed:.2f} 秒")
    return lda_model

# 提取主题关键词
def get_topics(lda_model):
    print()
    print("提取主题关键词...")
    rows = []
    for topic_id in range(NUM_TOPICS):

        words = lda_model.show_topic(
            topic_id,
            topn=TOP_N_WORDS
        )
        keywords = ", ".join(
            [word for word, _ in words]
        )
        rows.append([
            topic_id,
            keywords
        ])
    topic_df = pd.DataFrame(
        rows,
        columns=[
            "Topic",
            "Keywords"
        ]
    )
    topic_df.to_csv(
        OUTPUT_DIR / "lda_topics.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("主题关键词已保存")
    return topic_df

# 计算C_v
def calculate_cv(
    lda_model,
    tokenized_docs,
    dictionary
):
    print()
    print("计算C_v一致性...")
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v"
    )
    cv_score = coherence_model.get_coherence()
    print(f"C_v = {cv_score:.4f}")
    return cv_score

# 保存模型
def save_model(
    lda_model,
    dictionary
):
    print()
    print("保存模型...")
    lda_model.save(
        str(MODEL_DIR / "lda_model.gensim")
    )
    dictionary.save(
        str(MODEL_DIR / "dictionary.gensim")
    )
    print("模型保存完成")

# Topic数量柱状图
def plot_topic_distribution(
    lda_model,
    corpus
):
    print()
    print("生成Topic柱状图...")
    topic_count = np.zeros(NUM_TOPICS)
    for doc in corpus:
        for topic, prob in lda_model.get_document_topics(doc):
            topic_count[topic] += prob
    plt.figure(figsize=(10,6))
    plt.bar(
        range(NUM_TOPICS),
        topic_count
    )
    plt.xticks(range(NUM_TOPICS))
    plt.xlabel("Topic")
    plt.ylabel("Weight")
    plt.title("LDA Topic Distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR /
        "lda_topic_distribution.png",
        dpi=300
    )
    plt.close()
    print("柱状图已保存")

# 保存C_v
def save_cv(cv_score):
    cv_df = pd.DataFrame({
        "Model": ["LDA"],
        "C_v": [cv_score]
    })
    cv_df.to_csv(
        OUTPUT_DIR /
        "lda_cv.csv",
        index=False,
        encoding="utf-8-sig"
    )

# 主程序
def main():
    total_start = time.time()
    docs = load_dataset()
    tokenized_docs = tokenize(docs)
    dictionary = build_dictionary(
        tokenized_docs
    )
    corpus = build_corpus(
        tokenized_docs,
        dictionary
    )
    lda_model = train_lda(
        corpus,
        dictionary
    )
    get_topics(
        lda_model
    )
    cv_score = calculate_cv(
        lda_model,
        tokenized_docs,
        dictionary
    )
    save_model(
        lda_model,
        dictionary
    )
    save_cv(
        cv_score
    )
    plot_topic_distribution(
        lda_model,
        corpus
    )
    elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("LDA全部完成")
    print("=" * 60)
    print(f"主题数量：{NUM_TOPICS}")
    print(f"C_v：{cv_score:.4f}")
    print(f"总耗时：{elapsed:.2f} 秒")
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":

    main()