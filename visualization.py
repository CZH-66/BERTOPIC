"""
BERTopic 可视化分析
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from wordcloud import WordCloud

# 路径
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
#MODEL_DIR = OUTPUT_DIR / "bertopic_model"
MODEL_DIR = BASE_DIR / "model" / "bertopic_model"
OUTPUT_LDA_DIR = BASE_DIR / "output_lda"
FIGURE_DIR = BASE_DIR / "figures"

FIGURE_DIR.mkdir(exist_ok=True)
TOPIC_INFO = OUTPUT_DIR / "topic_info.csv"
DOCUMENT_INFO = OUTPUT_DIR / "document_info.csv"
TOPIC_TIME = OUTPUT_DIR / "topics_over_time.csv"
LDA_CV = OUTPUT_LDA_DIR / "lda_cv.csv"

# Matplotlib
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False

# 数据读取
def load_topic_info():
    if not TOPIC_INFO.exists():
        raise FileNotFoundError(TOPIC_INFO)
    df = pd.read_csv(str(TOPIC_INFO))
    print(f"Topic数量：{len(df)}")
    return df

def load_document_info():
    if not DOCUMENT_INFO.exists():
        raise FileNotFoundError(DOCUMENT_INFO)
    df = pd.read_csv(str(DOCUMENT_INFO))
    print(f"文档数量：{len(df)}")
    return df

def load_topics_over_time():
    if not TOPIC_TIME.exists():
        print("未找到 topics_over_time.csv")
        return None
    return pd.read_csv(str(TOPIC_TIME))

def load_lda_cv():
    if not LDA_CV.exists():
        print("未找到 lda_cv.csv")
        return None
    return pd.read_csv(str(LDA_CV))

# BERTopic C_v
def load_bertopic_cv():
    cv_file = OUTPUT_DIR / "bertopic_cv.csv"
    if not cv_file.exists():
        # 只打印一次警告（通过全局标志或简单处理）
        if not hasattr(load_bertopic_cv, "warned"):
            print("警告：未找到 bertopic_cv.csv，请先运行 modeling.py 生成BERTopic C_v值。")
            load_bertopic_cv.warned = True
        return None
    df = pd.read_csv(str(cv_file))
    return float(df.iloc[0]["C_v"])

# 保存图片
def save_figure(name):
    plt.tight_layout()
    plt.savefig(
        str(FIGURE_DIR / name),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"已保存：{name}")

# 删除异常Topic
def remove_outlier(df):
    if "Topic" not in df.columns:
        return df
    df = df[df["Topic"] != -1]
    return df.reset_index(drop=True)

# Topic数量柱状图
def plot_topic_distribution(topic_info):
    print()
    print("生成 Topic 数量柱状图...")
    df = remove_outlier(topic_info)
    if "Count" not in df.columns:
        print("topic_info.csv 缺少 Count 字段")
        return
    plt.figure(figsize=(10,6))

    label_col = "Name" if "Name" in df.columns else "Topic"

    plt.figure(figsize=(12,6))
    plt.bar(
        df[label_col],
        df["Count"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Topic")
    plt.ylabel("Documents")
    plt.title("Topic Distribution")

    plt.xlabel("Topic")
    plt.ylabel("Documents")
    plt.title("BERTopic Topic Distribution")
    save_figure("topic_distribution.png")

# Top10 Topic 横向柱状图
def plot_top10_topics(topic_info):
    print()
    print("生成 Top10 Topic 图...")
    df = remove_outlier(topic_info)
    if "Count" not in df.columns:
        return
    df = df.sort_values(
        "Count",
        ascending=False
    ).head(10)
    plt.figure(figsize=(9,6))

    label_col = "Name" if "Name" in df.columns else "Topic"
    plt.barh(
        df[label_col],
        df["Count"]
    )
    plt.xlabel("Documents")
    plt.ylabel("Topic")

    plt.xlabel("Documents")
    plt.ylabel("Topic")
    plt.title("Top 10 Largest Topics")
    plt.gca().invert_yaxis()
    save_figure("top10_topics.png")

# Topic占比饼图
def plot_topic_pie(topic_info):
    print()
    print("生成 Topic 饼图...")
    df = remove_outlier(topic_info)
    if "Count" not in df.columns:
        return
    df = df.sort_values(
        "Count",
        ascending=False
    ).head(8)

    others = topic_info["Count"].sum() - df["Count"].sum()
    if others > 0:
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    "Topic":["Others"],
                    "Count":[others]
                })
            ],
            ignore_index=True
        )
    label_col = "Name" if "Name" in df.columns else "Topic"
    plt.figure(figsize=(7,7))
    plt.pie(
        df["Count"],
        labels=df[label_col],
        autopct="%.1f%%",
        startangle=90
    )
    plt.title("Topic Proportion")
    plt.axis("equal")
    save_figure("topic_pie.png")

# 年份趋势折线图
def plot_topic_trend(topics_over_time, topic_info):

    print()
    print("生成年份趋势折线图...")

    if topics_over_time is None:
        print("没有 topics_over_time.csv")
        return

    if len(topics_over_time) == 0:
        print("topics_over_time.csv 为空")
        return

    df = topics_over_time.copy()

    if "Topic" in df.columns:
        df = df[df["Topic"] != -1]

    # 找年份字段
    year_col = None
    for col in ["Timestamp", "Year", "year"]:
        if col in df.columns:
            year_col = col
            break

    if year_col is None:
        print("未找到年份字段")
        return

    # 找数量字段
    count_col = None
    for col in ["Frequency", "Count"]:
        if col in df.columns:
            count_col = col
            break

    if count_col is None:
        print("未找到Frequency字段")
        return

    # 取文档数量最多的前5个主题
    top_topics = (
        df.groupby("Topic")[count_col]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # 建立 Topic编号 → Topic名称 映射
    topic_name = dict(
        zip(
            topic_info["Topic"],
            topic_info["Name"]
        )
    )

    plt.figure(figsize=(10,6))

    for topic in top_topics:

        temp = df[df["Topic"] == topic]

        plt.plot(
            temp[year_col],
            temp[count_col],
            marker="o",
            linewidth=2,
            label=topic_name.get(topic, f"Topic {topic}")
        )

    plt.xlabel("Year")
    plt.ylabel("Documents")
    plt.title("Topic Trends Over Time")

    plt.legend()

    save_figure("topic_trend.png")

# Topic词云图
FONT_PATH = None
# 例如：
# FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

def plot_wordcloud(topic_info):
    print()
    print("生成Topic词云...")
    df = remove_outlier(topic_info)
    word_freq = {}
    keyword_col = None

    for col in ["Representation", "Name"]:
        if col in df.columns:
            keyword_col = col
            break
    if keyword_col is None:
        print("没有找到关键词字段")
        return
    for _, row in df.iterrows():
        text = str(row[keyword_col])
        text = (
            text.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace('"', "")
        )
        words = text.split(",")
        weight = row["Count"] if "Count" in row else 1
        for w in words:
            w = w.strip()
            if len(w) < 2:
                continue
            word_freq[w] = word_freq.get(w, 0) + weight

    wc = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        font_path=FONT_PATH,
        max_words=100,
        random_state=42
    )
    wc.generate_from_frequencies(word_freq)
    plt.figure(figsize=(12,7))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("BERTopic WordCloud")
    save_figure("topic_wordcloud.png")

# BERTopic vs LDA C_v 对比图
def plot_cv_compare(bertopic_cv=None, lda_cv=None):
    print()
    print("生成 C_v 对比图...")
    if bertopic_cv is None or lda_cv is None:
        print("缺少C_v数据，跳过对比图。")
        return
    plt.figure(figsize=(6,5))
    models = ["BERTopic", "LDA"]
    scores = [bertopic_cv, lda_cv]
    bars = plt.bar(models, scores)
    plt.ylabel("C_v")
    plt.title("Topic Coherence Comparison")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, score+0.01, f"{score:.3f}", ha="center", fontsize=10)
    save_figure("cv_compare.png")

# BERTopic 官方气泡图
def plot_topic_bubble():
    print()
    print("生成 BERTopic 官方气泡图...")
    if not MODEL_DIR.exists():
        print("没有找到 bertopic_model")
        return
    topic_model = BERTopic.load(str(MODEL_DIR))
    fig = topic_model.visualize_topics(
        top_n_topics=20
    )
    fig.write_html(
        str(FIGURE_DIR / "topic_bubble.html")
    )
    try:
        fig.write_image(
            str(FIGURE_DIR / "topic_bubble.png"),
            width=1600,
            height=900
        )
        print("已保存 topic_bubble.png")
    except Exception:
        print("未安装 kaleido，仅保存HTML")

# 主程序
def main():
    print("=" * 60)
    print("BERTopic 可视化")
    print("=" * 60)
    topic_info = load_topic_info()
    document_info = load_document_info()
    topics_over_time = load_topics_over_time()
    bertopic_cv = load_bertopic_cv()      # 返回 float 或 None
    lda_df = load_lda_cv()                # 返回 DataFrame 或 None
    if lda_df is not None and not lda_df.empty:
        lda_cv = float(lda_df.iloc[0]["C_v"])
    else:
        lda_cv = None
    print()
    plot_topic_distribution(topic_info)
    plot_top10_topics(topic_info)
    plot_topic_pie(topic_info)
    plot_topic_trend(topics_over_time, topic_info)
    plot_wordcloud(topic_info)
    plot_topic_bubble()
    plot_cv_compare(bertopic_cv, lda_cv) 
    print()
    print("=" * 60)
    print("全部图片生成完成！")
    print("=" * 60)
    print()
    print("保存目录：")
    print(FIGURE_DIR)
    print()
    print("生成文件：")
    print("① topic_distribution.png")
    print("② top10_topics.png")
    print("③ topic_pie.png")
    print("④ topic_trend.png")
    print("⑤ topic_wordcloud.png")
    print("⑥ cv_compare.png")
    print()
    print("可直接用于论文和PPT。")




if __name__ == "__main__":

    main()