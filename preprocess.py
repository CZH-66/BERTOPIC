"""
数据预处理模块（v4.0）
- 固定随机种子确保可重复
- 过滤 stat.ML 和 eess.IV（统计时忽略）
- CSV 增加 doc_id 列
- 列名 year -> published_year
- 完整优化：流式读取、随机抽样、文本清洗、NLTK 处理
"""

import json
import re
import time
import random
from pathlib import Path
from datetime import datetime
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
"""""
# ---------- 确保 NLTK 资源已下载 ----------
def ensure_nltk_resources():
    resources = ['stopwords', 'wordnet', 'punkt', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            print(f"正在下载 NLTK 资源: {res}")
            nltk.download(res, quiet=False)


ensure_nltk_resources()
"""

# ---------- 配置参数 ----------
CONFIG = {
    "input_json": "arxiv-metadata-oai-snapshot.json",      # 原始 JSON
    "output_csv": "arxiv_processed.csv",                   # 输出 CSV
    "years_back": 5,                                       # 保留近 N 年
    "max_samples": 5000,                                   # 最终保留论文数（抽样）
    "min_abstract_length": 80,                             # 摘要最短字符数
    "target_categories": {                                 # 计算机科学核心子领域
        "cs.AI", "cs.LG", "cs.CL", "cs.CV",
        "cs.IR", "cs.NE", "cs.RO", "cs.CR", "cs.DS"
    },
    "exclude_categories": {"stat.ML", "eess.IV"},          # 统计时忽略的类别
    "print_interval": 50000,                               # 扫描进度打印间隔
    "random_seed": 42,                                     # 固定随机种子
}

# ---------- 初始化 NLTK ----------
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------- 文本清洗 ----------
def clean_text(text: str) -> str:
    """基础清洗：去 URL、邮箱、数字、特殊字符，保留英文字母"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- 文本预处理（分词+词形还原） ----------
def preprocess_text(text: str):
    """返回 (processed_text, tokens)"""
    text = clean_text(text)
    if not text:
        return "", []
    try:
        tokens = word_tokenize(text)
    except:
        return "", []
    processed = []
    for token in tokens:
        if len(token) < 2 or not token.isalpha():
            continue
        if token in STOP_WORDS:
            continue
        lemma = LEMMATIZER.lemmatize(token, pos="v")
        if len(lemma) < 2:
            continue
        processed.append(lemma)
    if not processed:
        return "", []
    return " ".join(processed), processed

# ---------- 流式扫描 JSON ----------
def scan_papers(config):
    """生成器：筛选符合条件（类别、年份、摘要长度）的论文"""
    input_path = Path(config["input_json"])
    if not input_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {config['input_json']}")

    current_year = datetime.now().year
    min_year = current_year - config["years_back"]
    target_cats = config["target_categories"]
    min_abstract_len = config["min_abstract_length"]

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % config["print_interval"] == 0:
                print(f"已扫描 {line_num:,} 条记录...")

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 类别筛选（至少包含一个目标类别）
            categories_str = paper.get("categories", "")
            if not categories_str:
                continue
            cat_set = set(categories_str.split())
            if not (cat_set & target_cats):
                continue

            # 摘要长度
            abstract = paper.get("abstract", "").strip()
            if len(abstract) < min_abstract_len:
                continue

            # 发布日期（取第一版本）
            versions = paper.get("versions", [])
            if not versions:
                continue
            created = versions[0].get("created", "")
            if not created:
                continue
            try:
                pub_date = pd.to_datetime(created, utc=False, errors="coerce")
            except:
                continue
            if pd.isna(pub_date) or pub_date.year < min_year:
                continue

            yield {
                "title": paper.get("title", "").strip(),
                "abstract": abstract,
                "categories": " ".join(cat_set),
                "published_year": pub_date.year,   # 改为此列名
            }

# ---------- 构建 DataFrame ----------
def build_dataframe(config):
    print("=" * 60)
    print("开始扫描并筛选 arXiv 论文...")
    print("=" * 60)

    # 扫描
    papers = []
    start_time = time.time()
    for paper in scan_papers(config):
        papers.append(paper)

    elapsed = time.time() - start_time
    total_found = len(papers)
    print(f"扫描完成，共找到 {total_found:,} 篇符合条件的论文。")

    if total_found == 0:
        return pd.DataFrame()

    # 固定随机种子并抽样
    random.seed(config["random_seed"])
    max_samples = config["max_samples"]
    if max_samples > 0 and total_found > max_samples:
        sampled_papers = random.sample(papers, max_samples)
        print(f"随机抽样至 {max_samples:,} 篇 (种子={config['random_seed']})。")
    else:
        sampled_papers = papers

    # 文本预处理
    print("开始文本预处理（分词、词形还原）...")
    records = []
    for idx, paper in enumerate(sampled_papers, 1):
        processed_text, tokens = preprocess_text(paper["abstract"])
        if not processed_text:
            continue
        records.append({
            "doc_id": len(records),                 # 从0开始自动编号
            "title": paper["title"],
            "abstract": paper["abstract"],
            "processed_text": processed_text,
            "tokens": tokens,
            "token_count": len(tokens),
            "published_year": paper["published_year"],
            "categories": paper["categories"],
        })
        if idx % 500 == 0:
            print(f"已处理 {idx}/{len(sampled_papers)} 篇")

    df = pd.DataFrame(records)
    print(f"预处理完成，最终保留 {len(df):,} 篇有效论文。")
    return df

# ---------- 统计信息（过滤 stat.ML 和 eess.IV） ----------
def show_statistics(df, config):
    if df.empty:
        return

    exclude = config["exclude_categories"]

    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"总论文数: {len(df):,}")

    # 年份分布
    print("\n年份分布:")
    print(df["published_year"].value_counts().sort_index())

    # 类别统计（过滤掉排除的类别）
    cat_counter = {}
    for cats in df["categories"]:
        for c in cats.split():
            if c in exclude:
                continue
            cat_counter[c] = cat_counter.get(c, 0) + 1
    if cat_counter:
        cat_df = pd.DataFrame(cat_counter.items(), columns=["Category", "Count"])
        cat_df = cat_df.sort_values("Count", ascending=False)
        print("\n类别分布（Top 10，已过滤 stat.ML 和 eess.IV）:")
        print(cat_df.head(10))
    else:
        print("\n没有其他类别统计。")

    # Token 统计
    print("\nToken 数量统计:")
    print(df["token_count"].describe())

    # 样例
    print("\n样例数据（前3条标题及 doc_id）:")
    print(df[["doc_id", "title"]].head(3).to_string(index=False))

# ---------- 保存 CSV ----------
def save_dataframe(df, output_path):
    if df.empty:
        print("没有数据可保存。")
        return
    # 确保 doc_id 列存在且为整数
    if "doc_id" not in df.columns:
        df.insert(0, "doc_id", range(len(df)))
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV 已保存至: {output_path} (共 {len(df):,} 篇)")

# ---------- 主程序 ----------
def main():
    print("=" * 60)
    print("BERTopic 数据预处理 (v4.0)")
    print("=" * 60)

    input_path = Path(CONFIG["input_json"])
    if not input_path.exists():
        print(f"\n❌ 错误: 找不到数据文件 '{CONFIG['input_json']}'")
        print("请检查文件路径或修改 CONFIG。")
        return

    file_size_gb = input_path.stat().st_size / 1024**3
    print(f"\n数据文件: {CONFIG['input_json']} ({file_size_gb:.2f} GB)")
    print(f"目标类别: {', '.join(sorted(CONFIG['target_categories']))}")
    print(f"保留近 {CONFIG['years_back']} 年")
    print(f"目标论文数: {CONFIG['max_samples']:,}")
    print(f"随机种子: {CONFIG['random_seed']}")
    print(f"统计时忽略类别: {', '.join(CONFIG['exclude_categories'])}")

    start_all = time.time()
    df = build_dataframe(CONFIG)

    if df.empty:
        print("\n⚠️ 没有符合条件的数据，请检查配置或数据文件。")
        return

    show_statistics(df, CONFIG)
    save_dataframe(df, CONFIG["output_csv"])

    elapsed = time.time() - start_all
    print(f"\n总耗时: {elapsed:.2f} 秒")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序。")
    except Exception as e:
        print("\n" + "=" * 60)
        print("程序异常终止")
        print("=" * 60)
        import traceback
        traceback.print_exc()