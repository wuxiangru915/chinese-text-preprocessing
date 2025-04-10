# -*- coding: utf-8 -*-
"""
文本预处理实验

实现功能：
1. 中文分词（精确模式、搜索引擎模式）
2. 停用词过滤
3. 词频统计
4. 关键词提取（TF-IDF和TextRank）
"""

import os
import jieba
import jieba.analyse
from collections import Counter
import re

# 设置工作目录和文件路径
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORK_DIR, 'input.txt')
STOPWORDS_FILE = os.path.join(WORK_DIR, 'stopwords.txt')
WORD_FREQ_FILE = os.path.join(WORK_DIR, 'word_frequency.txt')
TOP_KEYWORDS_TFIDF_FILE = os.path.join(WORK_DIR, 'top_keywords_tfidf.txt')
TOP_KEYWORDS_TEXTRANK_FILE = os.path.join(WORK_DIR, 'top_keywords_textrank.txt')

# 默认停用词集合
DEFAULT_STOPWORDS = {
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
    '或', '一个', '没有', '我们', '你们', '他们', '她们', '它们',
    '这个', '那个', '这些', '那些', '不', '在', '人', '我', '有',
    '个', '好', '来', '去', '也', '很', '但', '吧', '啊', '呢', '啦'
}


def read_file(file_path, default_return=None, create_default=False):
    """通用文件读取函数，支持不同编码和默认值"""
    try:
        # 尝试UTF-8编码
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # 尝试GBK编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            return default_return
    except FileNotFoundError:
        if create_default and default_return:
            try:
                # 创建默认文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(default_return)
                print(f"已创建默认文件: {file_path}")
                return default_return
            except Exception as e:
                print(f"创建默认文件 {file_path} 失败: {e}")
        else:
            print(f"文件 {file_path} 不存在")
        return default_return
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return default_return


def write_file(file_path, content):
    """通用文件写入函数"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"写入文件 {file_path} 时出错: {e}")
        return False


def load_stopwords():
    """加载停用词表，如果不存在则创建默认停用词表"""
    # 尝试读取停用词文件
    content = read_file(STOPWORDS_FILE)
    if content:
        return set(line.strip() for line in content.splitlines() if line.strip())
    
    # 如果文件不存在，创建默认停用词表
    print(f"停用词文件 {STOPWORDS_FILE} 不存在，将创建默认停用词表")
    default_content = '\n'.join(sorted(DEFAULT_STOPWORDS))
    if write_file(STOPWORDS_FILE, default_content):
        print(f"已创建默认停用词表: {STOPWORDS_FILE}")
        return DEFAULT_STOPWORDS
    return set()


def preprocess_text(text):
    """文本预处理：去除特殊字符、数字、标点符号等"""
    # 使用一个正则表达式链完成多个替换
    text = re.sub(r'<[^>]+>|http[s]?://[^\s]+|\d+|[a-zA-Z]+', '', text)
    # 去除标点符号和特殊字符，只保留中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def segment_text(text, mode='default'):
    """分词函数，支持精确模式和搜索引擎模式"""
    return jieba.lcut_for_search(text) if mode == 'search' else jieba.lcut(text)


def filter_stopwords(word_list, stopwords):
    """过滤停用词"""
    return [word for word in word_list if word.strip() and word not in stopwords]


def extract_keywords(text, method='tfidf', topK=100):
    """提取关键词，支持TF-IDF和TextRank方法"""
    if method.lower() == 'textrank':
        return jieba.analyse.textrank(text, topK=topK, withWeight=True)
    else:  # 默认使用TF-IDF
        return jieba.analyse.extract_tags(text, topK=topK, withWeight=True)


def save_results(data, file_path, is_keywords=False):
    """保存结果到文件"""
    if is_keywords:
        # 关键词结果格式：word: weight
        content = '\n'.join(f"{word}: {weight:.6f}" for word, weight in data)
    else:
        # 词频结果格式：word: frequency
        content = '\n'.join(f"{word}: {freq}" for word, freq in data)
    
    if write_file(file_path, content):
        print(f"结果已保存到: {file_path}")
        return True
    return False


def main():
    # 检查并加载输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"输入文件 {INPUT_FILE} 不存在，请先准备文本文件")
        return
    
    # 加载和预处理文本
    print("正在加载和预处理文本...")
    text = read_file(INPUT_FILE)
    if not text:
        print("文本加载失败，请检查文件内容")
        return
    text = preprocess_text(text)
    
    # 加载停用词表
    print("正在加载停用词表...")
    stopwords = load_stopwords()
    
    # 添加自定义词典
    custom_words = ['人工智能', '机器学习', '深度学习', '自然语言处理']
    for word in custom_words:
        jieba.add_word(word)
    print(f"已添加 {len(custom_words)} 个自定义词")
    
    # 1. 中文分词（精确模式）
    print("正在进行中文分词（精确模式）...")
    words_default = segment_text(text)
    print(f"精确模式分词结果（前10个）: {words_default[:10]}")
    
    # 2. 中文分词（搜索引擎模式）
    print("正在进行中文分词（搜索引擎模式）...")
    words_search = segment_text(text, mode='search')
    print(f"搜索引擎模式分词结果（前10个）: {words_search[:10]}")
    
    # 3. 停用词过滤
    print("正在过滤停用词...")
    filtered_words = filter_stopwords(words_default, stopwords)
    print(f"停用词过滤后的结果（前10个）: {filtered_words[:10]}")
    
    # 4. 词频统计
    print("正在统计词频...")
    word_freq = Counter(filtered_words)
    print(f"词频统计结果（前5个）: {dict(word_freq.most_common(5))}")
    
    # 保存词频统计结果
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    save_results(sorted_word_freq, WORD_FREQ_FILE)
    
    # 5. 提取关键词（TF-IDF）
    print("正在使用TF-IDF提取关键词...")
    keywords_tfidf = extract_keywords(text, method='tfidf', topK=100)
    print(f"TF-IDF关键词提取结果（前5个）: {keywords_tfidf[:5]}")
    save_results(keywords_tfidf, TOP_KEYWORDS_TFIDF_FILE, is_keywords=True)
    
    # 6. 提取关键词（TextRank）
    print("正在使用TextRank提取关键词...")
    keywords_textrank = extract_keywords(text, method='textrank', topK=100)
    print(f"TextRank关键词提取结果（前5个）: {keywords_textrank[:5]}")
    save_results(keywords_textrank, TOP_KEYWORDS_TEXTRANK_FILE, is_keywords=True)
    
    print("\n文本预处理完成！")
    print(f"词频统计结果已保存到: {WORD_FREQ_FILE}")
    print(f"TF-IDF关键词提取结果已保存到: {TOP_KEYWORDS_TFIDF_FILE}")
    print(f"TextRank关键词提取结果已保存到: {TOP_KEYWORDS_TEXTRANK_FILE}")


if __name__ == "__main__":
    main()