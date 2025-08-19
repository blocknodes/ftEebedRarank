import json
def text_similarity(text1, text2):
    """
    计算两段文本的余弦相似度（用于判断近似程度）
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # 创建TF-IDF向量器
    vectorizer = TfidfVectorizer()

    # 将文本转换为TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # 计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity

def longest_overlapping_substring(s1, s2):
    """找出两个字符串之间的最长重叠子字符串"""
    max_len = 0
    longest_sub = ""
    
    # 确保s1是较短的字符串，优化搜索效率
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    # 检查所有可能长度的子字符串
    for length in range(len(s1), 0, -1):
        # 检查s1中所有该长度的子字符串
        for i in range(len(s1) - length + 1):
            sub = s1[i:i+length]
            if sub in s2:
                return sub  # 找到最长的就返回
    return ""  # 没有重叠子字符串

def string_similarity(s1, s2):
    """计算两个字符串的相似度，以最长重叠子字符串占比作为指标"""
    if not s1 or not s2:
        return 0.0  # 空字符串相似度为0
    
    # 找到最长重叠子字符串
    longest_sub = longest_overlapping_substring(s1, s2)
    overlap_length = len(longest_sub)
    
    if overlap_length == 0:
        return 0.0  # 没有重叠部分
    
    # 计算平均长度作为基准
    avg_length = (len(s1) + len(s2)) / 2
    
    # 计算相似度（最长重叠子串长度 / 平均长度）
    similarity = overlap_length / avg_length
    
    return similarity

def process_jsonl_file(file_path, output_file_path=None):
    """
    读取JSONL文件，处理每行数据：
    1. 从recall列表中移除所有在pos列表中出现的字符串
    2. 将'pos'字段重命名为'positive'
    3. 将'recall'字段重命名为'negative'

    参数:
        file_path (str): 输入JSONL文件的路径
        output_file_path (str, optional): 输出处理后的JSONL文件路径，默认为原文件加后缀"_processed"
    """
    # 设置默认输出文件路径
    if output_file_path is None:
        # 在原文件名后添加"_processed"后缀
        name_parts = file_path.split('.')
        if len(name_parts) > 1:
            output_file_path = '.'.join(name_parts[:-1]) + '_processed.' + name_parts[-1]
        else:
            output_file_path = file_path + '_processed'

    try:
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            line_number = 0
            for line in infile:
                line_number += 1
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                try:
                    # 解析JSON对象
                    data = json.loads(line)

                    # 检查数据结构是否符合预期
                    if not isinstance(data, dict) or 'query' not in data or 'pos' not in data or 'recall' not in data:
                        print(f"第 {line_number} 行: 数据结构不符合预期，跳过处理")
                        continue

                    # 确保pos和recall是列表
                    if not isinstance(data['pos'], list) or not isinstance(data['recall'], list) :
                        print(f"第 {line_number} 行: pos或recall不是列表，跳过处理")
                        continue

                    # 将pos转换为集合，提高查找效率
                    pos_set = set(data['pos'])

                    # 过滤recall列表，移除存在于pos中的元素
                    filtered_recall = []

                    for item in data['recall']:
                        same = False
                        for pos_item in pos_set:
                            if item[:-1] ==  pos_item or text_similarity(item,pos_item) >0.95 or string_similarity(item,pos_item)>0.95:
                                same = True
                                break
                        if same:
                            continue
                        filtered_recall.append(item)

                    filtered_recall = [item for item in data['recall'] if item[:-1] not in pos_set]

                    # 创建新的字典，实现字段重命名
                    processed_data = {
                        'query': data['query'],
                        'positive': data['pos'],  # 重命名pos为positive
                        'negative': filtered_recall  # 重命名recall为negative，并使用过滤后的值
                    }

                    # 输出处理后的行到新文件
                    json.dump(processed_data, outfile, ensure_ascii=False)
                    outfile.write('\n')

                    # 打印处理信息
                    removed_count = len(data['recall']) - len(filtered_recall)
                    if removed_count > 0:
                        print(f"第 {line_number} 行: 移除了 {removed_count} 个元素，并完成字段重命名")
                    else:
                        print(f"第 {line_number} 行: 未发现需要移除的元素，已完成字段重命名")

                except json.JSONDecodeError as e:
                    print(f"第 {line_number} 行解析错误: {str(e)}，已跳过")
                except Exception as e:
                    print(f"第 {line_number} 行处理错误: {str(e)}，已跳过")

        print(f"处理完成，结果已保存到: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
    except IOError as e:
        print(f"文件读写错误: {str(e)}")
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")

if __name__ == "__main__":
    # 替换为你的JSONL文件路径
    input_jsonl_path = "bge_m3_merge_0616.jsonl"
    # 可选：指定输出文件路径，不指定则使用默认路径
    # output_jsonl_path = "processed_data.jsonl"

    # 调用处理函数
    process_jsonl_file(input_jsonl_path)
    # 如果指定输出路径，使用下面的调用方式
    # process_jsonl_file(input_jsonl_path, output_jsonl_path)

