import json
import os
import sys
import argparse
import ast

class JSONLReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.offset_index = None  # 存储行号到文件偏移量的映射 {行号: 偏移量}

    def build_offset_index(self):
        """预构建行号与文件偏移量的索引，首次调用时建立"""
        self.offset_index = {1: 0}  # 第一行偏移量为0
        current_offset = 0

        with open(self.file_path, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break  # 文件结束
                current_offset += len(line)
                self.offset_index[len(self.offset_index) + 1] = current_offset

    def read_line(self, line_num):
        """读取指定行，利用预建的偏移量索引快速定位"""
        if not self.offset_index:
            # 如果未建索引，先构建（首次读取时自动触发）
            self.build_offset_index()

        if line_num not in self.offset_index:
            print(f"错误：第{line_num}行不存在")
            return None

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # 直接移动到目标行的偏移量位置
                f.seek(self.offset_index[line_num])
                line = f.readline()
                if not line:
                    print(f"错误：第{line_num}行读取失败")
                    return None
                return json.loads(line.strip())
        except json.JSONDecodeError:
            print(f"错误：第{line_num}行不是有效的JSON")
            return None
        except Exception as e:
            print(f"读取错误：{str(e)}")
            return None

# 结合原有功能的使用示例
def jsonl_to_query_dict(file_path):
    query_dict = {}
    reader = JSONLReader(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'query' in data:
                    query_dict[data['query']] = line_num
            except Exception as e:
                print(f"第{line_num}行处理失败：{e}")

    return query_dict

def sentence_hit(query, filenames, line):
    if not isinstance(filenames, list):
        filenames = [filenames]

    result_filenames = set(item['file_name'] for item in line['value'])
    return 1 if any(fn in result_filenames for fn in filenames) else 0

def segment_hit(query, filenames, line):
    if not isinstance(filenames, list):
        filenames = [filenames]

    result_filenames = set(item['file_name'] for item in line['value'])
    return 1 if any(fn in result_filenames for fn in filenames) else 0

def rerank_metrics(query, filenames, line):
    """计算rerank结果的命中情况、召回率和精确率
    召回率计算使用：实际召回数目 / min(应召回数目, 检索到的文件总数)
    """
    if not isinstance(filenames, list):
        filenames = [filenames]

    # 提取相关文件集合和检索结果集合
    filenames = ast.literal_eval(list(filenames)[0])
    relevant_files = set(filenames)
    retrieved_files = [item['filename'] for item in line['value']]
    total_retrieved = len(retrieved_files)
    total_relevant = len(relevant_files)

    # 计算命中的相关文件
    hit_files = [fn for fn in retrieved_files if fn in relevant_files]
    hits = len(hit_files)

    # 计算Hit@1和Hit@3
    hit1 = 1 if any(fn in relevant_files for fn in retrieved_files[:1]) else 0
    hit3 = 1 if any(fn in relevant_files for fn in retrieved_files[:3]) else 0

    # 计算召回率：使用实际召回数和应召回数的最小值作为分母
    # 避免当检索结果数量少于相关文件数量时，召回率被低估
    denominator = min(total_relevant, total_retrieved)
    recall = hits / denominator if denominator > 0 else 0

    # 计算精确率
    precision = hits / total_retrieved if total_retrieved > 0 else 0

    return hit1, hit3, recall, precision, total_relevant, total_retrieved, hits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval hit@1/3, recall and precision in rerank')
    parser.add_argument('input', help='JSONL文件的路径')
    parser.add_argument('outputdir', help='output目录')
    args = parser.parse_args()

    sentence_file = f'{args.outputdir}/sentence_recall_result.jsonl'
    segment_file = f'{args.outputdir}/segment_recall_result.jsonl'
    rerank_file = f'{args.outputdir}/rerank_result.jsonl'

    sentence_reader = JSONLReader(sentence_file)
    segment_reader = JSONLReader(segment_file)
    rerank_reader = JSONLReader(rerank_file)

    sentence_query_map = jsonl_to_query_dict(sentence_file)
    segment_query_map = jsonl_to_query_dict(segment_file)
    rerank_query_map = jsonl_to_query_dict(rerank_file)


    with open(args.input, 'r', encoding='utf-8') as file:
        line_number = 0

        total_query = 0
        total_sentenct_hit = 0
        total_segment_hit = 0
        total_rerank_hit1 = 0
        total_rerank_hit3 = 0

        # 用于计算rerank召回率和精确率的累计变量
        total_rerank_recall = 0
        total_rerank_precision = 0
        total_relevant_files = 0
        total_retrieved_files = 0
        total_hit_files = 0
        total_denominator = 0  # 新增：累计召回率计算中的分母（最小值）

        hit1_line_numbers = []
        hit3_line_numbers = []

        for line in file:
            line_number += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                if 'query' in data and 'file_name' in data:
                    filenames = data['file_name']
                    if not isinstance(filenames, list):
                        filenames = [filenames]

                    if data['query'] in sentence_query_map and \
                       data['query'] in segment_query_map and \
                       data['query'] in rerank_query_map:

                        total_query += 1
                        total_sentenct_hit += sentence_hit(
                            data['query'],
                            filenames,
                            sentence_reader.read_line(sentence_query_map[data['query']])
                        )
                        total_segment_hit += segment_hit(
                            data['query'],
                            filenames,
                            segment_reader.read_line(segment_query_map[data['query']])
                        )

                        # 调用修改后的函数获取更多指标
                        hit1, hit3, recall, precision, rel_count, ret_count, hit_count = rerank_metrics(
                            data['query'],
                            filenames,
                            rerank_reader.read_line(rerank_query_map[data['query']])
                        )

                        # 累计rerank指标
                        total_rerank_hit1 += hit1
                        total_rerank_hit3 += hit3
                        total_rerank_recall += recall
                        total_rerank_precision += precision
                        total_relevant_files += rel_count
                        total_retrieved_files += ret_count
                        total_hit_files += hit_count
                        total_denominator += min(rel_count, ret_count)  # 累计最小值分母

                        if hit1:
                            hit1_line_numbers.append(line_number)
                        if hit3:
                            hit3_line_numbers.append(line_number)

            except json.JSONDecodeError as e:
                print(f"Line {line_number}: JSON解析错误 - {str(e)}")
            except Exception as e:
                print(f"Line {line_number}: 处理错误 - {str(e)}")

    # 计算平均值
    sentence_hit_rate = total_sentenct_hit / total_query if total_query > 0 else 0
    segment_hit_rate = total_segment_hit / total_query if total_query > 0 else 0
    rerank_hit1_rate = total_rerank_hit1 / total_query if total_query > 0 else 0
    rerank_hit3_rate = total_rerank_hit3 / total_query if total_query > 0 else 0

    # 计算总体recall和precision（两种方式）
    # 1. 平均每个查询的recall和precision
    avg_rerank_recall = total_rerank_recall / total_query if total_query > 0 else 0
    avg_rerank_precision = total_rerank_precision / total_query if total_query > 0 else 0

    # 2. 总体recall和precision（所有查询合并计算）
    # 总体召回率使用总命中数 / 总最小值分母
    overall_recall = total_hit_files / total_denominator if total_denominator > 0 else 0
    overall_precision = total_hit_files / total_retrieved_files if total_retrieved_files > 0 else 0

    # 保存行号文件
    with open(f"{args.outputdir}/hit1_line_numbers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, hit1_line_numbers)))

    with open(f"{args.outputdir}/hit3_line_numbers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, hit3_line_numbers)))

    # 打印结果
    print("="*70)
    print("评估结果统计")
    print("="*70)
    print(f"总查询数: {total_query}")
    print(f"总相关文件数: {total_relevant_files}")
    print(f"总检索文件数: {total_retrieved_files}")
    print(f"总命中文件数: {total_hit_files}")
    print(f"总召回率计算分母(最小值): {total_denominator}")

    print("\n句子召回结果:")
    print(f"  命中数: {total_sentenct_hit}")
    print(f"  命中率: {sentence_hit_rate:.4f} ({sentence_hit_rate*100:.2f}%)")

    print("\n段落召回结果:")
    print(f"  命中数: {total_segment_hit}")
    print(f"  命中率: {segment_hit_rate:.4f} ({segment_hit_rate*100:.2f}%)")

    print("\n重排序结果:")
    print(f"  Hit@1 命中数: {total_rerank_hit1}")
    print(f"  Hit@1 命中率: {rerank_hit1_rate:.4f} ({rerank_hit1_rate*100:.2f}%)")
    print(f"  Hit@3 命中数: {total_rerank_hit3}")
    print(f"  Hit@3 命中率: {rerank_hit3_rate:.4f} ({rerank_hit3_rate*100:.2f}%)")
    print(f"  平均召回率 (按查询): {avg_rerank_recall:.4f} ({avg_rerank_recall*100:.2f}%)")
    print(f"  平均精确率 (按查询): {avg_rerank_precision:.4f} ({avg_rerank_precision*100:.2f}%)")
    print(f"  总体召回率 (合并计算，使用最小值分母): {overall_recall:.4f} ({overall_recall*100:.2f}%)")
    print(f"  总体精确率 (合并计算): {overall_precision:.4f} ({overall_precision*100:.2f}%)")

    print(f"\nHit@1 行号已保存至: {args.outputdir}/hit1_line_numbers.txt")
    print(f"Hit@3 行号已保存至: {args.outputdir}/hit3_line_numbers.txt")
    print("="*70)
