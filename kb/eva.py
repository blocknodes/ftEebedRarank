import json
import os
import sys
import argparse

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

def sentence_hit(query, filename, line):
    filenames = set(item['file_name'] for item in line['value'])
    return 1 if filename in filenames else 0

def segment_hit(query, filename, line):
    filenames = set(item['file_name'] for item in line['value'])
    return 1 if filename in filenames else 0

def rerank_hit(query, filename, line):
    hit1_filenames = set(item['filename'] for item in line['value'][:1])
    hit3_filenames = set(item['filename'] for item in line['value'][:3])
    hit1 = 1 if filename in hit1_filenames else 0
    hit3 = 1 if filename in hit3_filenames else 0
    return hit1, hit3

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='eval hit@1/3 in recall/rerank')
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

        # 存储hit1和hit3的行号
        hit1_line_numbers = []
        hit3_line_numbers = []

        for line in file:
            line_number += 1
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                data = json.loads(line)

                if 'query' in data:
                    if data['query'] in sentence_query_map and \
                       data['query'] in segment_query_map and \
                       data['query'] in rerank_query_map:

                        total_query += 1
                        total_sentenct_hit += sentence_hit(
                            data['query'],
                            data['filename'],
                            sentence_reader.read_line(sentence_query_map[data['query']])
                        )
                        total_segment_hit += segment_hit(
                            data['query'],
                            data['filename'],
                            segment_reader.read_line(segment_query_map[data['query']])
                        )
                        hit1, hit3 = rerank_hit(
                            data['query'],
                            data['filename'],
                            rerank_reader.read_line(rerank_query_map[data['query']])
                        )
                        total_rerank_hit1 += hit1
                        total_rerank_hit3 += hit3

                        # 记录命中的行号
                        if hit1:
                            hit1_line_numbers.append(line_number)
                        if hit3:
                            hit3_line_numbers.append(line_number)
                # 移除了"未找到query字段"的打印，只在有错误时输出

            except json.JSONDecodeError as e:
                print(f"Line {line_number}: JSON解析错误 - {str(e)}")

    # 计算平均值（命中率）
    sentence_hit_rate = total_sentenct_hit / total_query if total_query > 0 else 0
    segment_hit_rate = total_segment_hit / total_query if total_query > 0 else 0
    rerank_hit1_rate = total_rerank_hit1 / total_query if total_query > 0 else 0
    rerank_hit3_rate = total_rerank_hit3 / total_query if total_query > 0 else 0

    # 保存hit1和hit3的行号到文件
    with open(f"{args.outputdir}/hit1_line_numbers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, hit1_line_numbers)))

    with open(f"{args.outputdir}/hit3_line_numbers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, hit3_line_numbers)))

    # 打印结果
    print("="*50)
    print("评估结果统计")
    print("="*50)
    print(f"总查询数: {total_query}")
    print("\n句子召回结果:")
    print(f"  命中数: {total_sentenct_hit}")
    print(f"  命中率: {sentence_hit_rate:.4f} ({sentence_hit_rate*100:.2f}%)")
    print("\n段落召回结果:")
    print(f"  命中数: {total_segment_hit}")
    print(f"  命中率: {segment_hit_rate:.4f} ({segment_hit_rate*100:.2f}%)")
    print("\n重排序结果:")
    print(f"  Hit@1 命中数: {total_rerank_hit1}")
    print(f"  Hit@1 命中率: {rerank_hit1_rate:.4f} ({rerank_hit1_rate*100:.2f}%)")
    print(f"  Hit@1 行号已保存至: {args.outputdir}/hit1_line_numbers.txt")
    print(f"  Hit@3 命中数: {total_rerank_hit3}")
    print(f"  Hit@3 命中率: {rerank_hit3_rate:.4f} ({rerank_hit3_rate*100:.2f}%)")
    print(f"  Hit@3 行号已保存至: {args.outputdir}/hit3_line_numbers.txt")
    print("="*50)
