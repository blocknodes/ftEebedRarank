import json
import os
import sys
import argparse
import ast
from collections import defaultdict

class JSONLReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.offset_index = None  # 存储行号到文件偏移量的映射 {行号: 偏移量}
        self.line_count = 0      # 总行数缓存

    def build_offset_index(self):
        """预构建行号与文件偏移量的索引，首次调用时建立"""
        self.offset_index = {1: 0}  # 第一行偏移量为0
        current_offset = 0
        self.line_count = 0

        with open(self.file_path, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break  # 文件结束
                current_offset += len(line)
                self.line_count += 1
                self.offset_index[self.line_count + 1] = current_offset

    def read_line(self, line_num):
        """读取指定行，利用预建的偏移量索引快速定位"""
        if not self.offset_index:
            # 如果未建索引，先构建（首次读取时自动触发）
            self.build_offset_index()

        if line_num < 1 or line_num > self.line_count:
            print(f"错误：第{line_num}行不存在（文件共{self.line_count}行）")
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

    def get_line_count(self):
        """获取文件总行数"""
        if not self.offset_index:
            self.build_offset_index()
        return self.line_count

def jsonl_to_query_dict(file_path):
    """构建查询到行号的映射"""
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

def save_metrics_to_json(metrics, output_path, topk):
    """将评估指标保存为JSON文件（新增topk字段标注）"""
    # 在指标中添加topk字段，明确当前评估使用的Top-K值
    metrics_with_topk = {**metrics, "eval_topk": topk}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_with_topk, f, ensure_ascii=False, indent=2)

# 评估配置 - 可根据实际文件格式调整
EVAL_CONFIG = {
    'sentence': {
        'file_pattern': 'sentence_recall_result.jsonl',
        'hit_func': 'sentence_hit',
        'type': 'basic'
    },
    'segment': {
        'file_pattern': 'segment_recall_result.jsonl',
        'hit_func': 'segment_hit',
        'type': 'basic'
    },
    'qa': {
        'file_pattern': 'qa_recall_result.jsonl',
        'hit_func': 'qa_hit',
        'type': 'basic'
    },
    'rerank': {
        'file_pattern': 'rerank_result.jsonl',
        'metrics_func': 'rerank_metrics',
        'type': 'rerank',
        'filename_key': 'filename'
    },
    'rerank_qa': {
        'file_pattern': 'rerank_qa_result.jsonl',
        'metrics_func': 'rerank_qa_metrics',
        'type': 'rerank',
        'filename_key': 'filename'
    },
    'final': {
        'file_pattern': 'final_result.jsonl',
        'metrics_func': 'final_metrics',
        'type': 'rerank',
        'filename_key': 'filename'
    }
}

# 通用命中检查函数
def basic_hit(query, filenames, line, config):
    """基础命中检查，使用配置中的filename_key"""
    if not isinstance(filenames, list):
        filenames = [filenames]

    filename_key = config.get('filename_key', 'filename')
    result_filenames = set(item[filename_key] for item in line['value'])
    return 1 if any(fn in result_filenames for fn in filenames) else 0

# 为保持兼容性，定义特定名称的命中函数
def sentence_hit(query, filenames, line):
    return basic_hit(query, filenames, line, EVAL_CONFIG['sentence'])

def segment_hit(query, filenames, line):
    return basic_hit(query, filenames, line, EVAL_CONFIG['segment'])

def qa_hit(query, filenames, line):
    return basic_hit(query, filenames, line, EVAL_CONFIG['qa'])

# 通用重排序评估函数（支持动态Top-K）
def rerank_style_metrics(query, filenames, line, config, topk):
    """重排序结果评估，基于指定Top-K检索结果计算Recall和Precision"""
    relevant_files = set(filenames)
    filename_key = config.get('filename_key', 'filename')

    # 1. 读取全量检索结果，按Top-K截取
    retrieved_files_all = [item[filename_key] for item in line['value']]
    retrieved_files_topk = retrieved_files_all[:topk]  # 动态截取Top-K

    # 2. 基础统计量（基于Top-K）
    total_retrieved_topk = len(retrieved_files_topk)  # ≤ topk
    total_relevant = len(relevant_files)

    # 3. 计算命中的相关文件（基于Top-K）
    hit_files = set(fn for fn in retrieved_files_topk if fn in relevant_files)
    hits = len(hit_files)

    # 4. Hit@1/3/10仍基于全量结果（与原逻辑一致，不依赖Top-K）
    hit1 = 1 if any(fn in relevant_files for fn in retrieved_files_all[:1]) else 0
    hit3 = 1 if any(fn in relevant_files for fn in retrieved_files_all[:3]) else 0
    hit10 = 1 if any(fn in relevant_files for fn in retrieved_files_all[:10]) else 0

    # 5. 基于Top-K计算Recall和Precision
    denominator = min(total_relevant, total_retrieved_topk)
    recall = hits / denominator if denominator > 0 else 0
    precision = hits / total_retrieved_topk if total_retrieved_topk > 0 else 0

    return hit1, hit3, hit10, recall, precision, total_relevant, total_retrieved_topk, hits

# 特定重排序评估函数（新增topk参数传递）
def rerank_metrics(query, filenames, line, topk):
    return rerank_style_metrics(query, filenames, line, EVAL_CONFIG['rerank'], topk)

def rerank_qa_metrics(query, filenames, line, topk):
    return rerank_style_metrics(query, filenames, line, EVAL_CONFIG['rerank_qa'], topk)

def final_metrics(query, filenames, line, topk):
    return rerank_style_metrics(query, filenames, line, EVAL_CONFIG['final'], topk)

def main():
    parser = argparse.ArgumentParser(description='评估各类JSONL结果文件（支持自定义Top-K计算Recall/Precision）')
    parser.add_argument('input', help='输入JSONL文件的路径（含query和相关file_name）')
    parser.add_argument('outputdir', help='输出目录（存储指标文件和命中行号）')
    parser.add_argument('--include', nargs='+', help='指定要评估的文件类型（如sentence rerank）',
                      default=list(EVAL_CONFIG.keys()))
    parser.add_argument('--exclude', nargs='+', help='指定要排除的文件类型', default=[])
    # 新增--topk可选参数：默认3，支持正整数（需≥1）
    parser.add_argument('--topk', type=int, default=1, help='评估Recall/Precision时使用的Top-K值（默认3，需≥1）')
    args = parser.parse_args()

    # 校验Top-K参数合法性（避免无效值）
    if args.topk < 1:
        print(f"错误：--topk值必须≥1，当前输入为{args.topk}")
        sys.exit(1)

    # 创建输出目录（如果不存在）
    os.makedirs(args.outputdir, exist_ok=True)

    # 确定最终需要评估的文件类型
    included_types = [t for t in args.include if t in EVAL_CONFIG and t not in args.exclude]
    if not included_types:
        print("错误：没有有效的评估类型被选中")
        return

    # 初始化文件路径、读取器和query映射
    file_paths = {}
    readers = {}
    query_maps = {}
    for eval_type in included_types:
        config = EVAL_CONFIG[eval_type]
        file_path = f"{args.outputdir}/{config['file_pattern']}"
        file_paths[eval_type] = file_path

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，将跳过该类型评估")
            included_types.remove(eval_type)
            continue

        readers[eval_type] = JSONLReader(file_path)
        query_maps[eval_type] = jsonl_to_query_dict(file_path)
        print(f"已加载 {eval_type} 评估文件，共 {readers[eval_type].get_line_count()} 行")

    # 初始化指标存储结构
    total_query = 0
    metrics = defaultdict(dict)

    # 初始化基础召回指标和重排序指标
    for eval_type in included_types:
        if EVAL_CONFIG[eval_type]['type'] == 'basic':
            metrics[eval_type] = {'hit': 0, 'total': 0}
        else:  # rerank类型
            metrics[eval_type] = {
                'hit1': 0, 'hit3': 0, 'hit10': 0,
                'recall': 0, 'precision': 0,
                'rel_count': 0, 'ret_count': 0, 'hit_count': 0, 'denominator': 0,
                'hit1_lines': [], 'hit3_lines': [], 'hit10_lines': [],
                'total': 0
            }

    # 处理输入文件并计算指标（核心逻辑：传递topk参数）
    with open(args.input, 'r', encoding='utf-8') as file:
        line_number = 0
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

                    # 解析相关文件列表（处理可能的字符串格式列表，如"['a.txt','b.txt']"）
                    try:
                        filenames = ast.literal_eval(list(filenames)[0]) if filenames else []
                    except (SyntaxError, ValueError):
                        filenames = filenames if filenames else []

                    # 检查当前query在哪些评估类型中存在
                    query_in_types = [
                        t for t in included_types
                        if data['query'] in query_maps[t]
                    ]

                    if query_in_types:
                        total_query += 1

                        # 处理基础召回指标（无需Top-K）
                        for eval_type in query_in_types:
                            config = EVAL_CONFIG[eval_type]
                            metrics[eval_type]['total'] += 1

                            if config['type'] == 'basic':
                                line_data = readers[eval_type].read_line(
                                    query_maps[eval_type][data['query']]
                                )
                                hit_func = globals()[config['hit_func']]
                                metrics[eval_type]['hit'] += hit_func(
                                    data['query'], filenames, line_data
                                )
                            else:
                                # 处理重排序指标（传递topk参数到评估函数）
                                line_data = readers[eval_type].read_line(
                                    query_maps[eval_type][data['query']]
                                )
                                metrics_func = globals()[config['metrics_func']]
                                # 调用时传入args.topk，实现动态Top-K计算
                                hit1, hit3, hit10, recall, precision, rel_count, ret_count, hit_count = metrics_func(
                                    data['query'], filenames, line_data, args.topk
                                )

                                # 累计指标
                                metrics[eval_type]['hit1'] += hit1
                                metrics[eval_type]['hit3'] += hit3
                                metrics[eval_type]['hit10'] += hit10
                                metrics[eval_type]['recall'] += recall
                                metrics[eval_type]['precision'] += precision
                                metrics[eval_type]['rel_count'] += rel_count
                                metrics[eval_type]['ret_count'] += ret_count  # 基于Top-K的检索数
                                metrics[eval_type]['hit_count'] += hit_count  # 基于Top-K的命中数
                                metrics[eval_type]['denominator'] += min(rel_count, ret_count)

                                # 记录命中行号
                                if hit1:
                                    metrics[eval_type]['hit1_lines'].append(line_number)
                                if hit3:
                                    metrics[eval_type]['hit3_lines'].append(line_number)
                                if hit10:
                                    metrics[eval_type]['hit10_lines'].append(line_number)

            except json.JSONDecodeError as e:
                print(f"Line {line_number}: JSON解析错误 - {str(e)}")
            except Exception as e:
                print(f"Line {line_number}: 处理错误 - {str(e)}")

    # 保存指标文件和命中行号（指标文件新增topk标注）
    for eval_type in included_types:
        # 保存命中行号文件（重排序类型专属）
        if EVAL_CONFIG[eval_type]['type'] == 'rerank':
            for hit_type in ['hit1', 'hit3', 'hit10']:
                line_file_path = f"{args.outputdir}/{eval_type}_{hit_type}_top{args.topk}_line_numbers.txt"
                with open(line_file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(map(str, metrics[eval_type][f"{hit_type}_lines"])))

        # 保存指标JSON文件（含topk字段）
        metric_file_path = f"{args.outputdir}/{eval_type}_top{args.topk}_metrics.json"
        save_metrics_to_json(metrics[eval_type], metric_file_path, args.topk)

    # 打印综合评估报告（明确标注当前Top-K值）
    print("\n" + "="*80)
    print(f"综合评估报告（Recall/Precision基于Top-{args.topk}计算）- 总查询数: {total_query}")
    print("="*80)

    # 打印基础召回结果
    print("\n【基础召回结果】")
    print("-"*60)
    for eval_type in included_types:
        if EVAL_CONFIG[eval_type]['type'] == 'basic':
            cfg = metrics[eval_type]
            total = cfg['total']
            hit_rate = cfg['hit'] / total if total > 0 else 0.0

            print(f"{eval_type}:")
            print(f"  有效查询数: {total}")
            print(f"  命中数: {cfg['hit']}")
            print(f"  命中率: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
            print(f"  指标文件: {args.outputdir}/{eval_type}_top{args.topk}_metrics.json")
            print("-"*60)

    # 打印重排序结果（标注Top-K值）
    print("\n【重排序结果（Recall/Precision基于Top-{args.topk}）】")
    print("-"*60)
    for eval_type in included_types:
        if EVAL_CONFIG[eval_type]['type'] == 'rerank':
            cfg = metrics[eval_type]
            total = cfg['total']
            if total == 0:
                hit1_rate = hit3_rate = hit10_rate = 0.0
                avg_recall = avg_precision = 0.0
            else:
                hit1_rate = cfg['hit1'] / total
                hit3_rate = cfg['hit3'] / total
                hit10_rate = cfg['hit10'] / total
                avg_recall = cfg['recall'] / total  # 基于Top-K的平均召回率
                avg_precision = cfg['precision'] / total  # 基于Top-K的平均精确率

            # 总体指标（基于Top-K）
            overall_recall = cfg['hit_count'] / cfg['denominator'] if cfg['denominator'] > 0 else 0
            overall_precision = cfg['hit_count'] / cfg['ret_count'] if cfg['ret_count'] > 0 else 0

            print(f"{eval_type}:")
            print(f"  有效查询数: {total}")
            print(f"  总相关文件数: {cfg['rel_count']}")
            print(f"  总Top-{args.topk}检索文件数: {cfg['ret_count']}")
            print(f"  总Top-{args.topk}命中文件数: {cfg['hit_count']}")
            print(f"  Hit@1: {cfg['hit1']} ({hit1_rate:.2%})")
            print(f"  Hit@3: {cfg['hit3']} ({hit3_rate:.2%})")
            print(f"  Hit@10: {cfg['hit10']} ({hit10_rate:.2%})")
            print(f"  平均召回率（Top-{args.topk}）: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
            print(f"  平均精确率（Top-{args.topk}）: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
            print(f"  总体召回率（Top-{args.topk}）: {overall_recall:.4f} ({overall_recall*100:.2f}%)")
            print(f"  总体精确率（Top-{args.topk}）: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
            print(f"  指标文件: {args.outputdir}/{eval_type}_top{args.topk}_metrics.json")
            print(f"  命中行号文件: {args.outputdir}/{eval_type}_hit*_top{args.topk}_line_numbers.txt")
            print("-"*60)

    print("\n" + "="*80)
    print("评估完成！")

if __name__ == "__main__":
    main()