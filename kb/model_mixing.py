from LM_Cocktail import mix_models, mix_models_with_data
import sys

# 检查命令行参数数量是否正确
if len(sys.argv) != 6:
    print("使用方法: python mix_llms.py <模型1路径> <模型2路径> <权重1> <权重2> <输出路径>")
    print("示例: python mix_llms.py ./model1 ./model2 0.6 0.4 ./mixed_model")
    sys.exit(1)

# 解析命令行参数
model1_path = sys.argv[1]
model2_path = sys.argv[2]

# 将权重从字符串转换为浮点数
try:
    weight1 = float(sys.argv[3])
    weight2 = float(sys.argv[4])
except ValueError:
    print("错误: 权重必须是数字")
    sys.exit(1)

output_path = sys.argv[5]

# 混合模型
model = mix_models(
    model_names_or_paths=[model1_path, model2_path],
    model_type='decoder',
    weights=[weight1, weight2],
    output_path=output_path
)

print(f"模型已成功混合并保存到: {output_path}")
