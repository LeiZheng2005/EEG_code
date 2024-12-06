import pandas as pd

# 读取Excel文件
file_path = '/Users/leizheng/PyCharm_Study_Code/EEG_code/Other_Functions/test_code_excel.xlsx'  # 请替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 假设第一列是'Name'，第二列是'List'
# 请根据实际情况替换列名
first_column = df.iloc[:, 0]  # 获取第一列（假设是名字列）
second_column = df.iloc[:, 1]  # 获取第二列（假设是列表列）

# 将第二列的内容转化为一个集合，方便查找
second_column_set = set(second_column)

# 判断第一列的每个名字是否出现在第二列中
df['In_List'] = first_column.isin(second_column_set)

# 输出结果到一个新的Excel文件
output_file = 'output.xlsx'  # 输出文件路径
df.to_excel(output_file, index=False)

print(f'处理完成，结果已保存到 {output_file}')
