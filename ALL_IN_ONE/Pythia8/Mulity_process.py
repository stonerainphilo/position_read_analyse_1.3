import pandas as pd

def split_csv_into_three_parts(input_csv):
    # 读取 CSV 文件
    df = pd.read_csv(input_csv)
    
    # 计算每部分的大小
    total_rows = len(df)
    part_size = total_rows // 3
    
    # 分割数据
    part1 = df.iloc[:part_size]
    part2 = df.iloc[part_size:2 * part_size]
    part3 = df.iloc[2 * part_size:]
    
    # 保存为 1.csv, 2.csv, 3.csv
    part1.to_csv("1.csv", index=False)
    part2.to_csv("2.csv", index=False)
    part3.to_csv("3.csv", index=False)

    print("CSV file has been split into 3 parts: 1.csv, 2.csv, 3.csv")

# 示例调用
# split_csv_into_three_parts("your_file.csv")