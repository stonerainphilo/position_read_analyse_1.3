import pandas as pd


# split_csv_into_three_parts("your_file.csv")


def split_csv_into_different_parts(input_csv, parts):
    df = pd.read_csv(input_csv)
    
    total_rows = len(df)
    part_size = total_rows // parts
    remainder = total_rows % parts 

    start = 0
    for i in range(parts):

        end = start + part_size + (1 if i < remainder else 0) 
        part = df.iloc[start:end]
        part.to_csv(f"{i + 1}.csv", index=False)
        start = end 

    print(f"CSV file has been split into {parts} parts.")

# split_csv_into_different_parts("your_file.csv", 5)