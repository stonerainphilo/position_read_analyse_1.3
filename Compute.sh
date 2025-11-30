#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 1

CSV_FILE="test.csv"
CPP_PROGRAM="./main131_B_2HDM"

if [ ! -f "$CSV_FILE" ]; then
    echo "error: CSV file '$CSV_FILE' does not exist."
    exit 1
fi

if [ ! -x "$CPP_PROGRAM" ]; then
    echo "error: C++ program '$CPP_PROGRAM' does not exist or is not executable."
    exit 1
fi

echo "Processing csv: $CSV_FILE"
line_number=0

while IFS=, read -r mass ctau br seed Output_dir Br_Hee Br_HKK Br_HPiPi Br_Htautau Br_HGluon Br_Hmumu Br_Hgaga Br_H4Pi Br_Hss Br_Hcc theta Decay_Width
do
    # 跳过空行和标题行
    if [[ -z "$mass" || "$mass" == "mass" || "$mass" == *"#"* ]]; then
        continue
    fi

    # 去除可能的空格
    mass=$(echo "$mass" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    ctau=$(echo "$ctau" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    br=1
    seed=1
    Output_dir=$(echo "$Output_dir" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_Hee=$(echo "$Br_Hee" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_HKK=$(echo "$Br_HKK" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_HPiPi=$(echo "$Br_HPiPi" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_Htautau=$(echo "$Br_Htautau" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_HGluon=$(echo "$Br_HGluon" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_Hmumu=$(echo "$Br_Hmumu" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_Hgaga=$(echo "$Br_Hgaga" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_H4Pi=$(echo "$Br_H4Pi" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_Hss=$(echo "$Br_Hss" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Br_Hcc=$(echo "$Br_Hcc" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    theta=$(echo "$theta" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    Decay_Width=$(echo "$Decay_Width" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    line_number=$((line_number + 1))

    # 检查必要参数是否为空
    if [ -z "$mass" ] || [ -z "$ctau" ] || [ -z "$Output_dir" ]; then
        # echo "warning: line $line_number has empty necessary parameters. Skipping..."
        continue
    fi

    # 设置默认值（如果某些参数为空）
    br=${br:-1}
    seed=${seed:-1}
    Br_Hee=${Br_Hee:-0}
    Br_HKK=${Br_HKK:-0}
    Br_HPiPi=${Br_HPiPi:-0}
    Br_Htautau=${Br_Htautau:-0}
    Br_HGluon=${Br_HGluon:-0}
    Br_Hmumu=${Br_Hmumu:-0}
    Br_Hgaga=${Br_Hgaga:-0}
    Br_H4Pi=${Br_H4Pi:-0}
    Br_Hss=${Br_Hss:-0}
    Br_Hcc=${Br_Hcc:-0}
    theta=${theta:-0}
    Decay_Width=${Decay_Width:-0}

    # 创建输出目录
    if [ ! -d "$Output_dir" ]; then
        mkdir -p "$Output_dir"
        if [ $? -ne 0 ]; then
            # echo "Error: Cannot create directory '$Output_dir'"
            continue
        fi
    fi

    # 执行C++程序（按照你的参数顺序）
    # echo "Executing line $line_number: mass=$mass, ctau=$ctau"
    # echo "Command: $CPP_PROGRAM $mass $ctau $br $seed $Output_dir $Br_Hee $Br_HKK $Br_HPiPi $Br_Htautau $Br_HGluon $Br_Hmumu $Br_Hgaga $Br_H4Pi $Br_Hss $Br_Hcc $theta $Decay_Width"
    
    # 实际执行程序
    "$CPP_PROGRAM" "$mass" "$ctau" "$br" "$seed" "$Output_dir" "$Br_Hee" "$Br_HKK" "$Br_HPiPi" "$Br_Htautau" "$Br_HGluon" "$Br_Hmumu" "$Br_Hgaga" "$Br_H4Pi" "$Br_Hss" "$Br_Hcc" "$theta" "$Decay_Width" > /dev/null 2>&1

    # 检查执行结果
    # if [ $? -eq 0 ]; then
    #     # echo "✓ Line $line_number successfully processed"
    # else
    #     # echo "✗ Line $line_number processing failed"
    # fi

    # echo "----------------------------------------"

done < "$CSV_FILE"

# echo "Done! Processed $line_number lines."