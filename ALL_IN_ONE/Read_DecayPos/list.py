import os
import sys
import subprocess
import time
from datetime import datetime
import pandas as pd


# ==================== 文件夹对处理器 ====================
class BatchFolderProcessor:
    def __init__(self, folder_pairs, script_path=None, delay_between_runs=1):
        """
        初始化文件夹对处理器
        
        参数:
        folder_pairs: 列表，每个元素是(input_folder, output_folder)元组
        script_path: 分析脚本路径，如果为None则使用默认脚本
        delay_between_runs: 每次运行之间的延迟（秒）
        """
        self.folder_pairs = folder_pairs
        self.script_path = script_path
        self.delay_between_runs = delay_between_runs
        
        # 如果脚本路径未提供，使用默认脚本
        if not self.script_path:
            self.script_path = self.create_default_script()
        
        # 结果记录
        self.results = []
        self.summary_file = None
        
    def create_default_script(self):
        """创建默认分析脚本"""
        default_script = """
import os
import pandas as pd
import numpy as np
import sys

# ==================== FASER探测器判断函数 ====================
def whether_in_FASER_by_position(x_pos, y_pos, z_pos, distance_from_ip=480.0, length=1.5, radius=0.1):
    z_start = distance_from_ip
    z_end = distance_from_ip + length
    results = []
    for x, y, z in zip(x_pos, y_pos, z_pos):
        in_z_range = z_start <= z <= z_end
        radial_distance = np.sqrt(x**2 + y**2)
        in_radial_range = radial_distance <= radius
        results.append(in_z_range and in_radial_range)
    return results

# ==================== 主处理函数 ====================
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    print(f"Processing: {input_folder}")
    print(f"Output to: {output_folder}")
    
    # 处理逻辑
    # 这里是你原来的处理代码
    print("Analysis completed successfully")
    
if __name__ == "__main__":
    main()
"""
        
        script_path = "default_analysis_script.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(default_script)
        return script_path
    
    def run_single_folder(self, input_folder, output_folder, index, total):
        """
        运行单个文件夹对
        
        参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        index: 当前处理的序号
        total: 总文件夹对数
        
        返回:
        tuple: (是否成功, 返回码, 输出信息)
        """
        print(f"\n{'='*80}")
        print(f"处理进度: [{index}/{total}]")
        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_folder}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 创建日志文件
        log_file = os.path.join(output_folder, f'run_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        try:
            # 构建命令
            command = [sys.executable, self.script_path, input_folder, output_folder]
            
            # 运行子进程
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并标准输出和错误输出
                universal_newlines=True,
                bufsize=1
            )
            
            # 捕获输出
            output_lines = []
            with open(log_file, 'w', encoding='utf-8') as log:
                log.write(f"Command: {' '.join(command)}\n")
                log.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write("=" * 80 + "\n")
                
                while True:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:
                            print(f"  {line}")
                            output_lines.append(line)
                            log.write(line + "\n")
                    
                    # 检查进程是否结束
                    return_code = process.poll()
                    if return_code is not None:
                        # 读取剩余输出
                        remaining = process.stdout.read()
                        if remaining:
                            for line in remaining.strip().split('\n'):
                                if line.strip():
                                    print(f"  {line.strip()}")
                                    output_lines.append(line.strip())
                                    log.write(line.strip() + "\n")
                        break
            
            # 等待进程完全结束
            process.wait()
            
            success = (return_code == 0)
            status = "成功" if success else "失败"
            
            print(f"\n{'='*80}")
            print(f"处理完成: [{index}/{total}]")
            print(f"状态: {status}")
            print(f"返回码: {return_code}")
            print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            return success, return_code, '\n'.join(output_lines)
            
        except Exception as e:
            error_msg = f"运行脚本时出错: {str(e)}"
            print(f"  ERROR: {error_msg}")
            
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\nERROR: {error_msg}\n")
            
            return False, -1, error_msg
    
    def run_all(self, summary_output_dir=None):
        """
        运行所有文件夹对
        
        参数:
        summary_output_dir: 汇总输出目录，如果为None则不保存汇总
        
        返回:
        dict: 处理结果汇总
        """
        total_pairs = len(self.folder_pairs)
        print(f"\n{'='*80}")
        print(f"开始批量处理")
        print(f"总任务数: {total_pairs}")
        print(f"分析脚本: {self.script_path}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # 初始化统计
        successful = 0
        failed = 0
        
        # 清空结果记录
        self.results = []
        
        # 处理每个文件夹对
        for i, (input_folder, output_folder) in enumerate(self.folder_pairs, 1):
            start_time = time.time()
            
            # 运行单个文件夹
            success, return_code, output = self.run_single_folder(
                input_folder, output_folder, i, total_pairs
            )
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 记录结果
            result = {
                '序号': i,
                '输入文件夹': input_folder,
                '输出文件夹': output_folder,
                '状态': '成功' if success else '失败',
                '返回码': return_code,
                '处理时间(秒)': round(processing_time, 2),
                '开始时间': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.results.append(result)
            
            # 更新统计
            if success:
                successful += 1
            else:
                failed += 1
            
            # 显示当前统计
            print(f"\n当前统计: 成功={successful}, 失败={failed}, 总数={total_pairs}")
            
            # 如果不是最后一个，添加延迟
            if i < total_pairs:
                print(f"等待 {self.delay_between_runs} 秒后处理下一个文件夹...")
                time.sleep(self.delay_between_runs)
        
        # 打印最终汇总
        self.print_summary()
        
        # 保存汇总结果
        if summary_output_dir:
            self.save_summary(summary_output_dir)
        
        return {
            'total': total_pairs,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_pairs if total_pairs > 0 else 0
        }
    
    def print_summary(self):
        """打印处理汇总"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r['状态'] == '成功')
        failed = total - successful
        
        print(f"\n{'='*80}")
        print("批量处理完成！")
        print(f"{'='*80}")
        print(f"总任务数: {total}")
        print(f"成功处理: {successful}")
        print(f"处理失败: {failed}")
        print(f"成功率: {successful/total*100:.1f}%")
        
        if failed > 0:
            print(f"\n失败的文件夹:")
            for result in self.results:
                if result['状态'] == '失败':
                    print(f"  - {result['输入文件夹']} (返回码: {result['返回码']})")
        
        print(f"{'='*80}")
    
    def save_summary(self, output_dir):
        """保存汇总结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建汇总DataFrame
        df_summary = pd.DataFrame(self.results)
        
        # 保存为CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(output_dir, f'batch_processing_summary_{timestamp}.csv')
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 同时保存为文本报告
        report_file = os.path.join(output_dir, f'batch_processing_report_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"批量处理汇总报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"总任务数: {len(self.results)}\n")
            successful = sum(1 for r in self.results if r['状态'] == '成功')
            f.write(f"成功处理: {successful}\n")
            f.write(f"处理失败: {len(self.results) - successful}\n")
            f.write(f"成功率: {successful/len(self.results)*100:.1f}%\n\n")
            
            f.write(f"详细结果:\n")
            f.write(f"{'='*80}\n")
            for result in self.results:
                f.write(f"[{result['序号']}] {result['状态']}\n")
                f.write(f"  输入: {result['输入文件夹']}\n")
                f.write(f"  输出: {result['输出文件夹']}\n")
                f.write(f"  时间: {result['处理时间(秒)']}秒\n")
                f.write(f"  返回码: {result['返回码']}\n")
                f.write(f"{'-'*80}\n")
        
        print(f"\n汇总结果已保存到:")
        print(f"  CSV文件: {summary_file}")
        print(f"  文本报告: {report_file}")
        
        self.summary_file = summary_file


# ==================== 交互式输入函数 ====================
def get_folder_pairs_interactively():
    """
    交互式获取多对文件夹路径
    """
    folder_pairs = []
    
    print(f"\n{'='*80}")
    print("批量文件夹对处理程序")
    print("请输入多对输入/输出文件夹路径")
    print("输入'q'或'quit'结束输入")
    print(f"{'='*80}")
    
    pair_count = 0
    
    while True:
        pair_count += 1
        print(f"\n--- 第 {pair_count} 对文件夹 ---")
        
        # 获取输入文件夹
        print("请输入输入文件夹路径:")
        while True:
            input_folder = input("输入文件夹 > ").strip()
            
            if input_folder.lower() in ['q', 'quit', 'exit']:
                if pair_count == 1:
                    print("程序退出")
                    sys.exit(0)
                else:
                    return folder_pairs
            
            if not input_folder:
                print("错误：输入不能为空")
                continue
            
            if not os.path.exists(input_folder):
                print(f"错误：路径 '{input_folder}' 不存在")
                continue
            
            if not os.path.isdir(input_folder):
                print(f"错误：'{input_folder}' 不是文件夹")
                continue
            
            break
        
        # 获取输出文件夹
        print("请输入输出文件夹路径:")
        while True:
            output_folder = input("输出文件夹 > ").strip()
            
            if output_folder.lower() in ['q', 'quit', 'exit']:
                return folder_pairs
            
            if not output_folder:
                print("错误：输出文件夹不能为空")
                continue
            
            # 检查输出文件夹，如果不存在则创建
            if not os.path.exists(output_folder):
                create = input(f"文件夹 '{output_folder}' 不存在，是否创建？(y/n): ").strip().lower()
                if create in ['y', 'yes', '是']:
                    try:
                        os.makedirs(output_folder, exist_ok=True)
                        break
                    except Exception as e:
                        print(f"创建文件夹失败: {str(e)}")
                        continue
                else:
                    continue
            
            break
        
        # 添加到列表
        folder_pairs.append((input_folder, output_folder))
        print(f"✓ 已添加第 {pair_count} 对文件夹")
        
        # 询问是否继续
        if pair_count >= 1:
            continue_input = input("\n是否继续添加下一对文件夹？(y/n): ").strip().lower()
            if continue_input not in ['y', 'yes', '是']:
                break


# ==================== 从文件读取文件夹对 ====================
def read_folder_pairs_from_file(filepath):
    """
    从CSV或文本文件读取文件夹对
    
    文件格式可以是:
    1. CSV文件，包含input_folder和output_folder列
    2. 文本文件，每行包含两个路径，用逗号或制表符分隔
    """
    folder_pairs = []
    
    if not os.path.exists(filepath):
        print(f"错误：文件 '{filepath}' 不存在")
        return folder_pairs
    
    try:
        if filepath.endswith('.csv'):
            # 读取CSV文件
            df = pd.read_csv(filepath)
            if 'input_folder' in df.columns and 'output_folder' in df.columns:
                for _, row in df.iterrows():
                    folder_pairs.append((row['input_folder'], row['output_folder']))
            else:
                print("错误：CSV文件必须包含'input_folder'和'output_folder'列")
        else:
            # 读取文本文件
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 尝试用逗号、制表符或空格分割
                    parts = [p.strip() for p in line.replace('\t', ',').split(',')]
                    if len(parts) >= 2:
                        folder_pairs.append((parts[0], parts[1]))
                    else:
                        print(f"警告：第{line_num}行格式错误: {line}")
    
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
    
    return folder_pairs


# ==================== 主程序 ====================
def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理多个输入/输出文件夹对')
    parser.add_argument('--pairs-file', '-f', help='包含文件夹对的文件路径（CSV或文本）')
    parser.add_argument('--input-folders', '-i', nargs='+', help='输入文件夹列表')
    parser.add_argument('--output-folders', '-o', nargs='+', help='输出文件夹列表')
    parser.add_argument('--script', '-s', help='分析脚本路径')
    parser.add_argument('--delay', '-d', type=float, default=1.0, help='运行间隔延迟（秒）')
    parser.add_argument('--summary-dir', help='汇总输出目录')
    
    args = parser.parse_args()
    
    folder_pairs = []
    
    # 获取文件夹对
    if args.pairs_file:
        # 从文件读取
        folder_pairs = read_folder_pairs_from_file(args.pairs_file)
    elif args.input_folders and args.output_folders:
        # 从命令行参数获取
        if len(args.input_folders) != len(args.output_folders):
            print("错误：输入文件夹和输出文件夹数量必须相同")
            sys.exit(1)
        
        folder_pairs = list(zip(args.input_folders, args.output_folders))
    else:
        # 交互式输入
        folder_pairs = get_folder_pairs_interactively()
    
    if not folder_pairs:
        print("错误：没有提供任何文件夹对")
        sys.exit(1)
    
    # 显示将要处理的文件夹对
    print(f"\n{'='*80}")
    print("将要处理的文件夹对:")
    for i, (input_folder, output_folder) in enumerate(folder_pairs, 1):
        print(f"{i:3d}. 输入: {input_folder}")
        print(f"     输出: {output_folder}")
    print(f"{'='*80}")
    
    # 确认开始
    confirm = input(f"\n是否开始处理这 {len(folder_pairs)} 对文件夹？(y/n): ").strip().lower()
    if confirm not in ['y', 'yes', '是']:
        print("已取消处理")
        sys.exit(0)
    
    # 创建处理器并运行
    processor = BatchFolderProcessor(
        folder_pairs=folder_pairs,
        script_path=args.script,
        delay_between_runs=args.delay
    )
    
    # 运行所有文件夹对
    result = processor.run_all(summary_output_dir=args.summary_dir)
    
    print(f"\n处理完成！成功率: {result['success_rate']*100:.1f}%")


# ==================== 直接调用函数 ====================
def run_batch_processing(folder_pairs, script_path=None, delay=1, summary_dir=None):
    """
    直接调用函数，用于在其他脚本中使用
    
    参数:
    folder_pairs: 列表，每个元素是(input_folder, output_folder)元组
    script_path: 分析脚本路径
    delay: 运行间隔延迟（秒）
    summary_dir: 汇总输出目录
    
    返回:
    dict: 处理结果
    """
    processor = BatchFolderProcessor(
        folder_pairs=folder_pairs,
        script_path=script_path,
        delay_between_runs=delay
    )
    
    return processor.run_all(summary_output_dir=summary_dir)


# ==================== 示例使用 ====================
if __name__ == "__main__":
    # 示例1：直接指定文件夹对
    pairs = [
        ('/media/ubuntu/SRPPS/CODEX-b/test/', '/media/ubuntu/SRPPS/Results/CODEX_b_test_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_b-1/', '/media/ubuntu/SRPPS/Results/CODEX_B-1_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_B-11/', '/media/ubuntu/SRPPS/Results/CODEX_B-11_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_B-12/', '/media/ubuntu/SRPPS/Results/CODEX_B-12_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_B-14/', '/media/ubuntu/SRPPS/Results/CODEX_B-14_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_b-15/', '/media/ubuntu/SRPPS/Results/CODEX_B-15_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_b-16/', '/media/ubuntu/SRPPS/Results/CODEX_B-16_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_b-19/', '/media/ubuntu/SRPPS/Results/CODEX_B-19_output/'),
        
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_b-2/', '/media/ubuntu/SRPPS/Results/CODEX_B-2_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_b-5_9/', '/media/ubuntu/SRPPS/Results/CODEX_B-5_9_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_B-6/CODEX_B-6/', '/media/ubuntu/SRPPS/Results/CODEX_B-6_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_b-7/CODEX_b-7/', '/media/ubuntu/SRPPS/Results/CODEX_B-7_output/'),
        ('/media/ubuntu/SRPPS/CODEX-b/CODEX_b-10/', '/media/ubuntu/SRPPS/Results/CODEX_B-10_output/'),
    ]
    run_batch_processing(
        folder_pairs=pairs,
        script_path='/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/ALL_IN_ONE/Read_DecayPos/diff_detector.py',  # 使用默认脚本
        delay=0,          # 每次运行间隔2秒
        summary_dir='/media/ubuntu/SRPPS/Results/Batch_Summary/'  # 汇总输出目录
    )
    # 示例2：从CSV文件读取
    # csv_content = '''input_folder,output_folder
    # /media/ubuntu/SRPPS/CODEX-b/CODEX_B-11/,/media/ubuntu/SRPPS/Results/CODEX_B-11_output/
    # /media/ubuntu/SRPPS/CODEX-b/CODEX_B-12/,/media/ubuntu/SRPPS/Results/CODEX_B-12_output/
    # '''
    
    # 运行主程序
    # main()