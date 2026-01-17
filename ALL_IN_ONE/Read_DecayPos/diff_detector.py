import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/ALL_IN_ONE")
from Pythia8.functions_for_calculation import whether_in_the_detector_by_position
from Pythia8.cross_section import counting_total_LLP
from Pythia8.functions_for_run import mkdir_1


def get_user_input():
    """
    获取用户输入的文件夹路径
    """
    print("\n" + "="*60)
    print("FASER/CODEX-b/MATHUSLA 探测器接受度计算程序")
    print("="*60)
    
    # 输入数据文件夹路径
    print("\n请输入LLP数据文件夹路径:")
    print("示例: /media/ubuntu/SRPPS/CODEX-b/CODEX_B-11/")
    while True:
        input_folder = input("> ").strip()
        
        # 检查输入是否为空
        if not input_folder:
            print("错误：输入不能为空，请重新输入")
            continue
            
        # 检查路径是否存在
        if not os.path.exists(input_folder):
            print(f"错误：路径 '{input_folder}' 不存在，请重新输入")
            continue
            
        # 检查是否为文件夹
        if not os.path.isdir(input_folder):
            print(f"错误：'{input_folder}' 不是文件夹，请重新输入")
            continue
            
        break
    
    # 输入输出文件夹路径
    print("\n请输入输出结果文件夹路径（直接回车使用默认路径）:")
    print("默认：在输入文件夹下创建 'Detect_ALL' 子文件夹")
    
    while True:
        output_folder = input("> ").strip()
        
        # 如果用户直接回车，使用默认路径
        if not output_folder:
            output_folder = os.path.join(os.path.dirname(input_folder), 'Detect_ALL')
            print(f"使用默认路径: {output_folder}")
            break
            
        # 检查路径是否存在，如果不存在则创建
        if not os.path.exists(output_folder):
            create = input(f"路径 '{output_folder}' 不存在，是否创建？(y/n): ").strip().lower()
            if create in ['y', 'yes', '是']:
                try:
                    os.makedirs(output_folder, exist_ok=True)
                    print(f"已创建文件夹: {output_folder}")
                    break
                except Exception as e:
                    print(f"创建文件夹失败: {str(e)}")
                    continue
            else:
                print("请重新输入输出文件夹路径:")
                continue
        
        # 检查是否为文件夹
        if not os.path.isdir(output_folder):
            print(f"错误：'{output_folder}' 不是文件夹，请重新输入")
            continue
            
        break
    
    # 显示探测器配置
    print("\n" + "-"*60)
    print("探测器配置:")
    print("1. FASER: 距离IP点480m，长度1.5m，半径0.1m")
    print("2. CODEX-b: 使用默认配置")
    print("3. MATHUSLA: 使用默认配置")
    print("-"*60)
    
    # 确认开始处理
    print("\n准备开始处理:")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    
    confirm = input("\n是否开始处理？(y/n): ").strip().lower()
    if confirm not in ['y', 'yes', '是']:
        print("已取消处理")
        sys.exit(0)
    
    return input_folder, output_folder

# ==================== FASER探测器判断函数 ====================
def whether_in_FASER_by_position(x_pos, y_pos, z_pos, distance_from_ip=480000, length=1500, radius=100):
    """
    判断粒子是否在FASER探测器中衰变
    
    参数:
    x_pos, y_pos, z_pos: 衰变位置的x, y, z坐标 (米)
    distance_from_ip: 探测器前端距离IP点的距离 (米)，默认480.0
    length: 探测器长度 (米)，默认1.5
    radius: 探测器半径 (米)，默认0.1
    
    返回:
    list: 每个位置对应的判断结果 (True/False)
    """
    # 计算探测器z轴范围
    z_start = distance_from_ip
    z_end = distance_from_ip + length
    
    # 初始化结果列表
    results = []
    
    # 遍历每个位置进行判断
    for x, y, z in zip(x_pos, y_pos, z_pos):
        # 1. 检查z坐标是否在探测器范围内
        in_z_range = z_start <= z <= z_end
        
        # 2. 计算径向距离
        radial_distance = np.sqrt(x**2 + y**2)
        
        # 3. 检查径向距离是否在探测器半径内
        in_radial_range = radial_distance <= radius
        
        # 4. 综合判断
        is_inside = in_z_range and in_radial_range
        
        results.append(is_inside)
    
    return results


def whether_in_FASER2_by_position(x_pos, y_pos, z_pos, distance_from_ip=480000, length=5000, radius=1000):
    """
    判断粒子是否在FASER探测器中衰变
    
    参数:
    x_pos, y_pos, z_pos: 衰变位置的x, y, z坐标 (米)
    distance_from_ip: 探测器前端距离IP点的距离 (米)，默认480.0
    length: 探测器长度 (米)，默认1.5
    radius: 探测器半径 (米)，默认0.1
    
    返回:
    list: 每个位置对应的判断结果 (True/False)
    """
    # 计算探测器z轴范围
    z_start = distance_from_ip
    z_end = distance_from_ip + length
    
    # 初始化结果列表
    results = []
    
    # 遍历每个位置进行判断
    for x, y, z in zip(x_pos, y_pos, z_pos):
        # 1. 检查z坐标是否在探测器范围内
        in_z_range = z_start <= z <= z_end
        
        # 2. 计算径向距离
        radial_distance = np.sqrt(x**2 + y**2)
        
        # 3. 检查径向距离是否在探测器半径内
        in_radial_range = radial_distance <= radius
        
        # 4. 综合判断
        is_inside = in_z_range and in_radial_range
        
        results.append(is_inside)
    
    return results


# ==================== CODEX-b探测器判断函数 ====================
def whether_in_CODEX_b_by_position(x_pos, y_pos, z_pos, detector_type="CODEX-b"):
    """
    判断粒子是否在CODEX-b探测器中衰变
    使用原函数的参数，保持兼容性
    """
    return whether_in_the_detector_by_position(x_pos, y_pos, z_pos)


# ==================== MATHUSLA探测器判断函数 ====================
def whether_in_MATHUSLA_by_position(x_pos, y_pos, z_pos):
    """
    判断粒子是否在MATHUSLA探测器中衰变
    使用原函数的参数，保持兼容性
    """
    return whether_in_the_detector_by_position(x_pos, y_pos, z_pos, 
                                                -100000, 100000, 
                                                100000, 125000, 
                                                100000, 300000)


# ==================== 主处理函数 ====================
def detect_folder_files_cross_section_all_detectors(LLP_data_folder_dir, final_data_folder=None):
    """
    处理文件夹中的所有LLP数据文件，计算在各个探测器中的接受度
    
    参数:
    LLP_data_folder_dir: 包含LLP数据文件的文件夹路径
    final_data_folder: 输出结果的文件夹路径，如果为None则自动创建
    
    返回:
    str: 处理完成的状态信息
    """
    # 创建输出文件夹
    if final_data_folder:
        mkdir_1(final_data_folder)
    else:
        final_data_folder = os.path.dirname(LLP_data_folder_dir) + '/Detect_ALL'
        mkdir_1(final_data_folder)
    
    # 初始化汇总DataFrame
    all_llp = pd.DataFrame()
    all_llp['m'] = []
    all_llp['tau'] = []
    all_llp['CODEX-b_acceptance'] = []
    all_llp['MATHUSLA_acceptance'] = []
    all_llp['FASER_acceptance'] = []  # 新增FASER接受度
    all_llp['FASER2_acceptance'] = []  # 新增FASER2接受度
    
    print(f"开始处理文件夹: {LLP_data_folder_dir}")
    print(f"输出文件夹: {final_data_folder}")
    
    # 获取文件列表
    file_list = [f for f in os.listdir(LLP_data_folder_dir) 
                 if os.path.isfile(os.path.join(LLP_data_folder_dir, f))]
    
    # 遍历处理每个文件
    for filename in tqdm(file_list, desc="Processing files"):
        filepath = os.path.join(LLP_data_folder_dir, filename)
        
        try:
            # 读取数据文件
            llp_data = pd.read_csv(filepath)
            
            # 创建检测结果DataFrame
            llp_detect = pd.DataFrame()
            
            # 判断粒子是否在各个探测器中
            llp_in_CODEX_b = whether_in_CODEX_b_by_position(
                llp_data['decay_pos_x'], 
                llp_data['decay_pos_y'], 
                llp_data['decay_pos_z']
            )
            
            llp_in_MATHUSLA = whether_in_MATHUSLA_by_position(
                llp_data['decay_pos_x'], 
                llp_data['decay_pos_y'], 
                llp_data['decay_pos_z']
            )
            
            llp_in_FASER = whether_in_FASER_by_position(
                llp_data['decay_pos_x'], 
                llp_data['decay_pos_y'], 
                llp_data['decay_pos_z'],
                distance_from_ip=480000,  # FASER距离IP 480米
                length=1500,              # FASER长度1.5米
                radius=100               # FASER半径0.1米
            )
            
            llp_in_FASER2 = whether_in_FASER2_by_position(
                llp_data['decay_pos_x'], 
                llp_data['decay_pos_y'], 
                llp_data['decay_pos_z'],
                distance_from_ip=480000,  # FASER2距离IP 480
                length=5000,              # FASER2长度5米
                radius=1000               # FASER2半径1米
            )
            # 计算总LLP数量
            total_llp = counting_total_LLP(llp_data)
            
            # 计算各个探测器的接受度
            llp_detect['detected_CODEX-b'] = llp_in_CODEX_b
            llp_detect['detected_MATHUSLA'] = llp_in_MATHUSLA
            llp_detect['detected_FASER'] = llp_in_FASER
            
            # 获取截面值（假设所有行都有相同的截面）
            cross_section = llp_data['Cross_section_fb'].iloc[0]
            
            # 计算接受度：探测器内衰变的粒子数 × 截面 / 总LLP数
            codex_acceptance = sum(llp_in_CODEX_b) * cross_section / total_llp
            mathusla_acceptance = sum(llp_in_MATHUSLA) * cross_section / total_llp
            faser_acceptance = sum(llp_in_FASER) * cross_section / total_llp
            faser2_acceptance = sum(llp_in_FASER2) * cross_section / total_llp
            # 添加到汇总数据
            all_llp = pd.concat([all_llp, pd.DataFrame({
                'm': [llp_data['m'].iloc[0]],
                'tau': [llp_data['tau'].iloc[0]],
                'CODEX-b_acceptance': [codex_acceptance],
                'MATHUSLA_acceptance': [mathusla_acceptance],
                'FASER_acceptance': [faser_acceptance],
                'FASER2_acceptance': [faser2_acceptance]
            })], ignore_index=True)
            
            # 保存单个文件的结果（可选）
            # single_result_path = os.path.join(final_data_folder, f"result_{filename}")
            # llp_detect.to_csv(single_result_path, index=False)
            
        except Exception as e:
            print(f"\n处理文件出错: {filename}")
            print(f"错误信息: {str(e)}")
            continue
    
    # 保存汇总结果
    output_path = os.path.join(final_data_folder, 'all_llp_detect_all_detectors_cross_section.csv')
    all_llp.to_csv(output_path, index=False)
    
    print(f"\n处理完成！结果已保存到: {output_path}")
    return 'Detection and Calculation Cross-Section Completed for All Detectors'

def main():
    """
    主函数，程序入口
    """
    try:
        # 获取用户输入
        input_folder, output_folder = get_user_input()
        
        # 执行处理
        result = detect_folder_files_cross_section_all_detectors(input_folder, output_folder)
        
        print(f"\n{result}")
        
        # 询问是否查看结果
        view_result = input("\n是否打开输出文件夹查看结果？(y/n): ").strip().lower()
        if view_result in ['y', 'yes', '是']:
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_folder)
                elif os.name == 'posix':  # Linux/Mac
                    os.system(f'xdg-open "{output_folder}"')
            except:
                print(f"无法自动打开文件夹，请手动访问: {output_folder}")
                
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
    finally:
        print("\n程序结束")


# ==================== 直接调用模式（保持兼容性） ====================
def direct_mode(folder=None, out_folder=None):
    """
    直接调用模式，用于脚本调用
    """
    if folder is None:
        print("错误：请提供输入文件夹路径")
        return
    
    if out_folder is None:
        out_folder = os.path.join(os.path.dirname(folder), 'Detect_ALL')
    
    return detect_folder_files_cross_section_all_detectors(folder, out_folder)


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 检查是否通过命令行参数调用
    if len(sys.argv) > 1:
        # 命令行模式
        if len(sys.argv) >= 3:
            folder = sys.argv[1]
            out_folder = sys.argv[2]
        elif len(sys.argv) >= 2:
            folder = sys.argv[1]
            out_folder = None
        else:
            folder = None
            out_folder = None
        
        if folder:
            result = direct_mode(folder, out_folder)
            print(result)
        else:
            main()
    else:
        # 交互式模式
        main()


# # ==================== 示例使用 ====================
# if __name__ == "__main__":
#     # 示例文件夹路径
#     folder = '/media/ubuntu/SRPPS/CODEX-b/CODEX_B-11/'
#     out_folder = '/media/ubuntu/SRPPS/CODEX-b/CODEX_b-11/Detect_ALL/'
    
#     # 调用主处理函数
#     result = detect_folder_files_cross_section_all_detectors(folder, out_folder)
#     print(result)
    
#     # 你也可以使用自动创建的输出文件夹
#     # result = detect_folder_files_cross_section_all_detectors(folder)
#     # print(result)