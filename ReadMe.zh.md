# 我需要哪些环境来运行这个程序包？
    1. Python 3.10.11 或更高版本
        （旧版本可能也能运行，但我使用3.10.11版本，且未进行向下兼容测试。建议使用3.10.11以避免不可预见的错误）
        需要安装若干依赖包

    2. Pythia 8 粒子物理软件
        （不同版本的Pythia 8有特定要求。本代码最初为旧版Pythia 8编写，但兼容最新的8.3版本）
        
        重要注意事项：
        a. 必须配置支持HepMC2的Pythia 8环境
        b. 需要正确设置Pythia 8的安装路径


# 如何运行程序？
## 对于CODEX-b探测器
    如果你在使用Pythia8.3及其衍生版本，打开'generate_save_analyse_judge_for_pythia83XX.ipynb'以使用；
    如果你在使用Pythia8.2或其他仍然存在\example\main41.cc的版本，请使用'generate_save_analyse_judge_for_pythia82XX.ipynb'。
    
    其他.ipynb文件非必需文件，基础使用无需操作。
    所有.py文件是核心功能模块，除非有定制需求，否则无需修改。

## 对于其他任意多面体探测器

    您可以使用 ALL_IN_ONE/Detector 程序来定制任何多边形检测器的 generate_save_analyse_judge_for_pythia.ipynb。

    该程序采用 Möller–Trumbore 方法进行判断。但是，它尚未应用于任何正式代码。

# 注意

    如果你想在实时接收更新的前提下自定义你的代码，请活用.gitignore来确保你的个人设定不会被更新到main brunch中。

# 输出csv文件包含哪些信息？
    CSV文件记录长寿命粒子（LLP）的以下参数：
        '衰变分支比（br）、固有寿命（ctau）、质量（m）、能量（e）、动量分量（p_x/y/z）'
        '产生位置坐标（x/y/z_Prod）、衰变位置坐标（x/y/z_Decay）'
        '可探测性标识（0表示不可探测，1表示可探测）'
    
    绘制图像时，需在"plot.ipynb"中输入.csv数据文件的存储路径。


# "ALL IN ONE"整合项目说明

    该项目旨在开发Pythia8与轻标量粒子衰变（LSD）的联合接口，实现自动化的数据处理和结果分析，目标是构建简捷高效的长寿命粒子模拟平台。
    当前进度：
    - Pythia8相关分析模块已基本完成，可进行测试
    - LSD功能模块仍在开发中，尚未投入使用
    