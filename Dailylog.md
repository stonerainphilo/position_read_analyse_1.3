### Daliy log

#### 2024-11-11----2024-11-23(0.5, 1, 1.5, 2, 2.5, 3 GeV)

The data set has a few problems:
    Mass between 1 and 2 has relatively well performance. 
    HOWEVER, the mass of 0.5 GeV has remarkable difference with our expectation: 
    the upper lim is significantly lower than it should be, 
    while the mass above 2 Gev(To be more specific: 2.5 and 3) 
    has unusually MORE data than it should be. 

Analyzation:
    Our simulation's error obiviously is relavant to LLP's mass. Naturally, we should consider simulation's part which has relation with particle's mass. 
    In our simulation: the

    ```C++
        IsResonance
        tauCalcu
    ```
    functions are related to mass.

    The sim we used during the test is:

    ```C++
        pythia.readString("999999:all = GeneralResonance void 0 0 0 " + doubleToScientificString(mA) + " 0 0 10 " + doubleToScientificString(ctau));
        pythia.readString("999999:addchannel = 1 1 101 13 -13");
        pythia.readString("999999:isResonance = true");
        // pythia.readString("ResonanceWidths:minWidth = 1e-30" + doubleToScientificString(calculate_Gamma_aka_minwidth(ctau)));
        pythia.readString("ResonanceWidths:minWidth = 1e-30");
        pythia.readString("999999:tauCalc = false");
    ```

Solution:

    ```c++
        pythia.readString("999999:all = GeneralResonance void 0 0 0 " + doubleToScientificString(mA) + " 0 0 10 " + doubleToScientificString(ctau));
        pythia.readString("999999:addchannel = 1 1 101 13 -13");
        pythia.readString("999999:isResonance = false");
        // pythia.readString("ResonanceWidths:minWidth = 1e-30" + doubleToScientificString(calculate_Gamma_aka_minwidth(ctau)));
        // pythia.readString("ResonanceWidths:minWidth = 1e-30");
        // pythia.readString("999999:tauCalc = false");
    ```
    Among which we set the LLP as NOT RESONANCE anymore. 

    RESONS:
        Our LLP's mass has no significant Giant mass, while the function:

    ```c++
        void ParticleDataEntry::setIsResonance(bool isResonance)  
        bool ParticleDataEntry::isResonance()  
    ```
        is said to be:

    ```c++
        void ParticleDataEntry::setIsResonance(bool isResonance)  
        bool ParticleDataEntry::isResonance()  
        a flag telling whether a particle species are considered as a resonance or not. Here "resonance" is used as shorthand for any massive particle where the decay process should be counted as part of the hard process itself, and thus be performed before showers and other event aspects are added. Restrictions on allowed decay channels is also directly reflected in the cross section of simulated processes, while those of normal hadrons and other light particles are not. In practice, it is reserved for states above the b bbar bound systems in mass, i.e. for W, Z, t, Higgs states, supersymmetric states and (most?) other states in any new theory. All particles with m0 above 20 GeV are by default initialized to be considered as resonances.
    ```

        Consider our LLP are NOT VERY MASSIVE, we set is as false to give it a shot.

#### 2024-11-24
Again, there's problems with simulation:
    The upper lim of 0.5 GeV is fixed, but the LOWER LIM IS WRONG as bad as it could be.
    Along with the 0.5 GeV, 3 GeV has fewer data and smaller range, but still not correct.

Solution:

    ```C++

        pythia.readString("999999:addchannel = 1 1 101 13 -13"); // memode is 101
        pythia.readString("999999:isResonance = true");
        // pythia.readString("ResonanceWidths:minWidth = 1e-30" + doubleToScientificString(calculate_Gamma_aka_minwidth(ctau)));
        // pythia.readString("ResonanceWidths:minWidth = 1e-30");
        pythia.readString("999999:tauCalc = true");

    ```


#### 2024-11-25

New thoughts:
    ResonanceDecay: memode???? 
    mWidth(\Gamma) is needed if 
    ```C++
    pythia.readString("999999:tauCalc = true");
    ```
    ???

    SEE PhaseSpace.cc MUST


    Seems we need to add every channel of our LLP to the PYTHIA8 programm
 About Onmode and memode:
    memode:
        在高能物理中，粒子的衰变宽度和分支比是描述粒子衰变特性的关键参数。以下是对这几种计算部分宽度的方法的解释：

        ### 100: 固定部分宽度

        - **计算方法**：部分宽度通过存储的分支比乘以存储的总宽度来计算。
        - **特点**：当共振粒子质量波动时，部分宽度保持不变。即使共振质量低于阈值，也假设衰变道仍然开放。

        ### 101: 阶跃阈值

        - **计算方法**：部分宽度通过存储的分支比乘以存储的总宽度，再乘以一个阶跃阈值。
        - **特点**：如果子粒子的质量和超过当前母粒子的质量，衰变道关闭。

        ### 102: 平滑阈值因子

        - **计算方法**：部分宽度通过存储的分支比乘以存储的总宽度，再乘以一个平滑阈值因子。
        - **两体衰变**：使用因子 \(\beta = \sqrt{(1 - m_1^2/m^2 - m_2^2/m^2)^2 - 4 m_1^2 m_2^2/m^4}\)。
        - **多体衰变**：使用因子 \(\sqrt{1 - \sum_i m_i / m}\)。
        - **特点**：考虑了相空间的大小，但忽略了矩阵元的复杂行为。

        ### 103: 修正的平滑阈值因子

        - **计算方法**：与 102 类似，但假设默认的分支比和总宽度已经考虑了阈值因子，因此需要除以在壳阈值因子。
        - **特点**：对于在壳质量，返回存储的分支比。为了避免除以零或不合理的缩放，设置了一个最小阈值。

        ### 总结

        这些方法用于模拟粒子衰变时，考虑了不同的物理条件和假设。选择合适的方法取决于对粒子质量波动和阈值效应的具体处理需求。


Solution:

    ```c++
    pythia.readString("Print:quiet = off");
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + std::to_string(seed));
    pythia.readString("Beams:eCM = 13000.");//  total Energy in Gev
    pythia.readString("HardQCD:hardbbbar  = on");
    // pythia.readString("HardQCD:all = on");
    pythia.readString("999999:all = GeneralResonance void 0 0 0 " + doubleToScientificString(mA) + " 0 0 10 " + doubleToScientificString(ctau));
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hmumu, 5) + " 102 13 -13"); // to mu-mu+
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hee, 5) + " 102 11 -11"); // to e-e+
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_HKK, 5) + " 102 311 311"); // to Kaon^0 Kaon^0
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_HPiPi, 5) + " 102 111 111"); // to pion pion
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Htautau, 5) + " 102 15 -15"); // to tau^- tau^+
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_HGluon, 5) + " 102 21 21"); // gg
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hgaga, 5) + " 102 22 22"); //gamma gamma
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_H4Pi, 5) + " 102 111 111 111 111"); // to 4 pion (uncharged)
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hss, 5) + " 102 3 -3"); // to s sbar
    pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hcc, 5) + " 102 4 -4"); // to c cbar
    pythia.readString("999999:isResonance = true");
    // pythia.readString("ResonanceWidths:minWidth = 1e-30" + doubleToScientificString(calculate_Gamma_aka_minwidth(ctau)));
    // pythia.readString("ResonanceWidths:minWidth = 1e-30");
    pythia.readString("999999:tauCalc = false");
    
    pythia.readString("999999:onMode = 1");
    pythia.readString("999999:MayDecay = true");
    // pythia.readString("ResonanceWidths:enhance = 999999 " + doubleToScientificString(br/2));
    // cout << "ok after prepare_pick " << endl;
    pythia.readString("521:oneChannel = 1 1 101 999999 321");
    pythia.readString("511:oneChannel = 1 1 101 999999 311");
    ```

    As shown above:
        All Branchings as their Ratio are provided and setted by a csv file.
        Unfortunately, this means I have to rewrited a lot of running functions.


#### 2024-11-29

Change:

    IsResonance = false
    tauCalc = false
    memode = 103


#### 2024-11-30

Change:

    HardQCD all: on
    ALL memode = 101

Change:

    HardQCD: hardbbbar = on //Only have B-mesons
    ALL memode = 101
    IsResonance = false
    tauCalc = false

Results:

    For mass = 0.3GeV, seems the tau input matches perfectly.
    Now will do more mass.

    HOWEVER, there are some doubts needed to be clearfied:
        1. After setting the 

        ```C++
            HardQCD: all = on
            ~~~~~~
            IsResonance = true
        ```

        The amount of data significantly dropped from 1700+/1kEvent, to 10/1kEvent.
        I think this is due to the amount of Bmesons. Cause:

        ```C++
            HardQCD: bbar = on
        ```
        The amount of data back to normal.

        But there still a question about

        ```C++
            IsResonance = true
        ```
        Will influence the data or not.
        During the test, at first we have ```IsResonance = false```, then the data remains unusually
        few and little. After seting ```IsResonance = true```, the data back to normal. 
        Then we setted ```IsResonance = false```, The data amount stays normal.

        I think the unusually phenomenon is due to latency of the make and ./. But if the 
        data cannot reach our expectation, this might be a point to be looking into.
    

    
#### 2024-12-04

NEW SULOTION For SETTINGS:

    If we set the mWidth(of BW-Distribution) in Pythia8 as we get in LSD, COULD IT BE ANY USE???

    ```C++
        pythia.readString("999999:all = GeneralResonance void 0 0 0 " + doubleToScientificString(mA) + " " + doubleToScientificString(mWidth) + " 0 20 " + doubleToScientificString(ctau));
    ```

    And, Since the Decay_Width is SIGNIFICALLY smaller than mass, I STILL think the ```IsResonace``` should be ```true```. So, I will set above as mentioned. 

NOTED AS ''NEW6''

#### 2025-2-26

OK I Vaguely remember how my code works, but completely forget the details. So basically, I'm going to review it and remember it.

    And it reminds me to write a manual THAT ACTUALLY READABLE.

#### 2025-3-2

OKKK I remember lots of details now, like how the hell to run pythia8 and LSD with my code, but my whole dictionary system is just imcomperhensive. And some of my pics gone lost. MAYBE they are in the other computer(I'm using the old MAC)?

About the whloe dictionary system:
    Unchangable at all. It needs too much of touble. Just Too Much.
    I might as well start a new folder whenever It is needed.

#### 2025-3-5

Rewrite the 'Readme' document, since the old version is USELESS.

Added CN ver of 'Readme', named `Readme_CN.txt`.


#### 2025-3-7

Provide support for Pythia83XX in `generate_save_analyse_judge_for_pythia83XX.ipynb`, but it isn't tested yet. The test will be performed LATER.

The old `Juypter Note Book` is now `generate_save_analyse_judge_for_pythia82XX.ipynb`, while removed some functions and instructions no longer needed.

Some functions in `loop.py, one_key_run.py, run_save.py` are added or removed.

Deleted LLP files in github for confidential reasons.

#### 2025-3-8

Rearranged the relation between `ALL_IN_ONE` Scheme and `generate_save_analyse_judge_for_pythia83XX.ipynb`, so that all funcs will exist only in 
`ALL_IN_ONE` Folder. Easier for read, change, and debug.

#### 2025-3-11

Fix bugs.

#### 2025-3-30 

Start working on the cross-section and 3-$\sigma$ significance problem. Expecting rule out the unusual mass data from 2.3GeV to 4.7 GeV by calculating Statical cross section and detection effciency. 

The new work is in the ALL_IN_ONE/Pythia8/.py, but mainly in ALL_IN_ONE/Pythia8/cross_section.py

The test function now completed and can be test as "from one_key_run import one_key_run_by_csv_cross_section_main41(main131)". 

Will test on newest data of 2~4 GeV(2025-3-29 and 2025-3-30).

To be updated