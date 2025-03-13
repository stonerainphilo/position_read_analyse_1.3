- [English](ReadMe.md)
- [中文](ReadMe.zh.md)


### What do I Need to Run the Package?
    1. Python 3.10.11 or later version
        (maybe older version can do as well, but I use 3.10.11, and I didn't do any test to ensure it can or cannot progress normally, so I just stick with 3.10.11 to avoid unpredicted trouble)
        And you need to install quite a few packages. 

    2. Pythia 8
        (There are some specific requirements for different version of Pythia 8. The code itself is originally writen for an older verison of Pythia 8, but can be used for 8.3, which is the newest version.)
        
        PAY ATTENTION:
        a. Requires Pythia 8 config with HepMC2.
        b. Requires the path of Pythia 8.

    3. Light Scalar Decay(LSD)
        A package for a new BSM Model, provide calculation of DecayWidth, Branch Functions and etc.



### How to Run This Program?
    Open the 'generate_save_analyse_judge_for_pythia83XX.ipynb'(if you are using the pythia8.2, use `generate_save_analyse_judge_for_pythia82XX.ipynb`), follow the instructions and run cells in order.

    Other .ipynb files are not necessary for simple use.
    
    ALL .py files are functions needed for necessary run. You don't have to change it unless it's needed for your use. 

# ATTENTION:
    If you want to use this while customize your Code, please creat another folder outside of the `main` brunch. For you might want to use this repository to get future update, use .gitignore to make sure your personal settings won't mess up others' codespace.
    
    
### What does the Out_put csv files contain?
    The csv files contain LLPs `br, ctau(tau), mass(m), energy(e), momentum(p_x/y/z)`
        ``production position(x/y/z_Prod), decay position x/y/z``
        `Whether can be detected (0 is cannot, 1 is can)`
    
    To plot, you need to type in the folder path for .csv data files in `plot.ipynb`.
        
    


### The 'ALL IN ONE' FOLDER

    It's a plan to write an interface for Pythia8 and Light Scalar Decay, as well as automatically process the results and data, aimming for simpler and quicker simulation of LLP. 
    At present, the analyse and functions for Pythia8 are roughly completed and can be used for testing. But functions for LSD are not ready yet. 