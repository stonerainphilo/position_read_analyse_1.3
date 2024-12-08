import pandas as pd
from data_read import read_dataframe_single_mass
from fit_func import fit_tau_ln_theta_log10_single_mass_linear, fit_linear

def classify_by_mass(filename):
    df = pd.read_csv(filename)
    grouped = df.groupby('mH')
    #dataframes = {}
    # for value, group in grouped:
    #     dataframes[value] = group #
    return grouped

def mH_theta_log10_tau_ln_2D_function_fit(filename):  
    # persume for single mass, log10(theta) = k * ln(tau) + b
    
    grouped_by_mH = classify_by_mass(filename)
    df = {}
    df_functions = pd.DataFrame()
    for mH, group in grouped_by_mH:

        df[mH] = group
        mass, tau_ln, theta_log10 = read_dataframe_single_mass(df[mH])
        # print(type(mass))
        inter_, coef_, err_square_ = fit_tau_ln_theta_log10_single_mass_linear(tau_ln, theta_log10)
        # print(type(coef_))
        mH = pd.Series(mH)
        inter_ = pd.Series(inter_)
        coef_ = pd.Series(coef_)
        err_square_ = pd.Series(err_square_) 
        names = ["mH",
            "k",
            "b",
            "err^2"]
        df_functions_ = pd.concat([mH, coef_, inter_, err_square_], axis = 1, keys = names)
        df_functions =  df_functions._append(df_functions_, ignore_index = True)
    return df_functions

def mH_k_b_fit_linear_check(filename):
    df_functions = mH_theta_log10_tau_ln_2D_function_fit(filename)
    # print(fit_linear(df_functions["mH"].values.reshape(-1, 1), df_functions["k"].values.reshape(-1, 1)))
    k_b, k_k, k_err = fit_linear(df_functions["mH"], df_functions["k"])
    b_b, b_k, b_err = fit_linear(df_functions["mH"], df_functions["b"])
    df_ = pd.DataFrame({
        'intercep_of_m-k': [k_b],
        'slop_of_m-k': [k_k],
        'err_of_m-k': [k_err],
        'intercep_of_m-b': [b_b],
        'slop_of_m-b': [b_k],
        'err_of_m-b': [b_err]
    })
    # print(fit_linear(df_functions["mH"].values.reshape(-1, 1), df_functions["k"].values.reshape(-1, 1)))

    return df_