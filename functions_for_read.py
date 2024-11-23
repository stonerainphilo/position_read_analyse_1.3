import pandas as pd
def convert_to_float(x):
    try:
        return float(x)
    except ValueError:
        print('error!!!!!!')
        return None
    
def Read_csv(path_trimed):
    df = pd.read_csv(path_trimed)
    rows = df.loc[df['id'] == '999999'].copy()  # Creat a copy to avoid Warning
    rows['m'] = rows['m'].apply(convert_to_float)
    return rows['m']

def get_llp_with_column_name(filename, column_name, id=999999):
    df = pd.read_csv(filename)
    particle = df[df['id'] == id][column_name]
    return particle.tolist()

def get_llp(filename, id=999999):
    df = pd.read_csv(filename)
    particle = df[df['id'] == id]
    return particle
