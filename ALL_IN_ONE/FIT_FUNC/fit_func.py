from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def fit_tau_ln_theta_log10_single_mass_linear(tau_ln, theta_log10, test_size_ = 0.2, random_ = 0):
    tau_ln = tau_ln.values.reshape(-1, 1)
    theta_log10 = theta_log10.values.reshape(-1, 1)
    tau_ln_train, tau_ln_test, theta_log10_train, theta_log10_test = train_test_split(tau_ln, theta_log10, test_size=test_size_, random_state=random_)
    model = LinearRegression()  
    model.fit(tau_ln_train, theta_log10_train) 
    theta_log10_pred = model.predict(tau_ln_test)
    # print('截距:', model.intercept_)
    # print('斜率:', model.coef_)
    # print('均方误差:', metrics.mean_squared_error(theta_test, theta_pred))
    return model.intercept_[0], model.coef_[0][0], metrics.mean_squared_error(theta_log10_test, theta_log10_pred)

def fit_linear(x, y, test_size_ = 0.2, random_ = 0):
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_, random_state=random_)
    model = LinearRegression()  
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_test)
    # print('截距:', model.intercept_)
    # print('斜率:', model.coef_)
    # print('均方误差:', metrics.mean_squared_error(theta_test, theta_pred))
    return model.intercept_[0], model.coef_[0][0], metrics.mean_squared_error(y_test, y_pred)
