import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Load the data
dwi = pd.read_stata('hansen_dwi.dta')

# Pre-process the data
dwi['avg_acc'] = dwi.groupby('bac1')['acc'].transform('mean')
dwi.loc[dwi['bac1'] > 0.15, 'dui'] = 2

# Prepare data for linear fit
dwi_linear = dwi[['bac1', 'avg_acc']].drop_duplicates()

# OLS for linear fit
X_lin = sm.add_constant(dwi_linear['bac1'])
model_lin = sm.OLS(dwi_linear['avg_acc'], X_lin).fit()
prstd_lin, predict_mean_ci_low_lin, predict_mean_ci_upp_lin = wls_prediction_std(model_lin)

# Prepare data for quadratic fit
dwi['bac1_squared'] = dwi['bac1']**2
dwi_quadratic = dwi[['bac1', 'bac1_squared', 'avg_acc']].drop_duplicates().reset_index()

# OLS for quadratic fit
X_quad = sm.add_constant(dwi_quadratic[['bac1', 'bac1_squared']])
model_quad = sm.OLS(dwi_quadratic['avg_acc'], X_quad).fit()
prstd_quad, predict_mean_ci_low_quad, predict_mean_ci_upp_quad = wls_prediction_std(model_quad)

# Sorting for quadratic plot
sorted_order = np.argsort(dwi_quadratic['bac1'].values)
bac1_sorted = dwi_quadratic['bac1'].values[sorted_order]
fittedvalues_quad_sorted = model_quad.predict(X_quad).iloc[sorted_order]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Actual data points
ax.scatter(dwi_linear['bac1'], dwi_linear['avg_acc'], alpha=0.3, color='gray', edgecolor='white')

# Linear fit and confidence intervals
ax.plot(dwi_linear['bac1'], model_lin.predict(X_lin), color='blue', label='Linear Fit')
ax.fill_between(dwi_linear['bac1'], predict_mean_ci_low_lin, predict_mean_ci_upp_lin, color='blue', alpha=0.2)

# Quadratic fit and confidence intervals
ax.plot(bac1_sorted, fittedvalues_quad_sorted, color='red', label='Quadratic Fit')
ax.fill_between(bac1_sorted, predict_mean_ci_low_quad.iloc[sorted_order], predict_mean_ci_upp_quad.iloc[sorted_order], color='red', alpha=0.2)

# Vertical lines
ax.axvline(x=0.08, color='tomato', linestyle='--')
ax.axvline(x=0.15, color='tomato', linestyle='--')

# Customizing the plot
ax.set_xlim(0, 0.21)
ax.set_ylim(0, 0.26)
ax.set_xlabel('BAC', fontsize=17)
ax.set_ylabel('Average Accident', fontsize=17)
ax.set_title('Panel A. Accident at Scene', fontsize=20)
ax.legend()

plt.show()


# Pre-process the data for Panel B - Male
dwi['avg_male'] = dwi.groupby('bac1')['male'].transform('mean')
dwi.loc[dwi['bac1'] > 0.15, 'dui'] = 2

# Preparing data for linear fit
dwi_male = dwi[['bac1', 'avg_male']].drop_duplicates()

# OLS for linear fit
X_male_lin = sm.add_constant(dwi_male['bac1'])
model_male_lin = sm.OLS(dwi_male['avg_male'], X_male_lin).fit()
prstd_male_lin, predict_mean_ci_low_male_lin, predict_mean_ci_upp_male_lin = wls_prediction_std(model_male_lin)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot for actual data points
ax.scatter(dwi_male['bac1'], dwi_male['avg_male'], alpha=0.3, color='gray', edgecolor='white')

# Linear model and CI
ax.plot(dwi_male['bac1'], model_male_lin.predict(X_male_lin), color='blue', label='Linear Fit')
ax.fill_between(dwi_male['bac1'], predict_mean_ci_low_male_lin, predict_mean_ci_upp_male_lin, color='blue', alpha=0.2)

# Vertical lines at BAC = 0.08 and BAC = 0.15
ax.axvline(x=0.08, color='tomato', linestyle='--')
ax.axvline(x=0.15, color='tomato', linestyle='--')

# Customizing the plot
ax.set_xlim(0, 0.21)
ax.set_ylim(0.73, 0.83)
ax.set_xlabel('BAC', fontsize=17)
ax.set_ylabel('Average Male', fontsize=17)
ax.set_title('Panel B. Male', fontsize=20)
ax.set_xticks([0.05, 0.1, 0.15, 0.2])
ax.set_yticks([0.74, 0.76, 0.78, 0.8, 0.82])
ax.legend()

plt.show()
#panel 3




# Pre-process the data for Panel C - Age
dwi['avg_age'] = dwi.groupby('bac1')['aged'].transform(lambda x: np.mean(x / 100))
dwi['dui_group'] = np.where(dwi['bac1'] > 0.15, 'Above 0.15', 'Below or Equal 0.15')

# Preparing data for linear fit
dwi_age = dwi[['bac1', 'avg_age', 'dui_group']].drop_duplicates()

# OLS for linear fit
X_age_lin = sm.add_constant(dwi_age['bac1'])
model_age_lin = sm.OLS(dwi_age['avg_age'], X_age_lin).fit()
prstd_age_lin, predict_mean_ci_low_age_lin, predict_mean_ci_upp_age_lin = wls_prediction_std(model_age_lin)

# Preparing data for quadratic fit
dwi_age['bac1_squared'] = dwi_age['bac1']**2
X_age_quad = sm.add_constant(dwi_age[['bac1', 'bac1_squared']])
model_age_quad = sm.OLS(dwi_age['avg_age'], X_age_quad).fit()
prstd_age_quad, predict_mean_ci_low_age_quad, predict_mean_ci_upp_age_quad = wls_prediction_std(model_age_quad)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot for actual data points with different colors based on DUI group
for name, group in dwi_age.groupby('dui_group'):
    ax.scatter(group['bac1'], group['avg_age'], alpha=0.3, label=name)

# Linear and Quadratic models
ax.plot(dwi_age['bac1'], model_age_lin.predict(X_age_lin), color='blue', label='Linear Fit')
ax.fill_between(dwi_age['bac1'], predict_mean_ci_low_age_lin, predict_mean_ci_upp_age_lin, color='blue', alpha=0.2)

ax.plot(dwi_age['bac1'], model_age_quad.predict(X_age_quad), color='red', label='Quadratic Fit')
ax.fill_between(dwi_age['bac1'], predict_mean_ci_low_age_quad, predict_mean_ci_upp_age_quad, color='red', alpha=0.2)

# Vertical lines at BAC = 0.08 and BAC = 0.15
ax.axvline(x=0.08, color='tomato', linestyle='--')
ax.axvline(x=0.15, color='tomato', linestyle='--')

# Customize the plot
ax.set_xlim(0, 0.21)
ax.set_ylim(0.33, 0.39)
ax.set_xlabel('BAC', fontsize=17)
ax.set_ylabel('Average Age', fontsize=17)
ax.set_title('Panel C. Age', fontsize=20)
ax.set_xticks([0.05, 0.1, 0.15, 0.2])
ax.set_yticks([0.34, 0.35, 0.36, 0.37, 0.38])
ax.legend(title='DUI Group')

plt.show()
