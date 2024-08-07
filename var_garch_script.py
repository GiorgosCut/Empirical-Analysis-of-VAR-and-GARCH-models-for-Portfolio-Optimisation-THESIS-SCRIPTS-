import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from arch import arch_model
import itertools
from statsmodels.stats.diagnostic import het_arch


#setup an empty dictionary which will be used to store a dataframe with the name of the asset it represents
dataframes = {}

#we want to loop over all files in the Assets folder, and importing into the dictionary the data of each asset and its name as the key
for file in os.listdir('Final_Assets'):
    #if the file is a .csv file we extract its info
    if file.endswith('.csv'):
        #construct the path name of the file
        path = os.path.join('Final_Assets', file)
        #load it up as a dataframe using pandas
        df = pd.read_csv(path)
        #get the key to the dictionary entry, which is the file name without .csv
        key = os.path.splitext(file)[0]
        #finally add the dataframe and key into the dictionary
        dataframes[key] = df
    else:
        print('not csv')


#We want to calculate the linear returns and load them up into a new dictionary

#empty dict for lin rets
lin_rets_train = {}
lin_rets_val = {}

#we now loop over each key in the dataframes dict
for key, value in dataframes.items():

    #we want to extract data from 2017-2023 (training set)
    #turn the format of the dates into a pandas datetime so we can manipulate them
    value['Date'] = pd.to_datetime(value['Date'])

    #filter out the date range we want
    training_set = value[(value['Date'] >= '2017-01-01') & (value['Date'] <= '2023-12-31')]
    validation_set = value[(value['Date'] > '2023-12-31') & (value['Date'] <= '2024-12-31')]

    #get adjusted close prices
    adj_close_train = training_set.iloc[:,5]
    adj_close_val = validation_set.iloc[:,5]
    #simplest way to simulate the lin rets formula is to use shift()
    #this is a functions which allows us to shift the data in a dataframe by the number specified
    linr_train = (adj_close_train - adj_close_train.shift(1)) / adj_close_train.shift(1)
    linr_val = (adj_close_val - adj_close_val.shift(1)) / adj_close_val.shift(1)
    #this naturally leaves a NaN value at index 0 so we remove it
    linr_train = linr_train.dropna()
    linr_val = linr_val.dropna()
    #finally construct our lin_rets dictionary
    lin_rets_train[key] = linr_train
    lin_rets_val[key] = linr_val


#turn lin_rets dictionary into a pandas dataframe so we can feed the data into stats models functions
lin_rets_train_df = pd.DataFrame(lin_rets_train)
lin_rets_val_df = pd.DataFrame(lin_rets_val)
#we want the 19 days after the end of the training data to compare to 19 days of predictions (Change if nessesary)
lin_rets_val_df = lin_rets_val_df.head(19)

print(lin_rets_train_df.shape)

#create a VAR object, passing it the training data
model = VAR(lin_rets_train_df)

#change lag length until you find most optimal
fit = model.fit(25)


# Forecasting the next 5 periods
forecast_steps = 19
#get lag number used in model (determined by criterion)
lags = fit.k_ar

#get lagged data to pass to prediction function
input_data = lin_rets_train_df.values[-lags:]

# Forecast the future values
forecast = fit.forecast(y=input_data, steps=forecast_steps)
forecast = pd.DataFrame(forecast, columns=list(lin_rets_train.keys()))

#get the residuals from the actual data and the predicted
res = lin_rets_val_df.values - forecast.values
residuals_df = pd.DataFrame(res, columns=lin_rets_train_df.columns)

# Calculate sum of squared residuals
sum_squared_residuals = np.sum(np.square(res))
#print("Sum of Squared Residuals:", sum_squared_residuals)

# Plotting actual data, predicted data, and residuals
plt.figure(figsize=(10, 5))
for column in residuals_df.columns:
    plt.plot(lin_rets_val_df.index[:forecast_steps], lin_rets_val_df[column][:forecast_steps], 'bo')
    plt.plot(lin_rets_val_df.index[:forecast_steps], forecast[column], 'ro')
    for i in range(forecast_steps):
        plt.plot([lin_rets_val_df.index[i], lin_rets_val_df.index[i]], 
                 [lin_rets_val_df[column].iloc[i], forecast[column].iloc[i]], 
                 'y-')

#custom legend
legend_elements = [
    Line2D([0], [0], color='red', marker='o', linestyle='None', label='Predicted'),
    Line2D([0], [0], color='blue', marker='o', linestyle='None', label='Actual'),
    Line2D([0], [0], color='yellow', lw=2, label='Residual')
]


plt.title('Actual vs Predicted Data and Residuals, 25 Lags included')
plt.xlabel('Time')
plt.ylabel('Linear Returns')
plt.legend(handles=legend_elements, loc='upper right', fontsize='small')
mid_index = forecast_steps // 2
xticks = [0, mid_index, forecast_steps - 1]
xtick_labels = [1, mid_index + 1, forecast_steps]

plt.gca().set_xticks(lin_rets_val_df.index[xticks])
plt.gca().set_xticklabels(xtick_labels)
plt.grid(True)
plt.show()


##plot the averages as a wave to see movement

# Calculate the average for each step
actual_avg = lin_rets_val_df.mean(axis=1)
predicted_avg = forecast.mean(axis=1)


plt.figure(figsize=(10, 5))
plt.plot(range(1, len(actual_avg)+1), actual_avg, label='Real data', color='blue')
plt.plot(range(1, len(predicted_avg)+1), predicted_avg, label='Predicted', color='black')

plt.title('Real vs Predicted Data (Averages)')
plt.xlabel('Time Steps')
plt.ylabel('Average Returns')
plt.legend()
plt.grid(True)
plt.show()

#First part done: predicted returns means
#print(forecast.sum(axis=0))
resids = fit.resid
#while running the GARCH fitting below i got scaling error so i had to scale the residuals by 100
resids = resids * 100

####GARCH

#empty dictionary to put the final models
garch_models = {}

#TEST FOR ARCH EFFECTS
for asset in resids.columns:
    #print(f"Results for {asset}:")
    
    #sample GARCH(1,1) model to access its residuals
    garch_model = arch_model(resids[asset], vol='Garch', p=1, q=1)
    result = garch_model.fit(disp='off')
    
    # Extract residuals from the fitted GARCH model
    residuals = result.resid
    
    # Perform the ARCH LM test for 12 lags
    lm_test = het_arch(residuals, nlags=8)
    # print(f"LM Test Statistic: {lm_test[0]}")
    # print(f"LM Test p-value: {lm_test[1]}")
    # print(f"F-Statistic: {lm_test[2]}")
    # print(f"F p-value: {lm_test[3]}")
    # print("=" * 40)  # Print a separator between asset results


#HERE WE SEE THAT ASSET CHTR DOES NOT HAVE ARCH EFFECT, THEREFORE CANNOT USE GARCH MODELS TO MODEL ITS VOLATILITY
#ONLY SOLUTION IS TO DROP THIS ASSET

#fit GARCH(8,8) for 9 rest assets
resids = resids.drop(columns='CHTR')

for asset in resids.columns:
    #fit GARCH(8,8) model 
    garch_model = arch_model(resids[asset], vol='Garch', p=8, q=8)
    result = garch_model.fit(disp='off')
    garch_models[asset] = result

#from our dictionary use this cool dictionary comprehension to extract the conditional volatility
#of each asset from their results and constract a new df
cond_vols = pd.DataFrame({asset: garch_models[asset].conditional_volatility for asset in resids.columns})
var_matrix = cond_vols.cov()
print(var_matrix)

#plot volatility clusters
groups = [
    ['ADBE', 'AMGN', 'BMY'],
    ['DLR', 'EQIX', 'KMB'],
    ['PFE', 'TMUS', 'VZ']
]

for group in groups:
    plt.figure(figsize=(12, 8))
    for asset in group:
        if asset in garch_models:
            cond_vol = garch_models[asset].conditional_volatility
            plt.plot(cond_vol.index, cond_vol, label=f'{asset} Volatility')
    plt.title(f'Volatility Clustering for Assets {", ".join(group)}')
    plt.xlabel('Time step')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()


##PORTFOLIO CONSTRUCTION

#we want to turn our data into numpy arrays, in order to perform easier calculations with them

cov = var_matrix.to_numpy()
mean = forecast.drop(columns='CHTR').sum(axis=0).to_numpy().reshape(-1, 1)

#we construct the information matrix

#vector of 9 1s
ones = np.ones((9,1))
#inverse of cov
cov_inv = np.linalg.inv(cov)

#doing it by parts

#i got so many errors here about the matrix dimensions when multiplying them, i needed to do all this testing

print(mean.shape)
print(ones.shape)
print(cov_inv.shape)
stack = np.hstack((mean, ones))
print(stack.shape)

#left multiplication
stack_t = stack.T
left = np.dot(stack_t, cov_inv)

#result of above should be (2,9) matrix
#print(left.shape)

#multiply left result with right component
inf_matrix = np.dot(left, stack)

#print(inf_matrix)
#inf matrix is correctly 2x2

#get weights for MVP portfolio
#indexis starting from 0
weights = (1/inf_matrix[1,1]) * np.dot(cov_inv, ones)
#print(weights)

lin_rets_val_df = pd.DataFrame(lin_rets_val).drop(columns='CHTR')
#print(lin_rets_val_df)


##BACKTESTING

#calculate the portfolio returns for each timestep of the validation data
#column vector times matrix of returns
portf_rets = lin_rets_val_df.dot(weights)
print(portf_rets)

#say we have 10000 total capital investment
capital = 10000

#create a list, which for each time will store how much the total capital amount changes
portf_value = [capital]

#for each timestep in the validation step (106 days), get the value of the portfolio that day and store it
for i in range(len(portf_rets)):
    curr_ret = portf_rets.iloc[i, 0]
    #get last day's portfolio value and multiply it by the return of the portfolio today + 1.  list[-1] gets the last element in a list
    temp = portf_value[-1] * (1 + curr_ret)
    #add new value to list
    portf_value.append(temp)

#convert to series for easier plotting
portf_value_ser = pd.Series(portf_value[1:], index=portf_rets.index)

# Plot the portfolio value over time
plt.figure(figsize=(14, 7))
portf_value_ser.plot()
plt.title('Portfolio Value Over Time')
plt.xlabel('Time step')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.legend()
plt.show()

#MBY asset returns during validation data

MBY = pd.Series(lin_rets_val_df['BMY'], index=lin_rets_val_df.index)

# Plot the portfolio value over time
plt.figure(figsize=(14, 7))
MBY.plot()
plt.title('BMY returns during validation data')
plt.xlabel('Time step')
plt.ylabel('BMY returns')
plt.grid(True)
plt.legend()
plt.show()