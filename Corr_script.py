#import os (operating system) library
#allows us to interact with the filder of assets and load up its contents
import os
#pandas library is a strong library, lets us handle these csv assets files into a dataframe
import pandas as pd
#numpy is another super useful library allowing us for some slick data structure manipulation etc
import numpy as np

#setup an empty dictionary which will be used to store a dataframe with the name of the asset it represents
dataframes = {}

#we want to loop over all files in the Assets folder, and importing into the dictionary the data of each asset and its name as the key
for file in os.listdir('Assets'):
    #if the file is a .csv file we extract its info
    if file.endswith('.csv'):
        #construct the path name of the file
        path = os.path.join('Assets', file)
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
lin_rets = {}

#we now loop over each key in the dataframes dict
for key, value in dataframes.items():
    adj_close = value.iloc[:,5]
    #simplest way to simulate the lin rets formula is to use shift()
    #this is a functions which allows us to shift the data in a dataframe by the number specified
    linr = (adj_close - adj_close.shift(1)) / adj_close.shift(1)
    #this naturally leaves a NaN value at index 0 so we remove it
    linr = linr.dropna()
    #finally construct our lin_rets dictionary
    lin_rets[key] = linr


#collect all asset lin rets into a list first otherwise the following numpy array is not 2D (matrix)
rets = list(lin_rets.values())
#also store the keys in a list to use them later
keys = list(lin_rets.keys())
#now turn the list into numpy array (for easier use)
rets_array = np.array(rets)


#numpy actually provides us with a Pearson correlation coefficient function, to which we can pass our returns matrix
#so we can produce a matrix of correlation coefficients 
corr_coeffs = np.corrcoef(rets_array)

#we want to print it out, having the keys as the rows and columns so we turn it back into a dataframe which makes it simpler for us to do so 
corr_coeffs_frame = pd.DataFrame(corr_coeffs, index = keys, columns=keys)
#50 assets are way too many to be able to show the whole matrix sadly


#Now the approach we want to take is that we want the most desirable assets are the ones with the most negative correlation coefficients.
#One way to determine them is to sum up their correlation coefficients and choose the 10 assets which have the most negative (or smallest) sum of coefficients

#first sum up the coefficients of each asset (along each rows (axis = 1))
coeff_sums = np.sum(corr_coeffs_frame, axis=1)

#now numpy provides the argsort() function which instead of actually sorting an array it returns the indexes of the the array had it been sorted
#with larger databases it can be crucial so that the app doesnt take too long to run
smallest_coeff_idx = np.argsort(coeff_sums)[:10]

#sorted from largest to smallest so we are interested in the last 10 indeces
#list comprehension here for shorter, more slick syntax
final_keys = [keys[i] for i in smallest_coeff_idx]
print(final_keys)

#then we get the subset of the correlation matrix using the keys as indexes (loc)
final_corr_matrix = corr_coeffs_frame.loc[final_keys, final_keys]
print(final_corr_matrix)