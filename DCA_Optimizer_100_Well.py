import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.optimize as opt


start_time = time.time()


# --------------------------- FUNCTIONS -----------------------------------

# q1 function
def q1_func(q0, D0, t1):
    q1 = q0 * np.exp(D0 * t1)
    return q1


# Exponential decline function
def exp_func(q0, D0, t):
    q_model = q0 * np.exp(D0 * t)
    return q_model


# Hyperbolic decline function
def hyperb_func(q0, b, D0, D1, t1, t):
    q1 = q1_func(q0, D0, t1)
    q_model = q1 / ((1 + b * D1 * (t - t1))**(1 / b))
    return q_model


# Decline function logic (switch-condition)
def decl_func(q0, D0, D1, b, t1, t):
    if t < t1:
        q_model = exp_func(q0, D0, t)
    else:
        q_model = hyperb_func(q0, b, D0, D1, t1, t)
    return q_model


# Objective function that needs to be minimizd (Mean-squared-error)
def obj_func(D1, b):
    global well_data, mse
    
    # Compute the time function and rate function for the model
    for i_count in range(len(well_data)):
        # Do not attempt to fit a model with less than 6 points
        if len(well_data) < 6:
            pass
        # Fit the model if you have greater than 6 points of data
        else:
            well_data['q_model'] = well_data['delta_time'].apply(lambda x: decl_func(q_min, D0, D1, b, t_max, x))
            well_data['q_error'] = (well_data['q_model'] - well_data['oil'])**2
        
    # Model prediction evaluation criteria (Mean-squared-error)
    mse = well_data['q_error'].sum() / len(well_data)
    return mse


# Defining an eggholder function for the use in the optimization function
def eggholder(Opt_var):
    return obj_func(Opt_var[0], Opt_var[1])

# -------------------------------------------------------------------------
    

# ------------------- PROGRAM EXECUTION -----------------------------------
    
# Import and read well data with Pandas
df = pd.read_csv('Project_data.csv', parse_dates= [['year','month']])
well_API = pd.unique(df['API'])
df_group = df.groupby('API')
mse_output_array = np.empty([len(well_API)])

# Outside loop that iterates based on each well API number
for jdx in range(len(well_API)):
    well_data = df_group.get_group(well_API[jdx]).reset_index()
    
    # Remove outliers from data using 1.5*IQR Rule from data science    
    iqr = well_data.oil.quantile(0.75) - well_data.oil.quantile(0.25)    
    for i_count in range(len(well_data)):
        if (well_data.loc[i_count, 'oil'] < (well_data.oil.quantile(0.25) - 1.5 * iqr)) or (well_data.loc[i_count, 'oil'] > (well_data.oil.quantile(0.75) + 1.5 * iqr)):
            well_data = well_data.drop(i_count)
        else:
            pass
    
    # Need to reset indices since the entire row for an outlier is removed
    well_data = well_data.reset_index(drop= True)
    
    # Find the maximum and minimum flow rates
    q_max = well_data['oil'].max()
    q_min = well_data.loc[0, 'oil']
    
    # Compute the delta time column for plotting
    for i_count in range(len(well_data)):
        well_data.loc[i_count, 'delta_time'] = (well_data.loc[i_count, 'year_month'].year - well_data.loc[0, 'year_month'].year) * 12 + (well_data.loc[i_count, 'year_month'].month - well_data.loc[0, 'year_month'].month)
    
    # Identify the delta_time where maximum flowrate is
    t_max = well_data.loc[well_data['oil'].idxmax(), 'delta_time']
    
    # Compute the first splice increasing slope for exponential function
    if t_max == 0:
        D0 = 0
    elif q_min <= 0:
        q_min = 0.1
        D0 = (1 / t_max) * np.log(q_max / q_min)
    else:
        D0 = (1 / t_max) * np.log(q_max / q_min)
    
    
    if len(well_data) < 6:
        # Plot Results and no optimization
        plt.figure(figsize= (6,6))
        ax = well_data.plot.scatter(x= 'delta_time', y= 'oil', color= 'r', label= 'Production Data', ax= plt.gca())
        ax.set_title('Well #'+ str(jdx) +' Decline Curve Analysis (DCA)')
        ax.set_xlabel('Time in Months')
        ax.set_ylabel('Production Rate')
        plt.savefig('Plot of well #' +str(jdx) + '.png')
        mse_output_array[jdx] = float('NaN')        
    else:
        # Establish the bounds for each of the optimization variables
        bounds = [(0.00001, 100), (0, 2)]
        
        # Perform the optimization function
        results = opt.differential_evolution(eggholder, bounds)
        D1 = results.x[0]
        b = results.x[1]
    
        # Report the mean-squared-error to an array
        mse_output_array[jdx] = mse
        
        # Allocate memory and compute the model solution after optimization
        t_model_plot = np.arange(0, well_data['delta_time'].max()+0.20, 0.20)
        q_model_plot = np.empty([len(t_model_plot)])
        for idx in range(len(t_model_plot)):
            q_model_plot[idx] = decl_func(q_min, D0, D1, b, t_max, t_model_plot[idx])
        
        # Plot results of modeled data after optimizing parameters
        plt.figure(figsize= (6,6))
        ax = plt.plot(t_model_plot, q_model_plot, label= 'Decline Model')
        well_data.plot.scatter(x= 'delta_time', y= 'oil', color= 'r', label= 'Production Data', ax= plt.gca())
        plt.title('Well #'+ str(jdx) +' Decline Curve Analysis (DCA)')
        plt.xlabel('Time in Months')
        plt.ylabel('Production Rate')
        plt.legend()
        plt.savefig('Plot of well #' +str(jdx) + '.png')
    
    # Report the Pandas Dataframe to csv for each well
    well_data.to_csv('Well #'+ str(jdx) + ' Dataframe.csv')
    print('Well #' + str(jdx) + ' completed')

# Export MSE results per well and record execution timing
np.savetxt('MSE_Results.csv', mse_output_array, delimiter= ',')
print('My program took', time.time() - start_time, 'seconds to run.')

# -------------------------------------------------------------------------