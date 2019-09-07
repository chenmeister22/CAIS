import pandas as pd 
import numpy as np 
import numpy.random as nr 
import matplotlib.pyplot as plt
import sklearn.model_selection as ms 
import sklearn.metrics as sklm
from sklearn import preprocessing
from sklearn import linear_model
import seaborn as sns
import math 

#Importing packages 

company_data = pd.read_csv(r"C:\Users\morga\Desktop\Machine Learning\Final Project\AdvWorksCusts.csv")#Reading csv file 
# print(company_data.shape)
# print(company_data.CustomerID.unique().shape)
company_data.drop_duplicates(subset="CustomerID", keep="first", inplace=True)#Removing duplicates
#print(company_data.shape)

#Visualization

#print(company_data.columns)
#print(company_data.head())
#print(company_data.dtypes)
#print(company_data.describe())

# def count_unique(company_data,cols):
#     for col in cols: 
#         print("For column " + col)
#         print(company_data[col].value_counts())

# cat_cols = ["CountryRegionName","Education","Occupation","NumberCarsOwned","NumberChildrenAtHome","TotalChildren"]
# count_unique(company_data,cat_cols)

# def plot_bars(company_data, cols):
#     for col in cols: 
#         fig = plt.figure(figsize=(6,6))
#         ax = fig.gca()
#         counts = company_data[col].value_counts()
#         counts.plot.bar(ax=ax)

#         ax.set_title("Number of Customers by" + col)
#         ax.set_xlabel(col)
#         ax.set_ylabel("Number of Customers")
#         plt.show()

# plot_cols = cat_cols
# plot_bars(company_data, plot_cols)

# def plot_histogram(company_data,cols, bins =6):
#     for col in cols: 
#         fig = plt.figure(figsize=(6,6))
#         ax = fig.gca()
#         company_data[col].plot.hist(bins=15)
#         ax.set_title("Histogram of " + col)
#         ax.set_xlabel(col)
#         ax.set_ylabel("Number of Customers")
#         plt.show()
num_cols = ["YearlyIncome"]
# plot_histogram(company_data,num_cols)

# def plot_density_hist(company_data,cols,bins=20, hist=True):
#     for col in cols:
#         sns.set_style("darkgrid")
#         sns.distplot(company_data[col], bins=bins, rug = True, hist = hist) 
#         plt.title("Histogram of " + col)
#         plt.xlabel(col)
#         plt.ylabel("Number of Customers")
#         plt.show()
# plot_density_hist(company_data,num_cols,)

# def plot_scatter(company_data,cols,col_y="YearlyIncome"):
#     for col in cols: 
#         fig = plt.figure(figsize = (6,6))
#         ax = fig.gca()
#         company_data.plot.scatter(x=col,y=col_y,ax=ax)
#         ax.set_title("Scatter Plot") 
#         ax.set_xlabel(col) 
#         ax.set_ylabel(col_y)
#         plt.show()
# scatter_cols = ["NumberCarsOwned","NumberChildrenAtHome","TotalChildren"]
# plot_scatter(company_data,scatter_cols)

# def plot_box(company_data,cols,col_y="YearlyIncome"):
#     for col in cols: 
#         sns.set_style("whitegrid")
#         sns.boxplot(col,col_y,data= company_data)
#         plt.title("Boxplot of " + col)
#         plt.xlabel(col)
#         plt.ylabel(col_y)
#         plt.show()
# plot_box(company_data,scatter_cols)

#company_data[["YearlyIncome"]]= company_data[["YearlyIncome"]].applymap(math.log) #Modifying data

# def plot_violin(company_data,cols,col_x="NumberCarsOwned"):
#     for col in cols:
#         sns.set_style("whitegrid")
#         sns.violinplot(col_x,col,data = company_data)
#         plt.xlabel(col_x)
#         plt.ylabel(col)
#         plt.show()
# plot_violin(company_data, num_cols)


#Linear Regression 

#Will do a classification and regression test 

#Used for both 

# print(company_data["Occupation"].unique())
Features = company_data["Occupation"]
enc = preprocessing.LabelEncoder() #Does alphabetically 
enc.fit(Features)
Features = enc.transform(Features)

ohe = preprocessing.OneHotEncoder()
encoded = ohe.fit(Features.reshape(-1,1))
Features = encoded.transform(Features.reshape(-1,1)).toarray()
Features = np.concatenate([Features,np.array(company_data[["NumberCarsOwned","TotalChildren","YearlyIncome"]])],axis=1)
# print(Features.shape)

#bike_counts = company_data.groupby("BikeBuyer").count()[["CustomerID"]]
# print(bike_counts)
labels = np.array(company_data["BikeBuyer"]) #Classification

nr.seed(8234)
# labels = np.array(company_data["AveMonthSpend"]) #Regression
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size =200)
x_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
x_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

scaler = preprocessing.StandardScaler().fit(x_train[:,5:])
x_train[:,5:] = scaler.transform(x_train[:,5:])
x_test[:,5:] = scaler.transform(x_test[:,5:])

# lin_mod = linear_model.LinearRegression(fit_intercept=False)
# lin_mod.fit(x_train,y_train)

# print(lin_mod.intercept_)
# print(lin_mod.coef_)

logistic_mod = linear_model.LogisticRegression()
logistic_mod.fit(x_train,y_train)

print(logistic_mod.intercept_)
print(logistic_mod.coef_)

probabilities = logistic_mod.predict_proba(x_test)
print(probabilities[:15,:])

def score_model(probabilities,threshold):
    return np.array([1 if x > threshold else 0 for x in probabilities[:,1]])
scores = score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])

# def print_metrics(y_true,y_predicted,n_parameters):
#     r2 = sklm.r2_score(y_true, y_predicted)
#     r2_adj = r2 - (n_parameters-1/y_true.shape[0]-n_parameters)*(1-r2)
#      ## Print the usual metrics and the R^2 values
#     print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
#     print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
#     print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
#     print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
#     print('R^2                    = ' + str(r2))
#     print('Adjusted R^2           = ' + str(r2_adj))
# y_score = lin_mod.predict(x_test) 
# print_metrics(y_test, y_score, 5)    

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

print_metrics(y_test,scores)


# def hist_resids(y_test, y_score):
#     ## first compute vector of residuals. 
#     resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
#     ## now make the residual plots
#     sns.distplot(resids)
#     plt.title('Histogram of residuals')
#     plt.xlabel('Residual value')
#     plt.ylabel('count')
#     plt.show()
    
# hist_resids(y_test, y_score)    

# def resid_plot(y_test, y_score):
#     ## first compute vector of residuals. 
#     resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
#     ## now make the residual plots
#     sns.regplot(y_score, resids, fit_reg=False)
#     plt.title('Residuals vs. predicted values')
#     plt.xlabel('Predicted values')
#     plt.ylabel('Residual')
#     plt.show()

# resid_plot(y_test, y_score) 









