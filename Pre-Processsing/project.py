import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import mode
df=pd.read_csv(r"C:\Users\91785\Documents\GitHub\Sales_forcasstng\Pre-Processsing\sales_prediction.csv")
df.info()
df.describe()
df.isnull().sum()
x=df.drop(columns=['Item_Outlet_Sales'])
y=df['Item_Outlet_Sales']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)

X_train.shape,X_test.shape
X_train.head(5)
X_train_N=X_train.copy()
X_train_N.info()
X_train_N.isnull().sum()
int_data=X_train_N.select_dtypes(exclude=['object'])
int_data
int_data.describe()
object_data=X_train.select_dtypes(include=['object'])
object_data.describe()
object_data.isnull().sum()
X_train_N[['Item_Identifier','Item_Weight']].drop_duplicates().sort_values(by=['Item_Identifier'])
Item_ID_Weight=X_train.pivot_table(values='Item_Weight',index='Item_Identifier',aggfunc='median').reset_index()
Item_ID_Weight_map=dict(zip(Item_ID_Weight['Item_Identifier'],Item_ID_Weight['Item_Weight']))
Item_ID_Weight_map.items()
Item_type_Weight=X_train_N.pivot_table(values='Item_Weight',index='Item_Type',aggfunc='median').reset_index()

Item_type_Weight_map=dict(zip(Item_type_Weight['Item_Type'],Item_type_Weight['Item_Weight']))
Item_type_Weight_map.items()
def impute_item_weight(data_frame):
    data_frame.loc[:,'Item_Weight']= data_frame.loc[:,'Item_Weight'].fillna(data_frame.loc[:,'Item_Identifier'].map(Item_ID_Weight_map))
    data_frame.loc[:,'Item_Weight']= data_frame.loc[:,'Item_Weight'].fillna(data_frame.loc[:,'Item_Type'].map(Item_ID_Weight_map))
    return data_frame
X_train_N=impute_item_weight(X_train_N)
X_train_N.isnull().sum()
X_train_N.groupby(by=['Outlet_Type','Outlet_Size']).size()
df.drop(['Outlet_Size'], axis=1)
df.corr()
df.hist()
x=df['Item_MRP']
y=df['Item_Outlet_Sales']
plt.scatter(x,y)
fig, ax = plt.subplots(1,2, figsize=(12,5))

sns.histplot(data=X_train_N,x='Item_Weight',ax=ax[0]);
sns.boxplot(data=X_train_N,y='Item_Weight',ax=ax[1]);
def vis_feature(data_frame,col_name):
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.histplot(data=data_frame,x=col_name,ax=ax[0]);
    sns.boxplot(data=data_frame,y=col_name,ax=ax[1]);
vis_feature(X_train_N,'Item_Weight')
vis_feature(X_train_N,'Item_Visibility')
vis_feature(X_train_N,'Item_MRP') 
vis_feature(X_train_N,'Outlet_Establishment_Year') 
sns.countplot(data=X_train_N,x='Outlet_Establishment_Year')
