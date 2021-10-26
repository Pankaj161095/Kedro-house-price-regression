import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



def independent_dependent_variable(train: pd.DataFrame,test: pd.DataFrame) -> pd.DataFrame:
    train_data = train.drop(['Id','SalePrice'], axis=1)
    test_data = test.drop(['Id'], axis=1)
    dependent_var = train['SalePrice']
    return train_data, test_data, dependent_var

def concat_csv(train_data: pd.DataFrame,test_data: pd.DataFrame) -> pd.DataFrame:
    all_data = pd.concat([train_data, test_data], axis=0)
    #print(all_data)
    return all_data

def preprocess_all_data(all_data: pd.DataFrame) -> pd.DataFrame:
    #all_data = pd.DataFrame(all_data)
    #print(all_data)
    all_data.drop(['Alley','FireplaceQu', 'PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
    #print(all_data)
    #print(all_data)
    #print(all_data.dtype)
    cat_var = ['BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','BsmtFinType2','GarageType', 'GarageFinish','GarageQual', 'GarageCond']
    
    #numerical_features = [f for f in all_data.columns if all_data[f].dtypes!='O']
    #categorical_feature = [f for f in all_data.columns if all_data[f].dtypes=='O']
    #discrete_feature = [f for f in all_data.columns if len(all_data[f].unique())<25]
    #continuous_feature = [f for f in all_data.columns if all_data[f].dtypes!='O' and f not in year_feature]
    
    #for column in continuous_feature:
        #median = all_data[column].median()
    
        #all_data[column].fillna(median,inplace=True)
        
    #for column in categorical_feature:
        #all_data[column].fillna('Missing',inplace=True)
        
    for column in cat_var:
        all_data[column].fillna('None', inplace=True)
        
    # Impute using the column mode
    cat_var1 = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']
    for column in cat_var1:
        all_data[column].fillna(all_data[column].mode()[0], inplace=True)
        
    imputer = SimpleImputer()
    num_var = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt','GarageCars', 'GarageArea']
    for column in num_var:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(all_data[[column]])
        all_data[[column]] = imputer.transform(all_data[[column]])
      
    return all_data
def label_encoding(preprocess_all_data: pd.DataFrame) -> pd.DataFrame:
    categorical_feature = [f for f in preprocess_all_data.columns if preprocess_all_data[f].dtypes=='O']
    for i in categorical_feature:
        le = LabelEncoder()
        preprocess_all_data[i] = le.fit_transform(preprocess_all_data[i])
        
    return preprocess_all_data
        
def scaling(preprocess_all_data: pd.DataFrame) -> pd.DataFrame:
    
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(preprocess_all_data), index=preprocess_all_data.index, columns=preprocess_all_data.columns)
    scaled_data = scaled_data.iloc[0:1460, :]
    return scaled_data
    
def target_variable_skew(dependent_var:pd.DataFrame) -> pd.DataFrame:
    dependent_var = np.log(dependent_var)
    return dependent_var
    