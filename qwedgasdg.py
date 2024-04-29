import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.regression.linear_model import OLS
from statsmodels.formula.api import ols


df_copy = pd.read_csv("./U.S._Chronic_Disease_Indicators.csv")

depression = df_copy[df_copy.Question == "Depression among adults"]

depression = depression[["YearStart", "LocationDesc", "DataValue", "LowConfidenceLimit", "HighConfidenceLimit", "StratificationCategory1", "Stratification1"]]

depression['LocationDesc'] = depression['LocationDesc'].astype('category')
depression['Stratification1'] = depression['Stratification1'].astype('category')

formula = "np.log(DataValue / (100 - DataValue)) ~ YearStart + C(LocationDesc) + C(Stratification1)"
model = ols(formula=formula, data=depression).fit_regularized()

#print(model.summary())
new_data = pd.DataFrame({
    'YearStart': [2021], # CHANGE DROPDOWN
    'LocationDesc': pd.Categorical(['California'], categories=depression['LocationDesc'].cat.categories), #CHANGE WITH DROPDOWJS
    'Stratification1': pd.Categorical(['Female'], categories=depression['Stratification1'].cat.categories) #CHANGE WITH DROPDOWN
})

new_data_with_intercept = sm.add_constant(new_data, has_constant='add')

predictions = model.predict(new_data_with_intercept)

probabilities = 1 / (1 + np.exp(-predictions))

print(probabilities)