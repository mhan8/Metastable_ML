import pandas as pd
import numpy as np
from sklearn import preprocessing
from numpy import mean
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("../../../Database/strain/Database.csv")
df = df.dropna()
X = df.drop(["Structure #", "Bandgap", "Enthalpy", "Total Energy","Volume",\
                   "Length4","Length5","Length6",\
                       "logi","logo","log1","log2","log3","loga","logb","logc",], axis=1)
y = df["Bandgap"]
df.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, random_state=42, shuffle=True)


lr_regressor = make_pipeline(preprocessing.StandardScaler(),
                              LinearRegression())
# Print the parameters of the random forest model
print("Linear Regressor: ", lr_regressor)
print(lr_regressor.get_params())

y_pred = cross_val_predict(lr_regressor, X, y, cv=cv)
scores_lr = cross_val_score(lr_regressor,
                                X, y, cv=cv, scoring='r2',
                                n_jobs=-1, error_score='raise')
# Print scores for each fold and round to 4 decimal places
print(np.round(scores_lr, 4))
print("LR: %0.4f accuracy with a standard deviation of %0.4f" % (scores_lr.mean(), scores_lr.std()))

m=np.mean(y)
m2=np.mean(y_pred)

mae=np.mean(abs(y_pred-y))
r2=1-(sum((y-y_pred)**2)/sum((y-m)**2))
mad1=np.mean(abs(y_pred-m2)) #predicted MAD
mad2=np.mean(abs(y-m)) #real MAD
ratio=mad2/mae

# Plot prediction accuracy
plt.figure()
plt.scatter(y, y_pred, alpha=0.75)
plt.plot(y, y, color='k')
font = {'family' : 'normal',
        'size'   : 20}
txt = "$R^2$={r:.4f}\nMAD/MAE={ratio:.2f}"
plt.annotate(txt.format(r = scores_lr.mean(),ratio=ratio), xy=(0.05, 0.8), xycoords='axes fraction',fontsize=16)
plt.xlabel('Calculated band gap (eV)',fontsize=16)
plt.ylabel('Predicted band gap (eV)',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()