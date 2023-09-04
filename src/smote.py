from imblearn.over_sampling import SMOTE
import pandas as pd

X_train_rfe=pd.read_csv('C:/Users/KORISNIK/Desktop/za fakultet/ToxicityOfMolecules/X_train_rfe.csv')
X_test_rfe=pd.read_csv('C:/Users/KORISNIK/Desktop/za fakultet/ToxicityOfMolecules/X_test_rfe.csv')
y_train=pd.read_csv('C:/Users/KORISNIK/Desktop/za fakultet/ToxicityOfMolecules/y_train.csv')
y_test=pd.read_csv('C:/Users/KORISNIK/Desktop/za fakultet/ToxicityOfMolecules/y_test.csv')


smote = SMOTE(sampling_strategy='auto', random_state=42)
X_test_smote, y_test_smote = smote.fit_resample(X_test_rfe, y_test)

X_test_smote.to_csv("C:/Users/KORISNIK/Desktop/za fakultet/ToxicityOfMolecules/X_test_smote.csv",index=False)
y_test_smote.to_csv("C:/Users/KORISNIK/Desktop/za fakultet/ToxicityOfMolecules/y_test_smote.csv",index=False)
