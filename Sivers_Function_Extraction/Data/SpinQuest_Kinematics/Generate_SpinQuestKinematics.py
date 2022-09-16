import numpy as np
import pandas as pd

def GenSpinQuestKinematics(x1min,x1max,x2min,x2max,QTmin,QTmax,QMmin,QMmax,Npoints):
    xdep = ["xF" for i in range(0,Npoints)]
    x1 = np.array(np.linspace(x1min,x1max,Npoints))
    x2 = np.array(np.linspace(x2min,x2max,Npoints))
    xf = x1 - x2
    QT = np.array(np.linspace(QTmin,QTmax,Npoints))
    QM = np.array(np.linspace(QMmin,QMmax,Npoints))
    data_dictionary = {"Dependence": [], "x1": [], "x2": [], "xF": [], "QT": [], "QM": [], "Siv": [], "tot_err": []}
    data_dictionary["Dependence"] = xdep
    data_dictionary["x1"] = x1
    data_dictionary["x2"] = x2
    data_dictionary["xF"] = xf
    data_dictionary["QT"] = QT
    data_dictionary["QM"] = QM
    data_dictionary["Siv"] = np.array([0 for i in range(0,Npoints)])
    data_dictionary["tot_err"] = np.array([0 for i in range(0,Npoints)])
    return pd.DataFrame(data_dictionary)

SpinQuestKins=GenSpinQuestKinematics(0.3,0.9,0.5,0.5,1.0,3.0,4.0,9.0,100)
SpinQuestKins.to_csv("SpinQuest_Kinematics_xF_dependence.csv")