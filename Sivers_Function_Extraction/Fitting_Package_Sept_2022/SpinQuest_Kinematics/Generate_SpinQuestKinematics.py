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

    
def GenSpinQuestKinematics(x1min,x1max,x2min,x2max,QTmin,QTmax,QMmin,QMmax,Npoints):
    xdep = ["x2" for i in range(0,Npoints)]
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

SpinQuestKins_QT_2_QM_4=GenSpinQuestKinematics(0.3,0.9,0.5,0.5,2.0,2.0,4.0,4.0,40)
SpinQuestKins_QT_2_QM_4.to_csv("SpinQuest_Kinematics_xF_dependence_QT_2_QM_4.csv")

SpinQuestKins_QT_2_QM_6=GenSpinQuestKinematics(0.3,0.9,0.5,0.5,2.0,2.0,6.0,6.0,40)
SpinQuestKins_QT_2_QM_6.to_csv("SpinQuest_Kinematics_xF_dependence_QT_2_QM_6.csv")

SpinQuestKins_QT_4_QM_4=GenSpinQuestKinematics(0.3,0.9,0.5,0.5,4.0,4.0,4.0,4.0,40)
SpinQuestKins_QT_4_QM_4.to_csv("SpinQuest_Kinematics_xF_dependence_QT_4_QM_4.csv")

SpinQuestKins_QT_4_QM_6=GenSpinQuestKinematics(0.3,0.9,0.5,0.5,4.0,4.0,6.0,6.0,40)
SpinQuestKins_QT_4_QM_6.to_csv("SpinQuest_Kinematics_xF_dependence_QT_4_QM_6.csv")

SpinQuestKins_QT_25_QM_4=GenSpinQuestKinematics(0.1,0.4,0.5,0.5,2.5,2.5,4.0,4.0,40)
SpinQuestKins_QT_25_QM_4.to_csv("SpinQuest_Kinematics_x2_dependence_QT_2_5_QM_4.csv")

SpinQuestKins_QT_1_QM_4=GenSpinQuestKinematics(0.1,0.4,0.5,0.5,1,1,4.0,4.0,40)
SpinQuestKins_QT_1_QM_4.to_csv("SpinQuest_Kinematics_xt_dependence_QT_1_QM_4.csv")

SpinQuestKins_QT_1_QM_4_xvar=GenSpinQuestKinematics(0.1,0.4,0.5,0.6,1,1,4.0,4.0,40)
SpinQuestKins_QT_1_QM_4_xvar.to_csv("SQ_xt_dependence_QT_1_QM_4.csv")

SpinQuestKins_QT_1_QM_4_xvar=GenSpinQuestKinematics(0.1,0.4,0.5,0.6,0.5,0.5,4.0,4.0,40)
SpinQuestKins_QT_1_QM_4_xvar.to_csv("SQ_xt_dependence_QT_05_QM_4.csv")
