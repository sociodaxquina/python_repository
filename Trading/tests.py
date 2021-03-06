
#1 Packages***********************************************************************
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import time
    import sklearn

    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

#2 Import data********************************************************************
    path1="C:\\Users\\pedro\\Desktop\\Trading\\Workspace\\Datasets\\merged_Ing_Esp_Pt_All_v2_TRAIN_2009_2015.xlsx"
    #df=pd.read_csv(path1,sep=";")
    df=pd.read_excel(path1)
    
    #testing
    #print (df.head(10))
    #print (df.dtypes)
    #print (df.columns)
    #print(df.loc[df['id'] >2285030])
    #print(df.describe())

#3 Exploratory Analysis - Continuous Variables************************************

    i=12
    df_col_i=df.iloc[:,i]
    var_type=df.dtypes[i]
    var_type=df.dtypes[i]
    Count_Rows=df.shape[0]
    Count_Miss_Val_i=df_col_i.isnull().sum()
    Max_col_i=df_col_i.max()
    Min_col_i=df_col_i.min() 
    
    def data_exp_intervalar(i):
        Mean_col_i=df_col_i.mean()
        Stdev_col_i=df_col_i.std()
        Third_qt_col_i=df_col_i.quantile(.75)
        Median_col_i=df_col_i.median()
        First_qt_col_i=df_col_i.quantile(0.25)
        Max_col_i=df_col_i.max()
        Min_col_i=df_col_i.min()    
        print('\x1b[0;30;46m' + "***EXPLORATORY ANALYSIS (UNIVARIATE)***" + '\x1b[0m')
        print("  Variable: " + df.columns[i])
        print("  Type:" + str(var_type))
        print("  Col index: " + str(i))
        print("  Row Count: " + str(Count_Rows))
        print("  Missing Values Count: " + str(Count_Miss_Val_i) + " (" + str(round(Count_Miss_Val_i/Count_Rows*100,2)) + "%)")
        print("  > Mean: " + str(round(Mean_col_i,3)))  
        print("  > Std Dev: " + str(round(Stdev_col_i,3))) 
        print("  > Max: " + str(round(Max_col_i,3)))
        print("  > 3rd Qt: " + str(round(Third_qt_col_i,3)))
        print("  > Median: " + str(round(Median_col_i,3)))  
        print("  > 1st Qt: " + str(round(First_qt_col_i,3)))
        print("  > Min: " + str(round(Min_col_i,3)))
        #plot
        N=25 #Define numbe of bins N
        plt.style.use("ggplot")
        df_plot=df.iloc[:,i]    
        plt.hist(df_plot, alpha=0.5, color="blue", normed=False, bins=N, cumulative=False, align="mid")
        plt.grid(True)
        #text_i=("  Variable: " + df.columns[i] + "\n" 
        #      + "  Type:" + str(var_type) + "\n" 
        #      + "  Col index: " + str(i) + "\n" 
        #      + "  Row Count: " + str(Count_Rows) + "\n"
        #      + "  Missing Values Count: " + str(Count_Miss_Val_i) + " (" + str(round(Count_Miss_Val_i/Count_Rows*100,2)) + "%)" + "\n" 
        #      + "  > Mean: " + str(round(Mean_col_i,3)) + "\n"
        #      + "  > Std Dev: " + str(round(Stdev_col_i,3)) + "\n"
        #      + "  > Max: " + str(round(Max_col_i,3)) + "\n"
        #      + "  > 3rd Qt: " + str(round(Third_qt_col_i,3)) + "\n"
        #      + "  > Median: " + str(round(Median_col_i,3)) + "\n"
        #      + "  > 1st Qt: " + str(round(First_qt_col_i,3)) + "\n"
        #      + "  > Min: " + str(round(Min_col_i,3))
        #      )
        #plt.text(Max_col_i,0, text_i, withdash=True,bbox=dict(facecolor='white', alpha=0.5))
        plt.title(df.columns[i])
        plt.xlabel("x")
        plt.ylabel("Frequency")
        plt.show
        
    def data_exp_categorica(i):
        Cat_i=df_col_i.names.unique()
        print('\x1b[0;30;46m' + "***EXPLORATORY ANALYSIS (UNIVARIATE)***" + '\x1b[0m')
        print("  Variable: " + df.columns[i])
        print("  Type:" + str(var_type))
        print("  Col index: " + str(i))
        print("  Row Count: " + str(Count_Rows))
        print("  Missing Values Count: " + str(Count_Miss_Val_i) + " (" + str(round(Count_Miss_Val_i/Count_Rows*100,2)) + "%)")
        print("Category Names: " + str(Cat_i))
    
    if df.dtypes[i]=="int64":
        data_exp_intervalar(i)
        #print("ok")
    else:
        print(df_col_i.head(5))
        print(var_type)
        print("WARNING! not and interval variable")
        data_exp_categorica(i)

#4 Filters**************************************************************************************

#5 Variable Transformation**********************************************************************
    
    # id e target
    df_transf=df.loc[:,['id','depvar1']]
  
    df_transf["T3_1_tot_hteam_pos"]=df["var3_01"]/(df["var3_01"]+df["var3_07"])
    df_transf["T8_7_tot_0_15"]=(df["var8_25"]+df["var8_38"])/(df["var8_25"]+df["var8_38"]+df["var8_37"]+df["var8_26"]+1)
    df_transf["T8_12_tot_76_90"]=(df["var8_35"]+df["var8_48"])/(df["var8_35"]+df["var8_48"]+df["var8_47"]+df["var8_36"]+1)
    print(df_transf.head(10))
    print(df_transf.describe())    
    

    df_Varsel=df.loc[:,['id','depvar1']]
    df_Varsel['var3_05']=df['var3_05']
    df_Varsel['var3_35']=df['var3_35']
    df_Varsel['var4_09']=df['var4_09']
    df_Varsel['var5_04']=df['var5_04']
    df_Varsel['var5_09']=df['var5_09']
    df_Varsel['var8_36']=df['var8_36']
    df_Varsel["T3_1_tot_hteam_pos"]=df["var3_01"]/(df["var3_01"]+df["var3_07"])
    df_Varsel["T8_7_tot_0_15"]=(df["var8_25"]+df["var8_38"])/(df["var8_25"]+df["var8_38"]+df["var8_37"]+df["var8_26"]+1)
    df_Varsel["T8_12_tot_76_90"]=(df["var8_35"]+df["var8_48"])/(df["var8_35"]+df["var8_48"]+df["var8_47"]+df["var8_36"]+1)
    #print(df_Varsel.head(20))
    #print(df_Varsel.describe())    

#6 Variable Normalization/Standardization*******************************************************
    
    #scale1: Min-Max
    from sklearn.preprocessing import MinMaxScaler
    df_Varsel_array1=df_Varsel.values
    scaler = MinMaxScaler()
    df_Varsel_array_scaled1 = scaler.fit_transform(df_Varsel_array1)
    df_Varsel_scaled1=pd.DataFrame(df_Varsel_array_scaled1,columns = df_Varsel.columns)
    #print(df_Varsel_scaled1.head(10))    
    
     #scale2: z-score (zero mean and unit variance)
     #xxxx
    #print(df_Varsel_scaled2.head(10)) 

#7 Data Partition*******************************************************************************

    #choose the scale
    df_Varsel=df_Varsel_scaled1  

    #10-Fold CV or holdout method
        
    
    #holdout method (70% train, 30% validation)
    df_Varsel_data=df_Varsel.loc[:,df_Varsel.columns!='depvar1']
    df_Varsel_target=df_Varsel.loc[:,'depvar1']
    #print(df_Varsel_data.head(10))
    #print(df_Varsel_target.head(10))
    from sklearn.cross_validation import train_test_split
    data_train_set, data_test_set, target_train_set, target_test_set=train_test_split(df_Varsel_data,
                                                                                      df_Varsel_target,
                                                                                      test_size=0.3)
    
    #train_set
     #data_train_set.shape, target_train_set.shape
     #print(data_train_set.head(10))
     #print(target_train_set.head(10))
    #test_set
     #data_test_set.shape, target_test_set.shape
     #print(data_test_set.head(10))
     #print(target_test_set.head(10))

#8 Modeling*************************************************************************************
    
    #model1 - Random Forests
    from sklearn.ensemble import RandomForestClassifier
    model1 = RandomForestClassifier(n_estimators=10000)
    model1.fit(data_train_set,target_train_set)
    model1.score(data_test_set,target_test_set)
    model1_desc='R Forests n=1000'
    
    #model2 - Gaussian Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    model2 = GaussianNB()
    model2.fit(data_train_set,target_train_set)
    model2.score(data_test_set,target_test_set)
    model2_desc='Gaussian Naive Bayes'
    
    #model3 - KNN (k=5, weight=uniform)
    k=5
    w='uniform'
    from sklearn.neighbors import KNeighborsClassifier
    model3 = KNeighborsClassifier(n_neighbors=k, weights=w)
    model3.fit(data_train_set,target_train_set)
    model3.score(data_test_set,target_test_set)
    model3_desc='KNN (k=5, weight=uniform)'
    
    #model4 -svm kernel=poly
    from sklearn import svm
    C=1.0
    kernel='poly'
    p=True
    model4 = svm.SVC(C=C,kernel=kernel,probability=p)
    model4.fit(data_train_set,target_train_set)
    model4.score(data_test_set,target_test_set)
    model4_desc='svm kernel=poly'

    #model5 -svm kernel=rbf
    from sklearn import svm
    C=1.0
    kernel='rbf'
    p=True
    model5 = svm.SVC(C=C,kernel=kernel,probability=p)
    model5.fit(data_train_set,target_train_set)
    model5.score(data_test_set,target_test_set)
    model5_desc='svm kernel=rbf'
    
    #model6 - svm kernel=linear
    from sklearn import svm
    C=1.0
    kernel='linear'
    p=True
    model6 = svm.SVC(C=C,kernel=kernel,probability=p)
    model6.fit(data_train_set,target_train_set)
    model6.score(data_test_set,target_test_set)
    model6_desc='svm kernel=linear'

    #model7 - Adaboost@svm (kernel=linear)
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    C=1.0
    kernel='linear'
    p=True
    model7 = AdaBoostClassifier(svm.SVC(C=C,kernel=kernel,probability=p),
                         algorithm="SAMME",
                         n_estimators=200)
    model7.fit(data_train_set,target_train_set)
    model7.score(data_test_set,target_test_set)
    model7_desc='Adaboost@svm (kernel=linear)'

#9 Model Assessment*****************************************************************************

    #Confusion Matrix   
    from sklearn.metrics import confusion_matrix
    target_true=target_test_set
    target_predicted=model1.predict(data_test_set) 
    tn, fp, fn, tp = confusion_matrix(target_true, target_predicted).ravel()
    (tn, fp, fn, tp)
    model1_accuracy=(tp+tn)/(tp+tn+fp+fn)
    model1_precision=(tp)/(tp+fp)
    model1_tpr=(tp)/(tp+fn)
    model1_fpr=(fp)/(fp+tn)
    print("CONFUSION MATRIX: model2 - Gaussian Naive Bayes",
          " 1) accuracy: " + str(round(model1_accuracy,4)),
          " 2) precision: " + str(round(model1_precision,4)),
          " 3) true positive rate: " + str(round(model1_tpr,4)),
          " 4) false positive rate: " + str(round(model1_fpr,4)),
          sep="\n")
    
    # Compute ROC curve and ROC area for class=1
    from sklearn.metrics import roc_auc_score
    from sklearn import metrics
    x_test=data_test_set.values
    y_true=target_test_set.values
    
    #y_score=model2.predict_proba(x_test)
    #fpr, tpr, threshold = metrics.roc_curve(y_true, y_score[:,1]) #class=1->index=1, class=0->index=0
    #roc_auc=metrics.auc(fpr, tpr) #tb poderia ser (é indiferente) roc_auc_score(y_true, y_score[:,1])
    
    
    #model1
    y_score1=model1.predict_proba(x_test)
    fpr1, tpr1, threshold1 = metrics.roc_curve(y_true, y_score1[:,1])
    roc_auc1=roc_auc_score(y_true, y_score1[:,1])    
    #model2
    y_score2=model2.predict_proba(x_test)
    fpr2, tpr2, threshold2 = metrics.roc_curve(y_true, y_score2[:,1])
    roc_auc2=roc_auc_score(y_true, y_score2[:,1])   
    #model3
    y_score3=model2.predict_proba(x_test)
    fpr3, tpr3, threshold3 = metrics.roc_curve(y_true, y_score3[:,1])
    roc_auc3=roc_auc_score(y_true, y_score3[:,1])   
    #model4
    y_score4=model4.predict_proba(x_test)
    fpr4, tpr4, threshold4 = metrics.roc_curve(y_true, y_score4[:,1])
    roc_auc4=roc_auc_score(y_true, y_score4[:,1])       
    #model5
    y_score5=model5.predict_proba(x_test)
    fpr5, tpr5, threshold5 = metrics.roc_curve(y_true, y_score5[:,1])
    roc_auc5=roc_auc_score(y_true, y_score5[:,1])   
    #model6  
    y_score6=model6.predict_proba(x_test)
    fpr6, tpr6, threshold6 = metrics.roc_curve(y_true, y_score6[:,1])
    roc_auc6=roc_auc_score(y_true, y_score6[:,1])   
    #model7  
    y_score7=model7.predict_proba(x_test)
    fpr7, tpr7, threshold7 = metrics.roc_curve(y_true, y_score7[:,1])
    roc_auc7=roc_auc_score(y_true, y_score7[:,1])       

    #report models   
    res_df = pd.DataFrame(columns=('.Models', 'Name','AUC' ))
    for i in range (1, 8):
        res_df.loc[i,:]=['model%d' % i,eval('model{0}_desc'.format(i)),eval('roc_auc{0}'.format(i).format(i))]
    res_df =res_df.sort_values('AUC', ascending=0)
    print(res_df)   
      
    #plot ROC curve
    plt.figure()
    lw = 1
    tick_min=0
    tick_max=1
    tick_step=0.1
    plt.plot(fpr1, tpr1, color='red',
             lw=lw, label='model1 - R Forests (area = %0.4f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='pink',
             lw=lw, label='model2 - N Bayes (area = %0.4f)' % roc_auc2)
    plt.plot(fpr3, tpr3, color='blue',
             lw=lw, label='model3 - KNN (k=5,w=unif) (area = %0.4f)' % roc_auc3)
    plt.plot(fpr4, tpr4, color='black',
             lw=lw, label='model4 - SVM (kernel=poly) (area = %0.4f)' % roc_auc4)
    plt.plot(fpr5, tpr5, color='green',
             lw=lw, label='model5 - SVM (kernel=rbf) (area = %0.4f)' % roc_auc5)
    plt.plot(fpr6, tpr6, color='brown',
             lw=lw, label='model6 - SVM (kernel=linear) (area = %0.4f)' % roc_auc6)
    plt.plot(fpr7, tpr7, color='black',
             lw=lw, label='model7 - Adaboost@svm (kernel=linear) (area = %0.4f)' % roc_auc7)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate',fontsize=10)
    plt.ylabel('True Positive Rate',fontsize=10)
    plt.title('ROC Curve')
    plt.legend(loc="lower right",fontsize=10)
    plt.xticks(np.arange(tick_min, tick_max+0.1, tick_step))
    plt.yticks(np.arange(tick_min, tick_max+0.1, tick_step))
    plt.grid(b=True, which='major', color='grey', linestyle='-')
    plt.show()