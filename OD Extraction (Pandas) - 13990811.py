# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:08:42 2020

@author: Mahdiyar/Rahmani
Part 2 OD extraction
"""
#%% Importing the required packages
import pandas as pd
import numpy as np                                               # This is useful for obtaining medoid index
import dask.dataframe as dd
#%%Importing Data
stayFinal = pd.read_csv(r'E:\2000weekendstay.csv',parse_dates=['Date_Time'])

cdrPopTract = pd.read_csv(r'E:\usertractpopcdrweekend2000.csv')

tractPop = pd.read_excel(r'E:\Arshad\Thesis\Datasets\Tract population\Pop.xlsx')

#%%Original Code

staygp=stayFinal.groupby(by=['ID'])
# weekend OD matrix
agg_matrix_weekend = np.zeros((23,23))
Hbw_agg_matrix_weekend = np.zeros((23,23)) #ccccccc ok
Hbo_agg_matrix_weekend = np.zeros((23,23))
Nhb_agg_matrix_weekend = np.zeros((23,23))


#weekend production attraction non matrix
attraction_production_weekend = np.zeros((23,2))#checked 
Hbw_attraction_production_weekend = np.zeros((23,2))
Hbo_attraction_production_weekend = np.zeros((23,2))    #okkkkkkkk
Nhb_attraction_production_weekend = np.zeros((23,2))

#Agg weekend P_A matrix
agg_daily_att_pro_mat_weekend = np.zeros((23,23))
Hbw_agg_daily_att_pro_mat_weekend = np.zeros((23,23))
Hbo_agg_daily_att_pro_mat_weekend = np.zeros((23,23))
Nhb_agg_daily_att_pro_mat_weekend = np.zeros((23,23))

# weekday OD matrix
agg_matrix_weekday = np.zeros((23,23))
Hbw_agg_matrix_weekday = np.zeros((23,23))     #OK
Hbo_agg_matrix_weekday = np.zeros((23,23))
Nhb_agg_matrix_weekday = np.zeros((23,23))



#weekday production attraction non matrix
attraction_production = np.zeros((23,2)) #index zero is production index /// index one is attraction
Hbw_attraction_production = np.zeros((23,2))
Hbo_attraction_production = np.zeros((23,2))    #okkkkkkkkk    
Nhb_attraction_production = np.zeros((23,2))                            
attraction_production11 = np.zeros((23,2))


#Agg Weekday P_A matrix
agg_daily_att_pro_mat_weekday=np.zeros((23,23))
Hbw_agg_daily_att_pro_mat_weekday = np.zeros((23,23))
Hbo_agg_daily_att_pro_mat_weekday = np.zeros((23,23))  #okkkkkk   
Nhb_agg_daily_att_pro_mat_weekday = np.zeros((23,23))



counterid=0

for Id in stayFinal.ID.unique():
    
        counterid=counterid+1
        
        #weekend temp matrix OD without trip purpose
        weekendMatrix = np.zeros((23,23))
        Hbw_weekendMatrix = np.zeros((23,23))
        Hbo_weekendMatrix = np.zeros((23,23))              #OD      ccccc ok
        Nhb_weekendMatrix = np.zeros((23,23))              #OD
        
        ####weekend attraction production total(Non matrix)
        daily_att_pro_weekend = np.zeros((23,2))        #without trip purpose
        Hbw_daily_att_pro_weekend = np.zeros((23,2))               
        Hbo_daily_att_pro_weekend = np.zeros((23,2))     #okkkkk
        Nhb_daily_att_pro_weekend = np.zeros((23,2))
        
        ####weekend attraction production
        daily_att_pro_mat_weekend = np.zeros((23,23))    #without trip purpose
        Hbw_daily_att_pro_mat_weekend = np.zeros((23,23))
        Hbo_daily_att_pro_mat_weekend = np.zeros((23,23)) #okkk
        Nhb_daily_att_pro_mat_weekend = np.zeros((23,23))
        
        
        #weekday temp matrix OD without trip purpose
        weekDayMatrix = np.zeros((23,23))
        Hbw_weekDayMatrix = np.zeros((23,23))             #OD        OK
        Hbo_weekDayMatrix = np.zeros((23,23))              #OD
        Nhb_weekDayMatrix = np.zeros((23,23))
        
        ####Weekday attraction production total (Non matrix)
        daily_att_pro = np.zeros((23,2))             #without trip purpose
        Hbw_daily_att_pro = np.zeros((23,2))                
        Hbo_daily_att_pro = np.zeros((23,2))                     #okkkkkkkkkkk
        Nhb_daily_att_pro = np.zeros((23,2))
        
        
        
        ####Weekday attraction production 
        daily_att_pro_mat_weekday = np.zeros((23,23))     #without trip purpose
        Hbw_daily_att_pro_mat_weekday = np.zeros((23,23))
        Hbo_daily_att_pro_mat_weekday = np.zeros((23,23)) #okkkkkkk
        Nhb_daily_att_pro_mat_weekday = np.zeros((23,23))
        
        
        
        
        
        
        
        
        numberOfweekend = 0
        numberOfWeekday = 0
        
        
        stopPointsLabeled = staygp.get_group(Id)
        stopPointsLabeledStorage = []
        
        Home = stopPointsLabeled[stopPointsLabeled.Purpose=='Home'].head(1)
        expansionFactorTract = int(tractPop.iloc[int(Home.manategh)]['POPULATION']/(cdrPopTract.iloc[int(Home.manategh)]))
        
        for time in ((pd.to_datetime(stopPointsLabeled['Date_Time'].dt.normalize().unique()) + pd.Timedelta(3 , unit='H'))):
            
                dataSliced = stopPointsLabeled.loc[(stopPointsLabeled['Date_Time']>=time) & (stopPointsLabeled['Date_Time']<=time+pd.Timedelta(1,unit='D'))]
                
                if (len(dataSliced) != 0) and ((time.day_name() == 'Saturday') or  (time.day_name() == 'Sunday') or  (time.day_name() == 'Monday') or  (time.day_name() == 'Tuesday') or  (time.day_name() == 'Wednesday')):
                    
                    numberOfWeekday = numberOfWeekday + 1
                    
                    if (dataSliced.iloc[0]['Purpose'] == 'Other') or (dataSliced.iloc[0]['Purpose'] == 'Work'):
                        
                        dataSliced = pd.concat([Home,dataSliced])
                        
                    if (dataSliced.iloc[-1]['Purpose'] == 'Other') or (dataSliced.iloc[-1]['Purpose'] == 'Work'):
                        
                            dataSliced = pd.concat([dataSliced,Home])
                            
                    for j1 in range(dataSliced.shape[0]-1):
                        
                        #this  part calculate OD matrix without purpose for weekday    
                        if dataSliced.iloc[j1]['manategh'] != dataSliced.iloc[j1+1]['manategh']:
                            
                            weekDayMatrix[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])] += 1
                            
                            
                            
                            
                            ####### add production attraction for WEEKday
                            if (dataSliced.iloc[j1]['Purpose'] == 'Home') and (dataSliced.iloc[j1+1]['Purpose'] == 'Work'):
                                
                                daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                                Hbw_daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                Hbw_daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                                
                            if (dataSliced.iloc[j1]['Purpose'] == 'Home') and (dataSliced.iloc[j1+1]['Purpose'] == 'Other'):
                                
                                daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                                Hbo_daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                Hbo_daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                                
                                
                            if (dataSliced.iloc[j1]['Purpose']=='Work') and (dataSliced.iloc[j1+1]['Purpose']=='Home'):
                                
                                daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][1] += 1
                                daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][0] += 1
                                
                                Hbw_daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][1] += 1
                                Hbw_daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][0] += 1
                                
                                
                            if (dataSliced.iloc[j1]['Purpose'] == 'Work') and (dataSliced.iloc[j1+1]['Purpose'] == 'Other'):    
                            
                                daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                                Nhb_daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                Nhb_daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                            if (dataSliced.iloc[j1]['Purpose'] == 'Other') and (dataSliced.iloc[j1+1]['Purpose'] == 'Work'):
                                
                                daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                                Nhb_daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                Nhb_daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                            if (dataSliced.iloc[j1]['Purpose'] == 'Other') and (dataSliced.iloc[j1+1]['Purpose'] == 'Home'):
                                
                                daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][1] += 1
                                daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][0] += 1
                                
                                Hbo_daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][1] += 1
                                Hbo_daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][0] += 1
                                
                                
                                
                            if (dataSliced.iloc[j1]['Purpose'] == 'Other') and (dataSliced.iloc[j1+1]['Purpose'] == 'Other'):
                                
                                daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                                Nhb_daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                Nhb_daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                            if (dataSliced.iloc[j1]['Purpose'] == 'Work') and (dataSliced.iloc[j1+1]['Purpose'] == 'Work'):    
                                 
                                daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                                Nhb_daily_att_pro[int(dataSliced.iloc[j1]['manategh'])][0] += 1
                                Nhb_daily_att_pro[int(dataSliced.iloc[j1+1]['manategh'])][1] += 1
                                
                            if (((dataSliced.iloc[j1]['Purpose'] == 'Home') and (dataSliced.iloc[j1+1]['Purpose'] == 'Work')) or ((dataSliced.iloc[j1]['Purpose'] == 'Work') and (dataSliced.iloc[j1+1]['Purpose'] == 'Home'))):
                                    
                                    if (dataSliced.iloc[j1]['Purpose'] == 'Home'):
                                        
                                        i1=int(dataSliced.iloc[j1]['manategh'])
                                        i2=int(dataSliced.iloc[j1+1]['manategh'])
                                        
                                    elif (dataSliced.iloc[j1+1]['Purpose'] == 'Home'):
                                        
                                        i1=int(dataSliced.iloc[j1+1]['manategh'])
                                        i2=int(dataSliced.iloc[j1]['manategh'])
                                        
                                    daily_att_pro_mat_weekday[i1][i2] += 1
                                    Hbw_daily_att_pro_mat_weekday[i1][i2] += 1
                                    Hbw_weekDayMatrix[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])] += 1    #OD for Hbw 
                                
                            if ((dataSliced.iloc[j1]['Purpose'] == 'Home') and (dataSliced.iloc[j1+1]['Purpose'] == 'Other')) or ((dataSliced.iloc[j1]['Purpose'] == 'Other') and (dataSliced.iloc[j1+1]['Purpose'] == 'Home')):
                                     
                                    if (dataSliced.iloc[j1]['Purpose']=='Home'):
                                        
                                        i1=int(dataSliced.iloc[j1]['manategh'])
                                        i2=int(dataSliced.iloc[j1+1]['manategh'])
                                        
                                    elif (dataSliced.iloc[j1+1]['Purpose']=='Home'):
                                        
                                        i1=int(dataSliced.iloc[j1+1]['manategh'])
                                        i2=int(dataSliced.iloc[j1]['manategh'])
                                        
                                    daily_att_pro_mat_weekday[i1][i2] += 1
                                    Hbo_daily_att_pro_mat_weekday[i1][i2] += 1
                                    Hbo_weekDayMatrix[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])] += 1    #OD for Hbo 
                                
                            if ((dataSliced.iloc[j1]['Purpose'] == 'Work') and (dataSliced.iloc[j1+1]['Purpose'] == 'Other')) or ((dataSliced.iloc[j1]['Purpose'] == 'Other') and (dataSliced.iloc[j1+1]['Purpose'] == 'Work')):    
                                    
                                    daily_att_pro_mat_weekday[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])]+=1
                                    Nhb_weekDayMatrix[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])] += 1    #OD for NHB
                                    Nhb_daily_att_pro_mat_weekday[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])] += 1
                                    
                            if ((dataSliced.iloc[j1]['Purpose'] == 'Other') and (dataSliced.iloc[j1+1]['Purpose'] == 'Other')) or ((dataSliced.iloc[j1]['Purpose'] == 'Work') and (dataSliced.iloc[j1+1]['Purpose'] == 'Work')): 
                                    
                                    daily_att_pro_mat_weekday[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])] += 1
                                    Nhb_weekDayMatrix[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])] += 1    #OD for NHB
                                    Nhb_daily_att_pro_mat_weekday[int(dataSliced.iloc[j1]['manategh'])][int(dataSliced.iloc[j1+1]['manategh'])] += 1
                                    
                     ########## weekend PARTS     
                    #add production attraction for work day        
                if (len(dataSliced) != 0) & ((time.day_name() == 'Friday') or (time.day_name() == 'Thursday')):
                        
                        numberOfweekend = numberOfweekend + 1
                        
                        for j in range( dataSliced.shape[0]-1 ):
                            
                            if dataSliced.iloc[j]['manategh'] != dataSliced.iloc[j+1]['manategh']:
                                
                                #OD matrix without purpose for WORK day
                                weekendMatrix[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1
                                
                                
                                if (dataSliced.iloc[j]['Purpose'] == 'Home') and (dataSliced.iloc[j+1]['Purpose'] == 'Work'):
                            
                                    daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                    daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                                    
                                    Hbw_daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                    Hbw_daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                            
                                if (dataSliced.iloc[j]['Purpose'] == 'Home') and (dataSliced.iloc[j+1]['Purpose'] == 'Other'):
                            
                                        daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                        daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                                        
                                        Hbo_daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                        Hbo_daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                                        
                                if (dataSliced.iloc[j]['Purpose'] == 'Work') and (dataSliced.iloc[j+1]['Purpose'] == 'Home'):
                            
                                        daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][1] += 1
                                        daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][0] += 1
                                        
                                        Hbw_daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][1] += 1
                                        Hbw_daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][0] += 1
                            
                                if (dataSliced.iloc[j]['Purpose'] == 'Work') and (dataSliced.iloc[j+1]['Purpose'] == 'Other'):    
                        
                                        daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                        daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                                        
                                        Nhb_daily_att_pro[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                        Nhb_daily_att_pro[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                            
                                if (dataSliced.iloc[j]['Purpose'] == 'Other') and (dataSliced.iloc[j+1]['Purpose'] == 'Work'):
                            
                                        daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                        daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                                        
                                        Nhb_daily_att_pro[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                        Nhb_daily_att_pro[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                            
                                if ( dataSliced.iloc[j]['Purpose'] == 'Other' ) and ( dataSliced.iloc[j+1]['Purpose'] == 'Home' ):
                            
                                        daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][1] += 1
                                        daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][0] += 1
                                        
                                        Hbo_daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][1] += 1
                                        Hbo_daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][0] += 1
                            
                                if (dataSliced.iloc[j]['Purpose'] == 'Other') and (dataSliced.iloc[j+1]['Purpose'] == 'Other'):
                            
                                        daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                        daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                                        
                                        Nhb_daily_att_pro[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                        Nhb_daily_att_pro[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                            
                                if (dataSliced.iloc[j]['Purpose'] == 'Work') and (dataSliced.iloc[j+1]['Purpose'] == 'Work'):    
                             
                                    daily_att_pro_weekend[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                    daily_att_pro_weekend[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                                    
                                    Nhb_daily_att_pro[int(dataSliced.iloc[j]['manategh'])][0] += 1
                                    Nhb_daily_att_pro[int(dataSliced.iloc[j+1]['manategh'])][1] += 1
                                    
                                if ((dataSliced.iloc[j]['Purpose'] == 'Home') and (dataSliced.iloc[j+1]['Purpose'] == 'Work')) or ((dataSliced.iloc[j]['Purpose'] == 'Work') and (dataSliced.iloc[j+1]['Purpose'] == 'Home')):
                                
                                        if (dataSliced.iloc[j]['Purpose'] == 'Home'):
                                            
                                                i1 = int(dataSliced.iloc[j]['manategh'])
                                                i2 = int(dataSliced.iloc[j+1]['manategh'])
                                                
                                        elif (dataSliced.iloc[j+1]['Purpose'] == 'Home'):
                                            
                                                i1 = int(dataSliced.iloc[j+1]['manategh'])
                                                i2 = int(dataSliced.iloc[j]['manategh'])
                                    
                                        daily_att_pro_mat_weekend[i1][i2] += 1                                        #This is for general P_A matrix extraction
                                        Hbw_daily_att_pro_mat_weekend[i1][i2] += 1
                                        Hbw_weekendMatrix[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1   #this is for OD HBW extraction
                                        
                                        
                                if ((dataSliced.iloc[j]['Purpose'] == 'Home') and (dataSliced.iloc[j+1]['Purpose'] == 'Other')) or ((dataSliced.iloc[j]['Purpose'] == 'Other') and (dataSliced.iloc[j+1]['Purpose'] == 'Home')):
                                 
                                        if (dataSliced.iloc[j]['Purpose'] == 'Home'):
                                    
                                                i1 = int(dataSliced.iloc[j]['manategh'])
                                                i2 = int(dataSliced.iloc[j+1]['manategh'])
                                    
                                        elif (dataSliced.iloc[j+1]['Purpose'] == 'Home'):
                                    
                                                i1 = int(dataSliced.iloc[j+1]['manategh'])
                                                i2 = int(dataSliced.iloc[j]['manategh'])
                                    
                                        daily_att_pro_mat_weekend[i1][i2] += 1
                                        Hbo_daily_att_pro_mat_weekend[i1][i2] += 1
                                        Hbo_weekendMatrix[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1   #this is for OD HBO extraction
                                        
                                if ((dataSliced.iloc[j]['Purpose'] == 'Work') and  (dataSliced.iloc[j+1]['Purpose'] == 'Other')) or ((dataSliced.iloc[j]['Purpose'] == 'Other') and (dataSliced.iloc[j+1]['Purpose'] == 'Work')):    
                                
                                            daily_att_pro_mat_weekend[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1
                                            Nhb_daily_att_pro_mat_weekend[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1
                                            Nhb_weekendMatrix[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1
                                            
                          
                                if ((dataSliced.iloc[j]['Purpose'] == 'Other') and (dataSliced.iloc[j+1]['Purpose'] == 'Other')) or ((dataSliced.iloc[j]['Purpose'] == 'Work') and (dataSliced.iloc[j+1]['Purpose'] == 'Work')): 
                                
                                            daily_att_pro_mat_weekend[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1
                                            Nhb_daily_att_pro_mat_weekend[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1
                                            Nhb_weekendMatrix[int(dataSliced.iloc[j]['manategh'])][int(dataSliced.iloc[j+1]['manategh'])] += 1
                               
                                
                            
               
        
        
        #weekday part
        
        #Total attraction production WEEKDAY
        if numberOfWeekday != 0 :
                attraction_production11 += (daily_att_pro/ 5) * expansionFactorTract 
                attraction_production += (daily_att_pro/ numberOfWeekday) * expansionFactorTract 
                Hbw_attraction_production += ( Hbw_daily_att_pro* expansionFactorTract / numberOfWeekday ) 
                Hbo_attraction_production += ( Hbo_daily_att_pro* expansionFactorTract / numberOfWeekday ) 
                Nhb_attraction_production += ( Nhb_daily_att_pro* expansionFactorTract / numberOfWeekday ) 
                
                
                
                
                #OD matrixes for weekdays
                agg_matrix_weekday += (weekDayMatrix / numberOfWeekday) * expansionFactorTract
                Hbo_agg_matrix_weekday += ( Hbo_weekDayMatrix / numberOfWeekday ) * expansionFactorTract
                Hbw_agg_matrix_weekday += (Hbw_weekDayMatrix / numberOfWeekday ) * expansionFactorTract
                Nhb_agg_matrix_weekday += (Nhb_weekDayMatrix / numberOfWeekday ) * expansionFactorTract
                
                
                
                
                #P_A matrices
                agg_daily_att_pro_mat_weekday += (daily_att_pro_mat_weekday / numberOfWeekday) * expansionFactorTract
                Hbw_agg_daily_att_pro_mat_weekday += ( Hbw_daily_att_pro_mat_weekday / numberOfWeekday ) * expansionFactorTract 
                Hbo_agg_daily_att_pro_mat_weekday += ( Hbo_daily_att_pro_mat_weekday / numberOfWeekday ) * expansionFactorTract
                Nhb_agg_daily_att_pro_mat_weekday += ( Nhb_daily_att_pro_mat_weekday / numberOfWeekday ) * expansionFactorTract
        
        
        #workday part
        if numberOfweekend != 0:
            
  
            #Total production attraction WORKDAY
            attraction_production_weekend += ( daily_att_pro_weekend / numberOfweekend ) * expansionFactorTract                       #need to be modified by expansion factor
            Hbw_attraction_production_weekend += ( Hbw_daily_att_pro_weekend / numberOfweekend ) * expansionFactorTract
            Hbo_attraction_production_weekend += ( Hbo_daily_att_pro_weekend / numberOfweekend ) * expansionFactorTract
            Nhb_attraction_production_weekend += ( Nhb_daily_att_pro_weekend / numberOfweekend ) * expansionFactorTract
            
            
            #P_A matrices
            agg_daily_att_pro_mat_weekend += ( daily_att_pro_mat_weekend / numberOfweekend ) * expansionFactorTract                   #need to be modified by expansion factor 
            Hbw_agg_daily_att_pro_mat_weekend += ( Hbw_daily_att_pro_mat_weekend / numberOfweekend ) * expansionFactorTract
            Hbo_agg_daily_att_pro_mat_weekend += ( Hbo_daily_att_pro_mat_weekend / numberOfweekend ) * expansionFactorTract
            Nhb_agg_daily_att_pro_mat_weekend += ( Nhb_daily_att_pro_mat_weekend / numberOfweekend ) * expansionFactorTract
            
            #OD matrixes for workdays
            agg_matrix_weekend +=  ( weekendMatrix / numberOfweekend) * expansionFactorTract
            Hbo_agg_matrix_weekend += ( Hbo_weekendMatrix / numberOfweekend ) * expansionFactorTract
            Hbw_agg_matrix_weekend += (Hbw_weekendMatrix / numberOfweekend ) * expansionFactorTract
            Nhb_agg_matrix_weekend += (Nhb_weekendMatrix / numberOfweekend ) * expansionFactorTract
            
            
            
            