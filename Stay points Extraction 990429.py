# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:55:42 2020

@author: Asus
"""




#%% 
from sklearn.metrics.pairwise import haversine_distances         # Import this for pairwise distances to get medoids
from geopy.distance import geodesic                              # This will calculate haversine to meters
import numpy as np                                               # This is useful for obtaining medoid index
from sklearn.cluster import AgglomerativeClustering
#import dask.dataframe as dd
import geopandas as gpd
import pandas as pd 
#import modin.pandas as ppp

#%%

callsDataset = pd.read_csv(r'C:\Users\Rahmani\Desktop\MobileData\Data\Run Data\Sample3000_5days.csv', parse_dates=['Date_Time'])
shapeFile = gpd.read_file(r'C:\Users\Rahmani\Desktop\MobileData\TehranGeorefed\Tehran\tehran.shp')  
#%%

pd.options.mode.chained_assignment = None
callsDataset.dropna(inplace=True)

def medoid(candidateSet=[]):
    """
    This func use sklearn pairwise matrices to calculate medoid for each canidateSet.
    A candidateSet is a set of points for each person which semms to be stop points
    based on the short distance criteria and the minimum activity period
    """
                                      
    # Convert coordinate to radians / IS IT BETTER TO APPEND THE RADIANS AS NEW FIELDS TO SAVE MEMRORY AND TO HAVE THEM FOR LATER CALs?
    l = [candidateSet[i][2:4] * (np.pi / 180) for i in range(len(candidateSet))]                                                          # CAN'T WE CHANGE THEM IN PLACE TO SAVE THE MEMORY?
    medIndex = np.argmin((haversine_distances(l) * 6731000).sum(axis=1)) # Harversine calculates the distance between two points in Kilometers
                                                                # and Argmin return the index of the minimum value in an array
                                                                # So with this line of code, we want to find the index of the point with minimum distance with others
    cSetMedoid = candidateSet[medIndex]
    cSetMedoid = np.append(cSetMedoid,[candidateSet[0][0],candidateSet[-1][0] - candidateSet[0][0]])
    
    l.clear()
    
    
    return cSetMedoid


def agg_cluster(stopPoints=[]):
    """                             
    This function will use sklearn hierarchical clustering to cluster the stop points into stayPoints.
    For example, two home location in the morning and in the evening (stopPoints) are clustered into one home location (stayPoint)
    """
     
    stayPointsTemp = []                       
    stopPointsModifiedTemp = pd.DataFrame(stopPoints,columns=['Date_Time','ID','Lat','Long','StayStart','Duration'])
    
    if len(stopPoints)>1: 
        dist = haversine_distances(stopPointsModifiedTemp[['Lat','Long']]) * (np.pi/180) * 6731000 # ADD COMMENT
       
        Agg = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=500).fit(dist) #$New 
                                                                                                    # https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
                                                                                                    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
                                                                                                    # The parameters need modifications. e.g., the linkage = 'complete' / AND WHY NOT DBSCAN                                                                         
        stopPointsModifiedTemp['cluster'] = Agg.labels_                                                             #add cluster index to final cluster
     
    
        for clusterNo in stopPointsModifiedTemp.cluster.unique():                                                    #this process will find medoid of each final cluster and then append medoids lat/long to the same clusters
            clusterN = stopPointsModifiedTemp.loc[stopPointsModifiedTemp.cluster == clusterNo][['Lat','Long','Date_Time']]
            medIndex = np.argmin(haversine_distances(clusterN[['Lat','Long']]).sum(axis=1))
            medClusterN = clusterN.iloc[medIndex]
            stayPointsTemp.append(medClusterN)   
            stopPointsModifiedTemp.loc[(stopPointsModifiedTemp.cluster == clusterNo),'Lat'] = medClusterN.Lat # WHY TWO =?
            stopPointsModifiedTemp.loc[(stopPointsModifiedTemp.cluster == clusterNo),'Long'] = medClusterN.Long # HERE WE CAN HAVE ANOTHER VARIABLE FOR UNIQUE POINTS WHICH ARE CLUSTERED. BUT, WE ARE JUST CHAGING THE LATLONG IN THE BASE DATASET
        
        stopPointsModifiedTemp = gpd.GeoDataFrame(stopPointsModifiedTemp, geometry=gpd.points_from_xy(x=stopPointsModifiedTemp.Long, y=stopPointsModifiedTemp.Lat), crs={'init': 'epsg:4326'}) #$NEW use geopandas to create a geo DataFrame
        stopPointsModifiedTemp = gpd.sjoin(stopPointsModifiedTemp, shapeFile, how='left', op='within') # Spatial join
        stopPointsModifiedTemp.drop(columns=['index_right', 'geometry'], inplace=True)                 # Drop sth which is not needed
        stopPointsModifiedTemp.fillna(float(0), inplace=True)                                                     #This is for filling points which are not in Tehran shapeFile
        
        
        return stopPointsModifiedTemp
        
    else:
         return stopPointsModifiedTemp


def HWO_finder(notLabeledStopPoints):
    """
    This function allocates home, work, and other labels to the stay points. At this stage,
    the inpute argument is the stayPoints list because we need the ferquency at which 
    a stay location has been observed. 
    """
    
    labeledStopPoints = notLabeledStopPoints # These variables will be removed later. They are here just for clarification now!
    labeledStopPoints['Day'] = labeledStopPoints['StayStart'].dt.day_name()               #Extract Day name                  
    labeledStopPoints['Week'] = labeledStopPoints['StayStart'].dt.week                    #Extract Week name
    labeledStopPoints.set_index('StayStart', inplace=True)                             #Set index as Start Stay to use between Func
    labeledStopPoints['Purpose'] = np.nan                                                #Create column of empty value
    #labeledStopPoints['TypeDay']=np.nan
    
    homeLoc = labeledStopPoints.loc[(labeledStopPoints.Day != 'Friday')][['Lat','Long']].between_time('19:00', '7:00').mode()                         #Find most used location as Home
    labeledStopPoints.loc[((labeledStopPoints.Lat.isin(homeLoc.Lat)) & (labeledStopPoints.Long.isin(homeLoc.Long))),'Purpose']='Home'                
    ##add frequencies 
    workLoc = labeledStopPoints.loc[(labeledStopPoints.Purpose != 'Home') & (labeledStopPoints.Day != 'Friday')][['Lat','Long']].between_time('7:00', '19:00').mode()  #Find most used location as Work. excluding Home
    labeledStopPoints.loc[((labeledStopPoints.Lat.isin(workLoc.Lat)) & (labeledStopPoints.Long.isin(workLoc.Long))), 'Purpose']='Work'   
    
    if ((labeledStopPoints.loc[(labeledStopPoints.Purpose == 'Work')].shape[0]) / (len(labeledStopPoints.Week.unique()))) < 1:  #New                                                                           #to get average trip frequency for five week 
        labeledStopPoints.loc[(labeledStopPoints.Purpose == 'Work'), 'Purpose'] = np.nan
    
    labeledStopPoints.loc[((labeledStopPoints.Purpose != 'Work') & (labeledStopPoints.Purpose != 'Home')), 'Purpose'] = 'Other' 
    
                                            #Nan value to other
    #labeledStopPoints.drop(columns=['Day'], inplace=True)      #&NEW   drop sth which is not needed
    
    return labeledStopPoints 




#%%
dataFrameStorage=[]
stay_labeled={}
usersCDRTract=np.zeros((23,1))

counter=0
callsDataset.sort_values(by=['Date_Time'], inplace=True) 
rr = callsDataset.groupby(by=['ID'])

for ID in callsDataset.ID.unique():
    
    counter+=1
    uniqIdCalls = rr.get_group(ID) # Extracting the callDataSet for person with ID==ID
    #uniqIdCalls.sort_values(by=['Date_Time'], inplace=True)       # Sorting the dataset based on time
    uniqIdCalls=uniqIdCalls.to_numpy()
    
    
    candidateSet = []                                               # For storing the candidate set for each person. A candidate set is a set of points that are locally close to each other.
    stopPoints = []                                                 # Stop points are the centroid (medoid) of candidate sets keeping in mind that the first and the last points in the candidate sets should have a minimum time difference
    allCandidateSets = []                                           # For storing all candidate sets of a person
    
    candidateSet.append(uniqIdCalls[0])   #using iloc will slice dataset by using rows true index. we can use advantages of lists and Pandas series 
    
    for i in range(len(uniqIdCalls) - 1):                                #this will iterate row by row of data frame which is a tuple and I use 0 index to get number of rows
        if geodesic(uniqIdCalls[i][2:4], uniqIdCalls[i+1][2:4]).meters <=500:   #This will combine geopy and data frame row by row indexes for getting staypoint
            candidateSet.append(uniqIdCalls[i+1])   
            
            
        else:
            if (candidateSet[-1][0] - candidateSet[0][0]).seconds > 600:                      #Use pandas timestamps to get durations 
                
                stopPoints.append(medoid(candidateSet)) # WE SHOULD ALSO KEEP THE INITIAL DATASET BECAUSE WE MIGHT NEED THEM IN FUTURE. WE CAN APPEND ONLY STAYPOINT AND LCUSTERS LABELS TO THE INITIAL DATASET
                
                allCandidateSets.append(candidateSet)
            candidateSet = []
            candidateSet.append(uniqIdCalls[i+1])
    
    
    if (candidateSet[-1][0] - candidateSet[0][0]).seconds > 600:  #append final cluster
       stopPoints.append(medoid(candidateSet))
       allCandidateSets.append(candidateSet)
       
    
    if len(stopPoints)>=1:     
           stopPointsModified = agg_cluster(stopPoints)
           stopPointsLabeled = HWO_finder(stopPointsModified)
           
    if ((stopPointsLabeled.loc[(stopPointsLabeled.Purpose == 'Home')].shape[0]) / (len(stopPointsLabeled.Week.unique()))) > 1:
    
        
        stopPointsLabeled.reset_index(inplace=True)
        Home=stopPointsLabeled.loc[stopPointsLabeled.Purpose=='Home'].head(1)
        usersCDRTract[int(Home['manategh'])]+=1
        stay_labeled['{}'.format(ID)]=stopPointsLabeled
        #stopPointsLabeled_ID['{}'.format(ID)]=stopPointsLabeled
        #dataFrameStorage.append(stopPointsLabeled)        
    #t=time.time()
    #if len(dataFrameStorage)>1:
   
        #dataFrameFinal=pd.concat(dataFrameStorage)
    #print(time.time()-t)
#dataFrameFinal.to_csv('E:dataframe.csv',index=False)
pd.DataFrame(usersCDRTract,columns=['Pop']).to_csv(r'C:\Users\Rahmani\Desktop\MobileData\Data\Stay Point Output files\usertractpopcdr.csv',index=False)

storing=[]
for key in stay_labeled:
    storing.append(stay_labeled['{}'.format(key)])
pd.concat(storing).to_csv(r'C:\Users\Rahmani\Desktop\MobileData\Data\Stay Point Output files\stayPointsDictionary.csv',index=False)
