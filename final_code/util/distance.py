import numpy as np

def arcdist(lat1, lon1, lat2, lon2):
    """Calculate arc distance from radians."""
    r = 3437.905   # Earth radius in miles
    angle = np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)
    return r*np.arccos(angle)

def degmin2rad(degmin):
   # if (np.abs(degmin)<10000):
   #     degmin = degmin*100
    m = (degmin/100.)-np.floor(degmin/10000.)*100.
    return ((degmin+(200.0/3.0)*m)/10000)*np.pi/180.0

def degmin2deg(degmin):
    # if (np.abs(degmin)<10000):
    #     degmin = degmin*100
    min_ = (degmin/100)-np.floor(degmin/10000.)*100.
    return (degmin+(200.0/3.0)*min_)/10000

def deg2degmin(deg):
    degmin = np.floor(deg)*10000.
    min_ = (deg  - np.floor(deg ))*100*60
    degmin = degmin + min_
    return degmin

def deg2point (Lat, Lon, Norm=1000):
   return deg2pointInverted(Lat, -Lon, Norm)

def degArr2point(arr):
    """Input: [[lat, lon] for point in points]"""
    return np.array(deg2point(*arr.T))

def deg2pointInverted (Lat, Lon, Norm):
    (MAXLON,MAXLAT,MINLAT,MINLON) = (-4,70,60,-32) # at 65lat 18lon
    lat65 = 65.*np.pi/180. 
    x65 = (111415.13*np.cos(lat65)-94.55*np.cos(3*lat65)+0.12*np.cos(5*lat65))/60
    M65 = 7915.704456*np.log10(np.tan(np.pi/4+lat65/2)) - np.sin(lat65)*(23.110771+0.052051*(np.sin(lat65))*np.sin(lat65)) # Automatic Scaling
    lat = MAXLAT*np.pi/180
    M = 7915.704456*np.log10(np.tan(np.pi/4+lat/2))-np.sin(lat)*(23.110771+0.052051*(np.sin(lat))*np.sin(lat65))
    Diff = M - M65
    lat = MINLAT*np.pi/180
    M = 7915.704456*np.log10(np.tan(np.pi/4+lat/2))-np.sin(lat)*(23.110771+0.052051*(np.sin(lat))*np.sin(lat65))
    Diff = Diff + M65 - M
    scale = Norm/(Diff*x65)
    x65 = scale*x65
    x = []; y = []
    for i in range(len(Lat)):
        lat = Lat[i]*np.pi/180
        M = 7915.704456*np.log10(np.tan(np.pi/4+lat/2))-np.sin(lat)*(23.110771+0.052051*(np.sin(lat))*np.sin(lat65))
        Diff = M65-M
        y.append(int(Diff*x65))
        x.append(int((Lon[i]+18)*x65*60))
    return x, y

CenterPoint = (65, 18)