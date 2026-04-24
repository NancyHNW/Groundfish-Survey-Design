import numpy as np
from matplotlib import pyplot as plt
from .distance import deg2point, degmin2deg

def drawTour(tour, Data, ports=None, connect_tour=True):
    # Mean catch represented by line with
    scale = np.max(Data[:,-1])-np.min(Data[:,-1])
    Data[:,-1] = 1 # +7*(Data[:,-1]-np.min(Data[:,-1]))/scale
    # Draw Iceland
    with open('data/island.bin', 'rb') as f:
        landata = np.fromfile(f, dtype=np.float32)
    LandDeg = np.vstack((landata[:int(landata.size/2)],landata[int(landata.size/2):])).T
    (x,y) = deg2point(LandDeg[:,0],LandDeg[:,1])
    fig,ax = plt.subplots(figsize=(40,30)) # figsize=(15,13)
    # ax.autoscale(enable=False)
    ax.plot(x,y)
    ax.invert_yaxis()
    # Find where the tour starts at any port
    pId = list(range(len(ports)))
    numports = len(pId)
    if numports % 2 == 0:
      stationstart = numports
    else:
      stationstart = numports+1
    n = len(tour)
    idx = 0
    for pid in pId:
      if pid in tour:
        idx = tour.index(pid)
        tour = tour[idx:] + tour[:idx] # reorder tour so we start at the start location
        break
    print("tour:", tour)
    t = tour[0]
    if t in pId:
      i = t
    else:
      i = numports+int((t-stationstart)/2) # this should have the index 0 (iStart)
    ilast = i # initialize our last location

    print("i=",i)
    (x_,y_) = deg2point([degmin2deg(Data[i,3])],-np.array([degmin2deg(Data[i,4])]))
    plt.plot(x_,y_,marker = 6, color = "g") # Draw the start location, we would also like to end here!

    if t in pId:
      (x1,y1) = deg2point([degmin2deg(Data[i,3])],-np.array([degmin2deg(Data[i,4])]))      
    elif (int(t-stationstart) % 2) == 0: # is even number we enter the station from defined start
      (x1,y1) = deg2point([degmin2deg(Data[i,5])],-np.array([degmin2deg(Data[i,6])]))
    else: # enter the station from the other end (reverse direction)
      (x1,y1) = deg2point([degmin2deg(Data[i,3])],-np.array([degmin2deg(Data[i,4])]))

    for t in tour[1:]:
      if t in pId:
        i = t
        (x2,y2) = deg2point([degmin2deg(Data[i,3])],-np.array([degmin2deg(Data[i,4])]))
      else:
        i = numports+int((t-stationstart)/2)
        if (int(t-stationstart) % 2) == 0:
          (x2,y2) = deg2point([degmin2deg(Data[i,3])],-np.array([degmin2deg(Data[i,4])]))
          ax.text(x2[0],y2[0],str(t))
        else:
          (x2,y2) = deg2point([degmin2deg(Data[i,5])],-np.array([degmin2deg(Data[i,6])]))
          ax.text(x2[0],y2[0],str(t))
 
      if (i != ilast): # this is not the same station as last
        if connect_tour:
          ax.plot([x1[0],x2[0]],[y1[0],y2[0]],color="r",linestyle='dotted', linewidth=0.4)
      else: # both last point and this one are from the same station (draw the station as an arrow)
        ax.arrow(x1[0],y1[0],x2[0]-x1[0],y2[0]-y1[0], head_width=1, head_length=1, linewidth = 0.4*Data[i,-1], fc='g', ec='g')
      (x1,y1) = (x2,y2)
      ilast = i # keep track of last station
    if connect_tour:
        ax.plot([x1[0],x_[0]],[y1[0],y_[0]],color="k",linestyle='dotted', linewidth=1)

    # let us also draw the ports available
    (x_,y_) = deg2point(ports['latitude'],ports['longitude'])
    for i in range(len(x_)):
        ax.plot(x_[i],y_[i], marker = 'o', color = "m", markersize = 2) # Draw port location
        ax.text(x_[i],y_[i],str(ports['description'].iloc[i]), fontsize = 8) 
    ax.set_xlim([-500,500])   
    ax.set_ylim([300,-300])     
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)