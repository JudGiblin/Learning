# Learning
Scripts compiled while learning python

## Basic info on opening text and plotting graphs
filename='RMI_sea_level_data_2016.csv'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as datetime
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

sealevel = np.loadtxt(filename, delimiter = ',', skiprows=1, usecols=[1], dtype=float)
time = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True, converters={0: mdates.strpdate2num('%d/%m/%Y %H:%M')})
time_cor = time[0,:]

#######################################
## Plotting the yearly sealevel data
years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%b')

fig = plt.figure(1, figsize=(12,5))
plt.plot_date(time_cor,sealevel,'k')
plt.xlabel('2016')
plt.ylabel('Sea level data [m]')
plt.grid(True)
plt.title('RMI Sea Level Data in 2016')
ax=plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(yearsFmt)
ax.autoscale_view()
plt.show()

## Using a loop to determine monthly maxima, mean, and percentile
f=open('RMI_output_2016.txt', 'w')
i=00
for i in range(1,12):
    indx=((time_cor>=mdates.date2num(datetime.datetime(2016,i,01,0,0,0))) & (time_cor<mdates.date2num(datetime.datetime(2016,i+1,01,0,0,0))))
    string1 = "Month"
    string2 = "2016:"
    string3 = "Maximum sealevel is"
    string4 = "m"
    string5 = "Mean sealevel is"
    string6 = "Standard deviation is"
    string7 = "The 50th percentile is"
    
    s= "%s %d %s %s %.2f %s\n" %(string1,i,string2,string3,max(sealevel[indx]),string4)
    t= "%s %d %s %s %.2f %s\n" %(string1,i,string2,string5,np.mean(sealevel[indx]),string4)
    u= "%s %d %s %s %.2f %s\n" %(string1,i,string2,string6,np.std(sealevel[indx]),string4)
    v= "%s %d %s %s %.2f %s\n" %(string1,i,string2,string7,np.percentile(sealevel[indx],50),string4)
    f.write(s), f.write(t), f.write(u), f.write(v)
    
    print "Month", i, "2016:", "Maximum sealevel is", "%.2f m" % max(sealevel[indx])
    print "Month", i, "2016:", "Mean sealevel is", "%.2f m" % np.mean(sealevel[indx])
    print "Month", i, "2016:", "Standard deviation is", "%.2f m" % np.std(sealevel[indx])
    print "Month", i, "2016:", "The 50th percentile is", "%.2f m" % np.percentile(sealevel[indx],50)

f.close()
np.savetxt('sealevel_2016.txt', sealevel, fmt='%f')
np.savetxt('time_2016.txt', time_cor, fmt='%f')

## Info on comparing SOI and sealevel when the time of SOI does not align with the sealevel time recorded
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import matplotlib.ticker as ticker
import datetime as datetime

sealevel = np.loadtxt('RMI_sealevel_1993-2016.txt', delimiter ='\t', dtype=float)
time = np.loadtxt('RMI_time_1993-2016.txt', delimiter = '\t', dtype=float)

## Inputting SOI data
filename = 'NOAA_SOI.txt'
SOI_data = np.loadtxt(filename, delimiter = '\t', skiprows=1, usecols=[1], dtype=float)
SOI_time = np.loadtxt(filename, delimiter = '\t', skiprows=1, usecols=[0], unpack=True, converters={0: mdates.strpdate2num('%Y%m')})

indx = (SOI_time>=mdates.date2num(datetime.datetime(1993,01,01,0,0,0)))
SOI_time_apl = SOI_time[indx] #applicable SOI time
SOI_data_apl = SOI_data[indx] #applicable corresponding SOI data

i=0
Mean_sealevel =[]
for i in SOI_time_apl:
    #3 month mean
    indx=((time>=(i-45)) & (time<(i+45)))
    Mean_sealevel = np.append(Mean_sealevel, np.mean(sealevel[indx]))
    
indexnonnan = (np.isnan(Mean_sealevel) == 0)
Corr = np.corrcoef(SOI_data_apl[indexnonnan], Mean_sealevel[indexnonnan]) # arrays dimensions must match
print Corr

#Calculate correlation coefficients and p values for all pairs of the matrix
import scipy.stats as ss

Pearson_Corr = ss.pearsonr(SOI_data_apl[indexnonnan], Mean_sealevel[indexnonnan])
print Pearson_Corr

#Cross correlation
Cross_corr = np.correlate(SOI_data_apl[indexnonnan], Mean_sealevel[indexnonnan])
print Cross_corr

## Plotting the yearly SOI data
years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')

##graphing for two axes
plt.rc('figure', figsize=(11.69,8.27))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot_date(SOI_time_apl, Mean_sealevel, 'b', label = 'Sealevel:3 month mean')
ax2.plot_date(SOI_time_apl, SOI_data_apl, 'k', label = 'SOI:3 month mean')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Mean sealevel [m]', fontsize=14, color='b')
ax2.set_ylabel('SOI', fontsize=14)
plt.title('RMI Mean Sealevel vs. SOI', fontsize=18)
ax=plt.gca()
plt.grid(True)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
plt.show()

## Handling ONI data
filename= 'ONI_data_Copy.txt'
data= np.loadtxt(filename, usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
print np.shape(data)

ONI_ind= np.reshape(data, (816))
print np.shape(ONI_ind)

i = 0
x = 0
ONI_time =[]
for i in range (1950,2018):
    for x in range (1,13):
        ONI_time = np.append(ONI_time, mdates.date2num(datetime.datetime(i,x,01,0,0,0)))

print np.shape(ONI_time)

ind = (ONI_time>=mdates.date2num(datetime.datetime(1993,01,01,0,0,0)))
ONI_time_apl = ONI_time[ind]
ONI_ind_apl = ONI_ind[ind]

ONI_ind_apl[ONI_ind_apl==-99.9]=np.nan

#plotting graph
years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')

plt.rc('figure', figsize=(11.69,8.27))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot_date(SOI_time_apl, Mean_sealevel, 'b', label = 'Sealevel:3 month mean')
ax2.plot_date(ONI_time_apl, ONI_ind_apl, 'k', label = 'ONI:3 month mean')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Mean sealevel [m]', fontsize=14, color='b')
ax2.set_ylabel('ONI', fontsize=14)
ax2.set_ylim(2.5,-2)
plt.title('RMI Mean Sealevel vs. ONI', fontsize=18)
ax=plt.gca()
plt.grid(True)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_major_locator(ticker.MultipleLocator(800))
plt.show()
fig.savefig('RMI Sealevel vs ONI.jpeg')

# Reading RMI tidegauge data from files in a directory

import os
import csv
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
directory = "C:\Users\judithg\Documents\Python Scripts\RMI_sealevel"
sealevel=[]
time=[]
for root,dirs,files in os.walk(directory):
for root,dirs,files in os.walk(directory):
    ##name1='IDO70002_'
    ##xt='.csv'
    ##for i in range(1993:2017):
       #filename = name1 + str(i) + xt
    for file in files:
        print file
        if file.endswith(".csv"):
            sealevel = np.append(sealevel, np.loadtxt(file, delimiter = ',', skiprows=1, usecols=[1], dtype=float))
            time = np.append(time, np.genfromtxt(file, delimiter=',', skiprows=1, usecols=[0], unpack=True, converters={0: mdates.strpdate2num('%d-%b-%Y %H:%M')}))

indx = sealevel>-8000
sealevel_cor = sealevel[indx]
time_cor = time[indx]

##Ploting graph
##plotting the yearly sealevel data
years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')

fig = plt.figure(1, figsize=(12,5))
plt.plot_date(time_cor,sealevel_cor,'k')
plt.xlabel('Years')
plt.ylabel('Sea level data [m]')
plt.grid(True)
plt.title('RMI Sea Level Data')
ax=plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.autoscale_view()
plt.show()
##saving data
fig.savefig('RMI_sealevel_1993-2016.jpeg')
np.savetxt('RMI_sealevel_1993-2016.txt', sealevel_cor, fmt='%f')
np.savetxt('RMI_time_1993-2016.txt', time_cor, fmt='%f')
