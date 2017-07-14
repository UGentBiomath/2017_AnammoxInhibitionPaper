# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:32:57 2015

@author: chaimdemulder
@purpose: read a measurements file and smoothen or filter the measurements with a method
    of choice; available methods: simple moving average, moving slope filtering,
    moving average filtering.
@copyright: (c) 2015, Chaïm De Mulder
"""
import sys
import os
from os import listdir
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt   #plotten in python
import xlrd
import datetime as dt
#from DateTime import DateTime


#####################
###   FUNCTIONS   ###
#####################
def delete_doubles(dataframe,data,
                   log_file=os.path.join(os.getcwd(),'delete_doubles_log.txt')):
    '''
    deletes double values that occur in a row, to avoid an abundance of
    relatively meaningless datapoints (e.g. measuring frequency too high)

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe from which double values need to be removed
    data : str
        column name of the column from which double values will be sought
        and removed

    Returns
    -------
    new_dataframe : pd.DataFrame
        the dataframe from which the double values of 'data' are removed
    '''
    original = len(dataframe)

    #Create temporary dataframe column with True boolean value if datapoint can
    #stay because it is different from the previous one
    dataframe['cond_to_drop'] = pd.Series([n for n in dataframe[data].diff() != 0])
    new_dataframe = dataframe.drop(dataframe[dataframe.cond_to_drop==False].index)
    new_dataframe.drop('cond_to_drop',axis=1,inplace=True)
    new_dataframe.reset_index(drop=True,inplace=True)

    log_file = open(log_file,'a')
    log_file.write(str('\nOriginal dataset: '+str(original)+' datapoints; New dataset: '+
                   str(len(new_dataframe))+' datapoints; '+str(original-len(new_dataframe))+
                   ' subsequent duplicates removed\n'))
    log_file.close()

    return new_dataframe

def calc_slopes(dataframe,xdata,ydata,time_unit=None):
    """
    Calculates slopes for given xdata and ydata

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe from which both the time- and the data-points will be used to
        calculate the slopes in different points
    xdata : str
        name of the column containing the xdata for slope calculation (e.g. time);
        this data should be in the form of datetime.timestamp types
    ydata : str
        name of the column containing the ydata for slope calculation

    Returns
    -------
    pd.DataFrame
        dataframe containing an added column with the slopes calculated for the
        chosen variable, named after the variable + _slopes
    """
    new_name = ydata+'_slopes'
    if time_unit == 'sec':
        dataframe[new_name] = dataframe[ydata].diff()/ \
                              (dataframe[xdata].diff().dt.seconds)
    elif time_unit == 'min':
        dataframe[new_name] = dataframe[ydata].diff() / \
                              (dataframe[xdata].diff().dt.seconds / 60)
    elif time_unit == 'hr':
        dataframe[new_name] = dataframe[ydata].diff() / \
                              (dataframe[xdata].diff().dt.seconds / 3600)
    elif time_unit == 'd':
        dataframe[new_name] = dataframe[ydata].diff() / \
                              (dataframe[xdata].diff().dt.days + \
                              dataframe[xdata].diff().dt.seconds / 3600 / 24)
    elif time_unit == None:
        dataframe[new_name] = dataframe[ydata].diff() / dataframe[xdata].diff()
    else :
        print('Something went wrong. If you are using time-units to calculate \
               slopes, please make sure you entered a valid time unit for slope \
               calculation (sec, min, hr or d)')
        return None

    return dataframe

def drop_peaks(dataframe,data,cutoff):
    """
    Filters out the peaks larger than a cut-off value in a dataseries

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe from which the peaks need to be removed
    data : str
        the name of the column to use for the removal of peak values
    cutoff : int
        cut off value to use for the removing of peaks; values with an
        absolute value larger than this cut off will be removed from the data

    Returns
    -------
    pd.DataFrame
        dataframe with the peaks removed
    """
    dataframe = dataframe.drop(dataframe[abs(dataframe[data]) > cutoff].index)

    return dataframe

def simple_moving_average(dataframe,data,window):
    """
    Calculate the Simple Moving Average of a dataseries from a dataframe, using
    a window within which the datavalues are averaged; This is a slower
    implementation, but allows more understanding of the meaning of a simple
    moving average.

    Parameters
    ----------
    dataframe : pd.DataFrame
        the dataframe containing the data that needs to be smoothened.
    data : str
        name of the column containing the data that needs to be smoothened
    window : int
        the number of values from the dataset that are used to take the average
        at the current point.

    Returns
    -------
    pd.DataFrame
        the dataframe, extended with a column containing the smoothened values.
    """
    #Check if the window for the average is not larger then the amount of data
    if len(dataframe) < window:
        raise ValueError("Window width exceeds number of datapoints!")

    new_name=data+'_smooth'
    dataframe[new_name] = pd.Series(pd.rolling_mean(dataframe[data],
                                                    window=window,
                                                    center=True),
                                    index=dataframe.index)

    return dataframe

def moving_average_filter(dataframe,data,window,cutoff,
                          log_file=os.path.join(os.getcwd(),'filter_log.txt')):
    """
    Filters out the peaks/outliers in a dataset by comparing it's values to a
    smoothened representation of the dataset (Moving Average Filtering)

    Parameters
    ----------
    dataframe : pd.DataFrame
        the dataframe containing the data that needs to be smoothened.
    data : str
        name of the column containing the data that needs to be smoothened
    window : int
        the number of values from the dataset that are used to take the average
        at the current point.
    cutoff: int
        the cutoff value to compare the difference between data and smoothened
        data with to apply the filtering.

    Returns
    -------
    pd.DataFrame
        the adjusted dataframe with the filtered values
    """

    original = len(dataframe)

    #Calculate smoothened dataset
    dataframe_smooth = simple_moving_average(dataframe,data,window)
    smooth_name = dataframe_smooth.columns[-1]

    #Compare data with smoothened data and remove datapoints that divert too much
    difference = dataframe_smooth[data]-dataframe_smooth[smooth_name]
    dataframe['difference'] = pd.Series([n for n in abs(difference) > cutoff])
    dataframe = dataframe.drop(dataframe[dataframe.difference==True].index)
    dataframe = dataframe.drop('difference',axis=1)
    dataframe.reset_index(drop=True,inplace=True)

    log_file = open(log_file,'a')
    log_file.write(str('\nOriginal dataset: '+str(original)+' datapoints; new dataset: '+
                   str(len(dataframe))+' datapoints')+str('\n'+str(original-len(dataframe))+
                   ' datapoints filtered'))
    log_file.close()

    return dataframe

def moving_slope_filter(dataframe,time,data,cutoff,time_unit=None,
                        log_file=os.path.join(os.getcwd(),'filter_log.txt')):
    """
    Filters out datapoints based on the difference between the slope in one point
    and the next (sudden changes like noise get filtered out), based on a given
    cut off; Moving Slope Filtering

    Parameters
    ----------
    dataframe : pd.DataFrame
        the dataframe containing the data that needs to be smoothened.
    time : str
        name of the time column in the dataframe
    data : str
        name of the column containing the data that needs to be filtered
    cutoff: int
        the cutoff value to compare the slopes with to apply the filtering.

    Returns
    -------
    pd.DataFrame
        the adjusted dataframe with the filtered values

    """
    original = len(dataframe)

    #calculate initial slopes
    new_dataframe = calc_slopes(dataframe,time,data,time_unit=time_unit)
    new_name = dataframe.columns[-1]

    #As long as the slope column contains values higher then cutoff, remove those
    #rows from the dataframe and recalculate slopes
    while abs(new_dataframe[new_name]).max() > cutoff:
        new_dataframe = new_dataframe.drop(new_dataframe[abs(new_dataframe[new_name]) > cutoff].index)
        new_dataframe = calc_slopes(new_dataframe,time,data,time_unit=time_unit)

    new_dataframe = new_dataframe.drop(new_name,axis='columns')
    new_dataframe.reset_index(drop=True,inplace=True)

    log_file = open(log_file,'a')
    log_file.write(str('Original dataset: '+str(original)+' datapoints; new dataset: '+
                   str(len(new_dataframe))+' datapoints'+str(original-len(new_dataframe))+
                   ' datapoints filtered\n'))
    log_file.close()

    return new_dataframe

def _select_slope(dataframe,ydata,down=True,limits=[0,0],
                  log_file=os.path.join(os.getcwd(),'select_slope_log.txt')):#,based_on_max=True):#,bounds=[1,1]):
    """
    Selects down- or upward sloping data from a given dataseries, based on the
    maximum in the dataseries. This requires only one maximum to be present in
    the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        pandas dataframe containing the series for which slopes, either up or
        down, need to be selected
    ydata : str
        name of the column containing the data for which slopes, either up or
        down, need to be selected
    down : bool
        if True, the downwards slopes are selected, if False, the upward slopes
    limits : array with two values
        min and max value that is allowed for the data

    based_on_max : bool
        if True, the data is selected based on the maximum of the data, if
        false it is based on the minimum
    bounds : array
        array containing two integer values, indicating the extra margin of
        values that needs to be dropped from the dataset to avoid selecting
        irregular data (e.g. not straightened out after reaching of maximum)
    Returns
    -------
    pd.DataFrame:
        a dataframe from which the non-down or -upward sloping data are dropped
    """
    #if based_on_max == True:
    drop_index = dataframe[ydata].idxmax()
    old_len = len(dataframe)
    if down == True:
        try:
            log_file = open(log_file,'a')
            log_file.write('\nSelecting downward slope...')
            dataframe = dataframe[drop_index:]
            dataframe = dataframe[limits[0] < dataframe[ydata]]
            dataframe = dataframe[dataframe[ydata] < limits[1]]
            #dataframe = dataframe[dataframe[ydata] > limits[0]]
            new_len = len(dataframe)
            dataframe.reset_index(drop=True,inplace=True)
            log_file.write(str(str(old_len-new_len)+' datapoints dropped, '+
                           str(new_len)+' datapoints left.\n'))
            return dataframe
        except:#IndexError:
            print( 'Not enough datapoints left for selection')
            return pd.DataFrame()

    elif down == False:
        try:
            log_file = open(log_file,'a')
            log_file.write('\nSelecting upward slope...')
            dataframe = dataframe[:drop_index]
            dataframe = dataframe[limits[0] < dataframe[ydata]]
            dataframe = dataframe[dataframe[ydata] < limits[1]]
            new_len = len(dataframe)
            dataframe.reset_index(drop=True,inplace=True)
            log_file.write(str(str(old_len-new_len)+'datapoints dropped, '+
                           str(new_len)+'datapoints left.\n'))
            return dataframe
        except:#IndexError:
            print( 'Not enough datapoints left for selection')
            return pd.DataFrame()

#    elif based_on_max == False:
#        drop_index = dataframe[ydata].idxmin()
#        if down == True:
#            try:
#                print( 'Selecting downward slope:',drop_index+sum(bounds),\
#                'datapoints dropped,',len(dataframe)-drop_index-sum(bounds),\
#                'datapoints left.')
#
#                dataframe = dataframe[bounds[0]:drop_index-bounds[1]]
#                dataframe.reset_index(drop=True,inplace=True)
#                return dataframe
#            except IndexError:
#                print( 'Not enough datapoints left for selection')
#
#        elif down == False:
#            try:
#                print( 'Selecting upward slope:',len(dataframe)-drop_index+sum(bounds),\
#                'datapoints dropped,',drop_index-sum(bounds),'datapoints left.')
#
#                dataframe = dataframe[drop_index+bounds[0]:-bounds[1]]
#                dataframe.reset_index(drop=True,inplace=True)
#                return dataframe
#            except IndexError:
#                print( 'Not enough datapoints left for selection')
#
def extract_slopes(path,xdata,ydata,filter_function,cutoff,
                   ext='text',comment='#',down=True,limits=[0,0],time_unit='sec',
                   delta_t=dt.timedelta(weeks=1),schrikkel=False,
                   plot=[True,(-500,-400,-300,-200,-100,0)]):
    """

    Parameters
    ----------
    path : str
        directory containing the files that need to be read and analysed
    xdata : str
        name of the columns in the datafiles containing the xdata, often time
    ydata : str
        name of the columns in the datafiles containing the ydata, usually
        measurements
    filter_function : function
        function to be used to clean the data from noise or anomalies
    cutoff : int
        cutoff value to use in the filter_function
    ext : str
        extension of the files to be read, possible: text, csv (not tested yet),
        defaults to text
    comment : str
        the sign indicating the beginning of a comment in the files, to remove
        the header before reading the files
    down : bool
        if True, the downwards slopes are selected, if False, the upward slopes
    time_unit : str
        time unit with which to calculate slopes
    schrikkel : bool
        if true, the year in which the measurements were done was a 'schrikkel-
        jaar'; important for exact date plotting
    plot : bool
        if True, a figure and axes object will be produced and added to the
        output for the user to adjust to his/her wishes

    Returns
    -------
    pd.DataFrame :
        dataframe containing two or three colums:
            1) timestamp (in pandas Timestamp type)
            2) the mean slope values of the selected data from every file in
            the directory
            3) the mean standard deviations of the selected data from every file
            in the directory
    plt.figure :
        matplotlib figure object depicting the data in the dataframe
    plt.axes :
        matplotlib axes object to go with the figure object

    """
    slopes_mean =[]
    slopes_std = []
    timestamp = []

    if ext == 'text':
        files = [f for f in listdir(path) if f.endswith('.txt')]
    elif ext == 'csv':
        files = [f for f in listdir(path) if f.endswith('.csv')]
    else:
        print( 'No files with',ext,'extension found in directory',path,'. \
               Please choose one of the following: text, csv')

    #Sort files alphabetically to make sure they are treated in the
    #correct order
    files.sort()

    print( 'Reading',len(files),'files...')
    log_file_location = os.path.join(path,'log.md')
    print( 'Creating log-file at',log_file_location)

    if os.path.exists(log_file_location):
        os.remove(log_file_location)
    log_file = open(log_file_location,'a')

    #create figure and axis object for combined histogram figure outside
    #for-loop!
    fig_hist, ax_hist = plt.subplots()
    fig_hist.hold(True)
    #Read files
    for file_name in files:
        log_file.write(str('\nReading file: '+file_name+
                       '\n----------------------------'))
        dir_file_path = os.path.join(path,file_name)
        with open(dir_file_path, 'r') as read_file:
            headerlength = _get_header_length(read_file,ext=ext,
                                              comment=comment)
            data = _read_file(dir_file_path,ext=ext,skiprows=headerlength)
            log_file.write(str('\nHeaderlength:'+str(headerlength)))
            #Drop subsequent double values from dataset
            log_file = open(log_file_location,'a')
            log_file.write('\nDeleting double values\n')
            log_file.close()
            data = data[[xdata,ydata]]
            data = delete_doubles(data,ydata,log_file_location)
            #If less than 5 dataopints available, do not use the file
            if len(data) < 5:
                log_file = open(log_file_location,'a')
                log_file.write(str('\nNot enough datapoints for reliable analysis. Dropping file '+str(file_name)+'\n'))
                continue
            #Check if slopes are calculated with reference to time
            if time_unit == 'sec' or time_unit == 'min' or time_unit == 'hr' or time_unit == 'd':
                log_file = open(log_file_location,'a')
                log_file.write('\nCalculating absolute time\n')
                log_file.close()
                data = add_absolute_time(data,xdata)
                xdata_abs = xdata + '_abs'
            else:
                xdata_abs = xdata + '_abs'
                data[xdata_abs] = data[xdata]

            log_file = open(log_file_location,'a')
            log_file.write(str('\nFiltering based on'+str(filter_function)))
            data = filter_function(data,xdata_abs,ydata,cutoff,time_unit,
                                   log_file_location)

            log_file.write('\nSelecting slope data\n')
            data = _select_slope(data,ydata,down,limits,log_file_location)
            #If less than 5 dataopints available, do not use the file
            if len(data) < 5:
                log_file = open(log_file_location,'a')
                log_file.write(str('\nNot enough datapoints for reliable analysis. Dropping file '+str(file_name)+'\n'))
                continue

            log_file.write('\nCalculating slopes\n')
            #split data in pieces for higher frequency slope calculation
            k = data[xdata_abs].iloc[0]
            while k < data[xdata_abs].iloc[-1]:
                begin = k
                end = k + delta_t
                help_data = data[data[xdata_abs] > begin]
                help_data = help_data[help_data[xdata_abs] < end]
                if len(help_data) < 5:
                    log_file = open(log_file_location,'a')
                    log_file.write(str('\nNot enough datapoints for reliable analysis. Dropping a part from file '+str(file_name)+'\n'))
                    k = end
                    continue
                with_slopes = calc_slopes(help_data,xdata_abs,ydata,time_unit=time_unit)
                slopes_name = ydata+'_slopes'

                slopes_mean.append(with_slopes[slopes_name].mean())
                slopes_std.append(with_slopes[slopes_name].std())
                log_file.write(str('\nAverage slope:'+str(slopes_mean[-1])+'±'+
                               str(slopes_std[-1])))

                timestamp.append(with_slopes[xdata_abs].iloc[-1])
                         #schrikkel=schrikkel)
                k = end

                if plot[0] == True:
                    directory = path+'/figures'
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    log_file.write('\nSaving histogram...\n')
                    fig2, ax2 = plt.subplots()
                    try:
                        with_slopes.hist(slopes_name,ax=ax2,bins=plot[1])
                    except (NameError, ValueError):
                        log_file.write('No datapoints for histogram')
                    t = str(timestamp[-1]).replace(':','.')
                    filename = file_name + t + 'HIST.png'
                    filename = os.path.join(directory,filename)
                    fig2.savefig(filename)

            if plot[0] == True:
                log_file.write('\nSaving dataplot...')
                fig, ax = plt.subplots()
                ax.plot(data[xdata_abs],data[ydata])
                filename = file_name+'DATA.png'
                filename = os.path.join(directory,filename)
                fig.savefig(filename)



#            try:
#                histogram_data = plt.hist(with_slopes[slopes_name],bins=plot[1])
#                bin_centers = [(j+i)/2 for i, j in zip(histogram_data[1][:-1],
#                                                       histogram_data[1][1:])]
#                relative_amount = histogram_data[0] / sum(histogram_data[0])
#                ax_hist.plot(bin_centers,relative_amount,label=timestamp[-1])
#            except (NameError, ValueError):
#                log_file.write('No datapoints for combined histogram')

    dataframe = pd.DataFrame(np.array([slopes_mean,slopes_std]).transpose(),
                             index=timestamp,columns=['mean','std'])

    if plot[0] == True:
        log_file.write('\nSaving combined histogram...')
        filename = 'HIST_ALL.png'
        filename = os.path.join(directory,filename)
        fig_hist.savefig(filename)

        figure, axes = plt_avg_and_std(dataframe['mean'],dataframe['std'],
                                       xax=dataframe.index)#,ylim=[-800,0])
        log_file.write('\nEnd')
        log_file.close()
        return dataframe, figure, axes
    else:
        log_file.write('\nEnd')
        log_file.close()
        return dataframe

print( 'DataAnalysisFcns.py loaded')


def _get_header_length(read_file,ext='text',comment='#'):
    """
    Determines the amount of rows that are part of the header in a file that is
    already opened and readable

    Parameters
    ----------
    read_file : opened file
        an opened file object that is readable
    ext : str
        the extension (in words) of the file the headerlength needs to be found
        for
    comment : str
        comment symbol used in the files

    Returns
    -------
    headerlength : int
        the amount of rows that are part of the header in the read file

    """

    headerlength = 0
    header_test = comment
    counter = 0
    if ext == 'excel' or ext == 'zrx':
        while header_test == comment:
            header_test = str(read_file.sheet_by_index(0).cell_value(counter,0))[0]
            headerlength += 1
            counter +=1

    elif ext == 'text' or ext == 'csv':
        while header_test == comment:
            header_test = read_file.readline()[0]
            headerlength += 1

    return headerlength-1

def _open_file(filepath,ext='text'):
    """
    Opens file of a given extension in readable mode

    Parameters
    ----------
    filepath : str
        the complete path to the file to be opened in read mode
    ext : str
        the extension (in words) of the file that needs to be opened in read
        mode

    Returns
    -------
    The opened file in read mode

    """
    if ext == 'text' or ext == 'zrx' or ext == 'csv':
        return open(filepath, 'r')
    elif ext == 'excel':
        return xlrd.open_workbook(filepath)

def _read_file(filepath,ext='text',skiprows=0):
    """
    Read a file of given extension and save it as a pandas dataframe

    Parameters
    ----------
    filepath : str
        the complete path to the file to be read and saved as dataframe
    ext : str
        the extension (in words) of the file that needs to be read and saved
    skiprows : int
        number of rows to skip when reading a file

    Returns
    -------
    A pandas dataframe containing the data from the given file

    """
    if ext == 'text':
        return pd.read_table(filepath,skiprows=skiprows,decimal='.')
    elif ext == 'excel':
        return pd.read_excel(filepath,skiprows=skiprows)
    elif ext == 'csv':
        return pd.read_csv(filepath,sep='\t',skiprows=skiprows)

def join_dir_files(path,ext='text',comment='#'):
    """
    Reads all files in a given directory, joins them and returns one pd.dataframe

    Parameters
    ----------
    path : str
        the path to the directory containing the files to be put together
    ext : str
        extention of the files to read; possible: excel, text
    comment : str
        comment symbol used in the files

    Returns
    -------
    pd.dataframe:
        pandas dataframe containin concatenated files in the given directory
    """
    #Initialisations
    data = pd.DataFrame()

    #Select files based on extension
    if ext == 'excel':
        files = [f for f in listdir(path) if '.xls' in f]
    elif ext == 'text':
        files = [f for f in listdir(path) if f.endswith('.txt')]
    elif ext == 'csv':
        files = [f for f in listdir(path) if f.endswith('.csv')]
    else:
        print( 'No files with',ext,'extension found in directory',path,'Please \
        choose one of the following: text, excel, csv')

        return None

    #Sort files alphabetically to make sure they are added to each other in the
    #correct order
    files.sort()

    #Read files
    for file_name in files:
        dir_file_path = os.path.join(path,file_name)
        with _open_file(dir_file_path,ext) as read_file:
            headerlength = _get_header_length(read_file,ext,comment)
            data = data.append(_read_file(dir_file_path,ext=ext,
                                          skiprows=headerlength),
                                ignore_index=True)
            print( 'File ',file_name,' has',headerlength,\
            'header lines, adding data to dataframe with columns',data.columns)

    return data


def get_avg(dataframe,name=['none'],plot=False):
    """
    Gets the averages of all or certain columns in a dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe containing the columns to calculate the average for
    name : arary of str
        name(s) of the column(s) containing the data to be averaged; defaults
        to ['none'] and will calculate average for every column
    plot : bool
        if True, plots the calculated mean values, defaults to False

    Returns
    -------
    pd.DataFrame :
        pandas dataframe, containing the average slopes of all or certain
        columns
    """
    if name == ['none']:
        slopes_mean = dataframe.mean()
    else:
        for i in name:
            slopes_mean.append(dataframe[name].mean())

    if plot == True:
        plt.plot(slopes_mean)

    return slopes_mean

def get_std(dataframe,name=['none'],plot=False):
    """
    Gets the standard deviations of all or certain columns in a dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe containing the columns to calculate the standard deviation for
    name : arary of str
        name(s) of the column(s) containing the data to calculate standard
        deviation for; defaults to ['none'] and will calculate standard
        deviation for every column
    plot : bool
        if True, plots the calculated standard deviations, defaults to False

    Returns
    -------
    pd.DataFrame :
        pandas dataframe, containing the average slopes of all or certain
        columns
    """
    if name == ['none']:
        slopes_std = dataframe.std()
    else:
        for i in name:
            slopes_std.append(dataframe[name].std())

    if plot == True:
        plt.plot(slopes_std)

    return slopes_std

def plt_avg_and_std(slopes_mean,slopes_std,xax=[]):#,labels=['Series','Average'],\
                    #figsize=(14,8),ylim=[-100,100]):
    """
    Plots a figure of given datapoints, along with their standard deviation.
    The x-axis can be given or is assumed as default if not entered as argument.

    Parameters
    ----------
    slopes_mean : pd.Series
        series containing the mean values to plot
    slopes_std : pd.Series
        series containing the standard deviations to plot
    xax : pd.Series
        series containing the x-axis values
    labels : array of strings
        array containing the labels to be given to the axes of the figure
    """
    if len(xax) == 0:
        xax = np.arange(0,len(slopes_mean))

    fig, ax = plt.subplots()
    ax.errorbar(xax,slopes_mean,slopes_std)#,linestyle='None',marker='^')
    #plt.xlim(xax[0]-1,xax[-1]+1)
    #ax.set_ylim(ylim)
    #ax.set_xlabel(labels[0])
    #ax.set_ylabel(labels[1])

    return fig, ax

print( 'DataReadingFcns.py loaded' )

def _make_month_day_array(schrikkel=False):
    """
    makes a dataframe containing two columns, one with the number of the month,
    one with the day of the month. Useful in creating datetime objects based on
    e.g. date serial numbers from the Window Date System
    (http://excelsemipro.com/2010/08/date-and-time-calculation-in-excel/)

    Returns
    -------
    pd.DataFrame :
        dataframe with number of the month and number of the day of the month
        for a whole year
    """
    if schrikkel == False:
        days_in_months = [31,28,31,30,31,30,31,31,30,31,30,31]
    elif schrikkel == True:
        days_in_months = [31,29,31,30,31,30,31,31,30,31,30,31]
    days = []
    months = []
    month = 1
    for i in days_in_months:
        for j in range(1,i+1):
            days.append(j)
            months.append(month)
        month += 1

    month_day_array = pd.DataFrame()
    month_day_array['month'] = months
    month_day_array['day'] = days

    return month_day_array

def _get_absolute_time(value,date_type='WindowsDateSystem',data_year=dt.datetime.now().year,
                       schrikkel=False,time_type='None',
                       date_format="%m.%d.%Y %H:%M:%S"):
    """
    Converts a time given in the Windows Date System to the absolute date at
    which the experiment was conducted
    (see also: http://excelsemipro.com/2010/08/date-and-time-calculation-in-excel/)
    """
    leap_years = int((data_year-1900) / 4)

    if date_type == 'WindowsDateSystem':
        #Calculate date
        #year_from_1900 = (int(value) - leap_years) / 365
        day_in_year = (int(value) - leap_years) % 365 - 2
        #decimals = (int(value) - leap_years) / 365. - year_from_1900
        if schrikkel == True & (day_in_year > 59):
            day_in_year = day_in_year - 3#int(366*decimals) - 1
        #elif schrikkel == False:
        #    day_in_year = int(365*decimals) - 1
        months_days = _make_month_day_array(schrikkel=schrikkel)
        month = months_days['month'][day_in_year]
        day_in_month = months_days['day'][day_in_year]

        #Calculate time
        decimals = value - int(value)
        seconds_total = decimals * 86400
        hours = int(seconds_total / (60 * 60))
        minutes = int((seconds_total - (hours * 60 * 60)) / 60)
        seconds = int(seconds_total - hours * 60 * 60 - minutes * 60)

        timestamp = dt.datetime(data_year,month,day_in_month,hours,minutes,seconds)

    elif date_type == 'String':
        timestamp = dt.datetime.strptime(value,date_format)

    return timestamp

def add_absolute_time(dataframe,timedata):
    """
    adds the absolute time to a dataframe based on a given column with time-
    values in a certian coding (default WindowsDateSystem)
    """
    timedata_abs = timedata + '_abs'
    dataframe[timedata_abs] = dataframe[timedata].apply(_get_absolute_time)

    return dataframe

print( 'TimeConversionFcns.py loaded')
