# -*- coding: utf-8 -*-
"""
ReadWriteFile.py
Reads and writes different types of files

    Read a CSV file
    colHeaders, rowHeaders, data = readCSV(filename)

Created on Thu Feb 14 16:08:51 2019

@author: ranquist
"""



"""
colHeaders, rowHeaders, data = readCSV(filename)
Read a CSV file

Input:
    filename - name of file.  (If file not in work directory, must provide
                               directory path)
    
Output:
    [column headers, row header, data]:
        column headers - string array of column headers
        row headers - string array of row headers
        data - string 2D array of data

Note:
    First element in column headers and row headers will be the same

@author: Drake Ranquist
"""
def readCSV(filename):
    
    import csv
    import numpy as np
    
    """
    Extract Zodiacal Light Data from Hubble CSV file
    """
    csvList = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            csvList.append(row)
    
    #Seperate Headings from data
    arr = np.array(csvList)
    colHeaders = arr[:,0]
    rowHeaders = arr[0,:]
    data = arr[1:,1:]

    return [colHeaders, rowHeaders, data]