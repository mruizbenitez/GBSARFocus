#medicion s21 automatizada
# -*- coding: utf-8 -*-
"""
Created 5/21/2021
Modified 10/24/2022

@author: CMT

OVERVIEW:
    This script allocates 2 traces of S11 and captures log mag and phase 
    measurement data into arrays of numbers. Eventually printing them
    
    The VNA software has to be running always when automating the VNA. The VNA 
    software is the driver for the VNA

Before running the code:
    1. Make sure VNA software is running
    2. S2,S4VNA:
       Go to system -> Misc setup -> network remote settings -> turn on 
       socket server
       R,TRVNA:
       Go to system -> network setup -> interface state (on)
    3. make sure socket server is 5025

Additional information:
    1. The SCPI programming manual is intalled with the VNA software. By default:
        C:\VNA\RVNA or TRVNA or S2VNA or S4VNA\Doc
"""

import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import csv
import datetime
from GBSARFunctions import *


if __name__ == '__main__':

    # Definición de parámetros de configuración del equipo, SI
    B       = 384e6
    fc      = 5.87e9
    fi      = fc-B/2 
    ff      = fc+B/2
    N       = 2001
    SWT     = 1
    IF      = 10e3
    Ptx     = 5
    
    #Imprime los parámetros de configuración
    ParameterPrint(fi,ff,N,Ptx,IF)

    #Realiza la conexión con el equipo de medición
    exec_time = 10000                       #tiempo de ejecución de 10s
    
    #IP = 'TCPIP::169.254.214.136::hislip0' # VNA R&S
    IP = 'TCPIP0::127.0.0.1::5025::SOCKET'  # VNA M5065

    CMT = ConnectVNA(IP, exec_time)     #Genera el objeto CMT, del equipo

    #Configuración de parámetros del equipo
    SetParameter(CMT,fi,ff,N,SWT,IF,Ptx)

    #Configuración del display del equipo
    SetDisplay_ReImg(CMT)

    print("\n--------------------------------------------------------")
    FolderName = str(input('Ingrese el nombre del directorio: '))
    #Crea el directorio donde se almacenaran los datos del ensayo
    date = str(datetime.date.today())
    main_directory = FolderName+'_'+date
    Makedir(main_directory)
    print("--------------------------------------------------------")


    #directorio = "C:\\Users\\Milena\\OneDrive\\Documents\\focus radar\\GB SAR\\24_05_2023\\Med_"+med
    #Makedir(directorio)


    #Solicita las posiciones que tendrá el radar sobre su rail
    print("\n--------------------------------------------------------")
    Np = int(input('Ingrese la cantidad de posiciones Np a medir en el eje X:'))
    print("--------------------------------------------------------")

    positions = []
    for i in range(Np):

        #Solicita la posición en cm del radar respecto al de ref. la posición 1 es la ref
        print("\n--------------------------------------------------------")
        position = float(input(f"Ingrese la posición {i+1} del radar en el eje x, en cm:"))
        positions.append(position)         #Guarda la la posición, SI 

        #Crea el subdirectorio donde se almacenaran las mediciones de la posicion i
        directory = main_directory+f"/Measurement_{str(position)}cm"
        Makedir(directory)
        
        #Recopila datos de los parámetros S del equipo CMT
        #Freq, S11, S12, S21, S22 = SParameter_Mag(CMT)
        Freq, S11, S12, S21, S22 = SParameter_ReImg(CMT)

        ReImgPlot(Freq,S11,S12,S21,S22,position)

        #Guarda cada parámetro en el subdirectorio de la medicion de la posicion i
        SaveFrameCSV(np.transpose(np.vstack((Freq,S11))),directory+'/S11_'+str(position)+'cm.csv')
        SaveFrameCSV(np.transpose(np.vstack((Freq,S12))),directory+'/S12_'+str(position)+'cm.csv')
        SaveFrameCSV(np.transpose(np.vstack((Freq,S21))),directory+'/S21_'+str(position)+'cm.csv')
        SaveFrameCSV(np.transpose(np.vstack((Freq,S22))),directory+'/S22_'+str(position)+'cm.csv')
        print(f"Medición {i+1} finalizada")
        print("--------------------------------------------------------")

    
    #Solicita la posición en cm del radar respecto al de ref. la posición 1 es la ref
    print("\n--------------------------------------------------------")
    descripcion = str(input(f"Ingrese una descripción de la medición realizada. Indique la escena que se tomó y si hubo incovenientes en algún paso:"))
    metadata = str(datetime.datetime.now())

    print("\n--------------------------------------------------------")
    txt_descripcion = save_txt(descripcion+'\n' +metadata, main_directory+'_'+FolderName)
    print("--------------------------------------------------------")

    print("\n--------------------------------------------------------")
    RawData = RawDataGeneration(main_directory,np.sort(positions))
    print(type(RawData))
    print(RawData.shape)
    print("--------------------------------------------------------")


