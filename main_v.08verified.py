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
    1. Make sure VNA software is reunning
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
from datetime import datetime


def SetParameter(fi,ff,N,SWT,IF,Ptx):
    # Configura los parametros de medicion
    CMT.write('SYST:PRES') # restaura los parámetros predeterminados
    CMT.write("SENS1:SWE:TYPE LIN")
    CMT.write('SENS1:FREQ:STAR {} Hz'.format(fi)) # frecuencia inicial
    CMT.write('SENS1:FREQ:STOP {} Hz'.format(ff)) # frecuencia final
    CMT.write('SENS1:SWE:POIN {}'.format(N)) # número de puntos
    CMT.write('SENS1:SWE:TIME {}'.format(SWT)) # sweep time
    CMT.write('SOUR1:POW:LEV:IMM:AMPL {} dBm'.format(Ptx)) # amplitud de 10 dBm
    CMT.write('SENS1:BWID {} Hz'.format(IF)) # frecuencia intermedia IF
    return

def SetDisplay():
    CMT.write('CALC1:PAR:COUN 8') # 8 Traces
    CMT.write('DISP:WIND:SPL 11') # allocate 8 trace windows

    CMT.write('CALC1:PAR1:DEF S11') #Choose S11 for trace 1
    CMT.write('CALC1:PAR2:DEF S11') #Choose S11 for trace 1
    CMT.write('CALC1:PAR3:DEF S12')
    CMT.write('CALC1:PAR4:DEF S12')
    CMT.write('CALC1:PAR5:DEF S21')
    CMT.write('CALC1:PAR6:DEF S21')
    CMT.write('CALC1:PAR7:DEF S22')
    CMT.write('CALC1:PAR8:DEF S22')

    CMT.write('CALC1:PAR1:SEL')
    CMT.write('CALC1:FORM REAL')

    CMT.write('CALC1:PAR2:SEL')
    CMT.write('CALC1:FORM IMAG')

    CMT.write('CALC1:PAR3:SEL')
    CMT.write('CALC1:FORM REAL')

    CMT.write('CALC1:PAR4:SEL')
    CMT.write('CALC1:FORM IMAG')
    
    
    CMT.write('CALC1:PAR5:SEL')
    CMT.write('CALC1:FORM REAL')

    CMT.write('CALC1:PAR6:SEL')
    CMT.write('CALC1:FORM IMAG')
    
    
    CMT.write('CALC1:PAR7:SEL')
    CMT.write('CALC1:FORM REAL')

    CMT.write('CALC1:PAR8:SEL')
    CMT.write('CALC1:FORM IMAG')
    return 

def SParameter():

    CMT.write('TRIG:SING')

    CMT.write(":INIT:IMM") # Iniciar la medición
    CMT.write("*WAI") # Esperar a que se termine el barrido
    time.sleep(2) # Esperar 0.1 segundos antes de verificar de nuevo

    Freq = CMT.query_ascii_values("SENS1:FREQ:DATA?",container=np.array)

    CMT.write('CALC1:PAR1:SEL')
    S11_r = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)
    
    CMT.write('CALC1:PAR2:SEL')
    S11_i = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)

    CMT.write('CALC1:PAR3:SEL')
    S12_r = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)
    
    CMT.write('CALC1:PAR4:SEL')
    S12_i = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)

    CMT.write('CALC1:PAR5:SEL')
    S21_r = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)
    
    CMT.write('CALC1:PAR6:SEL')
    S21_i = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)

    CMT.write('CALC1:PAR7:SEL')
    S22_r = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array) #Get data as string
    
    CMT.write('CALC1:PAR8:SEL')
    S22_i = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)

    S11_r = S11_r[::2]
    S12_r = S12_r[::2]
    S21_r = S21_r[::2]
    S22_r = S22_r[::2]
    S11_i = S11_i[::2]
    S12_i = S12_i[::2]
    S21_i = S21_i[::2]
    S22_i = S22_i[::2]

    return Freq, S11_r, S11_i, S12_r, S12_i, S21_r, S21_i, S22_r, S22_i

def Makedir(directorio):
    try:
        os.mkdir(directorio)
    except OSError:
        raise ValueError("\n\n-----------------------------\nERROR: \n\nLa creación del directorio %s FALLO !!\n-----------------------------\n\n" % directorio)
    else:
        print("Se ha creado el directorio: %s " % directorio)

def SaveDataCSV(NameFile, Freq, Real, Img):
    data = {'Freq (Hz)': Freq, 'Complex': list(zip(Real, Img))}
    df = pd.DataFrame(data)
    df.to_csv(NameFile + '.csv', index=False)
    print("Save ok   " + NameFile)

def ConnectVNA():
    # Main
    rm = pyvisa.ResourceManager('@py')

    try:
        CMT = rm.open_resource('TCPIP0::127.0.0.1::5025::SOCKET')
        print("\n\n---------------------------------\nConexión correcta con VNA\n---------------------------------\n\n")

    except:
        raise ValueError("\n\n---------------------------------\nERROR: \n\nFALLO LA CONEXIÓN CON EL VNA\n\nComprobar la configuración de red\n---------------------------------\n\n")

    return CMT

def RawDataGeneration(csv_dir: str, output_file_name: str, column_name: str, ensayo: str, fi: float, ff: float, N: int, Ptx: float, IF: float):
    """
    Combina las columnas con el nombre especificado de todos los archivos CSV en el directorio especificado y guarda el resultado en un nuevo archivo CSV.

    Input:
        :param csv_dir: Directorio donde se encuentran los archivos CSV.
        :param output_file_name: Nombre del archivo CSV de salida.
        :param column_name: Nombre de la columna a concatenar.
        :param ensayo: Número del ensayo correspondiente.
        :param fi: Frecuencia inicial de medición (Hz).
        :param ff: Frecuencia final de medición (Hz).
        :param N: Número de puntos de medición.
        :param Ptx: Potencia de transmisión (dBm).
        :param IF: Ancho de banda de FI (Hz).
        
    Output:
        Archivo CSV del RawData del ensayo. Compuesto por los parametros S21 de cada posicion y los paramteros seteados en el VNA.
        
    """
    # Lista para almacenar los DataFrames
    df_list = []

    # Recorrer todos los archivos en el directorio especificado
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            # Leer solo la columna con el nombre especificado del archivo CSV y agregarla a la lista de DataFrames
            file_path = os.path.join(csv_dir, filename)
            df = pd.read_csv(file_path, usecols=[column_name])
            df_list.append(df)

    # Concatenar todos los DataFrames en una sola matriz
    result = pd.concat(df_list, axis=1)

    # Guardar la matriz resultante en un nuevo archivo CSV con la nota adicional
    with open(output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Escribir los datos en el archivo CSV
        writer.writerows(result.values)

        # Agregar la nota adicional al final del archivo CSV como un comentario
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")

        nota = f"El RawData corresponde al ensayo {ensayo} realizado el día {date_time}. \n\nLos parámetros de medición configurados son: \nFrecuencia inicial: {fi} Hz \nFrecuencia final: {ff} Hz \nNúmero de puntos: {N} \nPower: {Ptx} dBm \nIF Bandwidth: {IF} Hz\n\n"
        writer.writerow(['# ' + nota])

    # Mostrar un mensaje de confirmación
    print("--------------------------------------------------")
    print(f"Archivo CSV resultante guardado como '{output_file_name}' con la nota adicional: \n\n{nota}")
    print("--------------------------------------------------\n\n\n")
if __name__ == '__main__':
    # Main del código
    CMT = ConnectVNA()
    
    CMT.read_termination='\n'
    CMT.timeout = 10000
 
    #%% Seteo de parametros
    # Unidades en el sistema internacional
    fi = 5.68e9 
    ff = 6.06e9
    N = 2001
    SWT = 1
    IF = 10e3
    Ptx = 5
    
    #Informar los parametros
    print("\n\n--------------------------------------------------")
    print('Los parametros configurados para la medición son:\n')
    print('Frecuencia inicial:  {:14.0f} KHz'.format(fi/1e3))
    print('Frecuencia final:    {:14.0f} GHz'.format(ff/1e9))
    print('Numero de puntos:    {:14.0f}'.format(N))
    print('Power:               {:14.0f} dBm'.format(Ptx))
    print('IF Bandwidth:        {:14.0f} KHz'.format(IF/1e3))
    print("--------------------------------------------------\n\n\n")

    #Configuración de parámetros
    SetParameter(fi,ff,N,SWT,IF,Ptx)
    
    #%% Configuración del display en S2VNA
    SetDisplay()
    
    #%% Pedir numero de iteracion de medicion
    print("--------------------------------------------------")
    med = str(input('Ingrese el número de la medición:\n'))
    print("--------------------------------------------------\n\n\n")
    
    #%% Se define el nombre de la carpeta o directorio a crear
    print("--------------------------------------------------")
    print("CREACIÓN DE DIRECTORIOS\n")
    directorio = "C:\\Users\\Milena\\OneDrive\\Documents\\focus radar\\GB SAR\\24_05_2023\\Med_"+med
    Makedir(directorio)
    
    dirs21="C:\\Users\\Milena\\OneDrive\\Documents\\focus radar\\GB SAR\\24_05_2023\\Med_"+med+'\\S21(x)'
    Makedir(dirs21)
    
    dirs12="C:\\Users\\Milena\\OneDrive\\Documents\\focus radar\\GB SAR\\24_05_2023\\Med_"+med+'\\S12(x)'
    Makedir(dirs12)
    
    dirs11="C:\\Users\\Milena\\OneDrive\\Documents\\focus radar\\GB SAR\\24_05_2023\\Med_"+med+'\\S11(x)'
    Makedir(dirs11)
    
    dirs22="C:\\Users\\Milena\\OneDrive\\Documents\\focus radar\\GB SAR\\24_05_2023\\Med_"+med+'\\S22(x)'
    Makedir(dirs22)
    print("--------------------------------------------------\n\n\n")
    
    #%% Pedir numero de posiciones
    print("--------------------------------------------------")
    rep = int(input('Ingrese cantidad de posiciones a medir en el eje X:\n'))
    print("--------------------------------------------------\n\n\n")
    
    
    posiciones = []
    
    for i in range(rep):
        # Pedir posición x en que se va a hacer la medición
        print("--------------------------------------------------")
        x = input(f"Ingrese, en centímetros, el valor de la posición {i+1} de X:\n")
        
        posiciones.append(x)  
        
        #Toma de datos
        Freq, S11_r, S11_i, S12_r, S12_i, S21_r, S21_i, S22_r, S22_i= SParameter()
    
    
        #Guardado de archivos
        SaveDataCSV(dirs11+'\\S11_'+ x,Freq,S11_r, S11_i )
        SaveDataCSV(dirs12+'\\S12_'+ x,Freq,S12_r, S12_i)
        SaveDataCSV(dirs21+'\\S21_'+ x,Freq,S21_r, S21_i)
        SaveDataCSV(dirs22+'\\S22_'+ x,Freq,S22_r, S22_i)
        print("--------------------------------------------------\n\n\n")

        #Ploteo de resultados
        plt.figure('R e I')
        plt.subplot(421)
        plt.plot(Freq,S11_r)
        plt.grid()
        
        plt.subplot(422)
        plt.plot(Freq,S11_i)
        plt.grid()
        
        plt.subplot(423)
        plt.plot(Freq,S12_r)
        plt.grid()
        
        plt.subplot(424)
        plt.plot(Freq,S12_i)
        plt.grid()

        plt.subplot(425)
        plt.plot(Freq,S21_r)
        plt.grid()
        
        plt.subplot(426)
        plt.plot(Freq,S21_i)
        plt.grid()
        
        plt.subplot(427)
        plt.plot(Freq,S22_r)
        plt.grid()
        
        plt.subplot(428)
        plt.plot(Freq,S22_i)
        plt.grid()

        plt.figure('Log mag')
        plt.subplot(411)
        plt.plot(Freq,np.abs(S11_r+1j*S11_i))
        plt.grid()
        
        plt.subplot(412)
        plt.plot(Freq,np.abs(S12_r+1j*S12_i))
        plt.grid()
        
        plt.subplot(413)
        plt.plot(Freq,np.abs(S21_r+1j*S21_i))
        plt.grid()
        
        plt.subplot(414)
        plt.plot(Freq,np.abs(S22_r+1j*S22_i))
        plt.grid()
        plt.show()
        
        
    # Nombre del archivo CSV de salida
    output_file = directorio + '/RawData.csv' 
    
    # Nombre de la columna a concatenar
    column_name = 'Complex'      
    
    # Llamar a la función para combinar los archivos CSV
    RawDataGeneration(dirs21, output_file, column_name, med, fi, ff, N, Ptx, IF)  
    
    print("--------------------------------------------------")
    SaveDataCSV(directorio+'\\Posiciones_'+ med,posiciones,[0] * len(posiciones), [0] * len(posiciones),)
    print("--------------------------------------------------")


