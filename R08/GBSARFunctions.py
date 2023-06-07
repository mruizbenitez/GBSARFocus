import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import csv
import datetime
import re

def ConnectVNA(IP,time):
    # Main
    rm = pyvisa.ResourceManager('@py')

    try:
        CMT = rm.open_resource(IP)
        print("\n--------------------------------------------------------")
        print("Conexión correcta con VNA")
        print("--------------------------------------------------------")

    except:
        raise ValueError("\nERROR: FALLO LA CONEXIÓN CON EL VNA. Comprobar la configuración de red")
    
    CMT.read_termination='\n'
    CMT.timeout = time
    
    return CMT

def SetParameter(CMT,fi,ff,N,SWT,IF,Ptx):
    # Configura los parametros de medicion
    CMT.write('SYST:PRES') # restaura los parámetros predeterminados
    CMT.write("SENS1:SWE:TYPE LIN")
    CMT.write('SENS1:FREQ:STAR {} Hz'.format(fi)) # frecuencia inicial
    CMT.write('SENS1:FREQ:STOP {} Hz'.format(ff)) # frecuencia final
    CMT.write('SENS1:SWE:POIN {}'.format(N)) # número de puntos
    CMT.write('SENS1:SWE:TIME {}'.format(SWT)) # sweep time
    CMT.write('SOUR1:POW:LEV:IMM:AMPL {} dBm'.format(Ptx)) # amplitud de 10 dBm
    CMT.write('SENS1:BAND {} Hz'.format(IF)) # frecuencia intermedia IF
    return

def SetDisplay_Pot(CMT):
    CMT.write('CALC1:PAR:COUN 4') # 4 Traces
    CMT.write('DISP:WIND:SPL 8') # allocate 4 trace windows

    CMT.write('CALC1:PAR1:DEF S11') #Choose S11 for trace 1
    CMT.write('CALC1:PAR2:DEF S12')
    CMT.write('CALC1:PAR3:DEF S21')
    CMT.write('CALC1:PAR4:DEF S22')

    CMT.write('CALC1:PAR1:SEL')
    CMT.write('CALC1:FORM MLOG')

    CMT.write('CALC1:PAR2:SEL')
    CMT.write('CALC1:FORM MLOG')

    CMT.write('CALC1:PAR3:SEL')
    CMT.write('CALC1:FORM MLOG')

    CMT.write('CALC1:PAR4:SEL')
    CMT.write('CALC1:FORM MLOG')
    return 

def SetDisplay_ReImg(CMT):
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

def Makedir(Directory):
    if not os.path.exists(Directory):
        os.makedirs(Directory)
        print(f"Se ha creado la carpeta '{Directory}'")
    else:
        print(f"La carpeta '{Directory}' ya existe")

def SParameter_Mag(CMT):

    CMT.write('TRIG:SING')

    CMT.write(":INIT:IMM") # Iniciar la medición
    CMT.write("*WAI") # Esperar a que se termine el barrido
    time.sleep(2) # Esperar 0.1 segundos antes de verificar de nuevo

    Freq = CMT.query_ascii_values("SENS1:FREQ:DATA?",container=np.array)

    CMT.write('CALC1:PAR1:SEL')
    S11 = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)

    CMT.write('CALC1:PAR2:SEL')
    S12 = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)

    CMT.write('CALC1:PAR3:SEL')
    S21 = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array)

    CMT.write('CALC1:PAR4:SEL')
    S22 = CMT.query_ascii_values("CALC1:DATA:FDAT?", container=np.array) #Get data as string

    S11 = S11[::2]; S12 = S12[::2]; S21 = S21[::2]; S22 = S22[::2]

    return Freq, S11, S12, S21, S22

def SParameter_ReImg(CMT):

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

    S11 = S11_r+1j*S11_i
    S12 = S12_r+1j*S12_i
    S21 = S21_r+1j*S21_i
    S22 = S22_r+1j*S22_i


    return Freq, S11, S12, S21, S22

def ParameterPrint(fi,ff,Nf,Ptx,IF):
    print("\n-------------- SOFTWARE GB-SAR C BAND ----------------")
    print('Los parametros configurados para la medición son:')
    print(f'Frecuencia inicial                  : {fi/1e9} GHz')
    print(f'Frecuencia final                    : {ff/1e9} GHz')
    print(f'Numero de puntos en frecuencia Nf   : {Nf}')
    print(f'Potencia transmitida                : {Ptx} dBm')
    print(f'IF Bandwidth                        : {IF/1e3} KHz')
    return

def RDPlot(Ski):
    Real_Ski = np.real(Ski)
    Phase_Ski = np.angle(Ski)
    plt.figure("RawData")
    plt.subplot(121)
    
    # Crear el mapa de calor
    plt.imshow(Real_Ski, cmap='hot', vmin=-1, vmax=+1)

    # Agregar una barra de color para indicar los valores
    plt.colorbar()

    # Añadir etiquetas a los ejes x e y
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')

    plt.subplot(122)
    # Crear el mapa de calor
    plt.imshow(Phase_Ski, cmap='hot', vmin=-np.pi, vmax=+np.pi)

    # Agregar una barra de color para indicar los valores
    plt.colorbar()

    # Añadir etiquetas a los ejes x e y
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')


    # Mostrar el mapa de calor
    plt.show()

def MagPlot(Freq,S11,S12,S21,S22,position):
    #Ploteo de resultados
    plt.figure(f"Measurement to {position} cm")
    plt.subplot(221)
    plt.plot(Freq,S11)
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (dBm)")
    plt.grid()
    
    plt.subplot(222)
    plt.plot(Freq,S12)
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (dBm)")
    plt.grid()
    
    plt.subplot(223)
    plt.plot(Freq,S21)
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (dBm)")
    plt.grid()
    
    plt.subplot(224)
    plt.plot(Freq,S22)
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (dBm)")
    plt.grid()
    plt.show()

def ReImgPlot(Freq,S11,S12,S21,S22,position):
    #Ploteo de resultados
    plt.figure(f"Measurement to {position} cm")
    plt.subplot(421)
    plt.plot(Freq,np.real(S11))
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (mW)")
    plt.grid()
    
    plt.subplot(422)
    plt.plot(Freq,np.imag(S11))
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (mW)")
    plt.grid()
    
    plt.subplot(423)
    plt.plot(Freq,np.real(S12))
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (mW)")
    plt.grid()
    
    plt.subplot(424)
    plt.plot(Freq,np.imag(S12))
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (mW)")
    plt.grid()

    plt.subplot(425)
    plt.plot(Freq,np.real(S21))
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (mW)")
    plt.grid()
    
    plt.subplot(426)
    plt.plot(Freq,np.imag(S21))
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (mW)")
    plt.grid()

    plt.subplot(427)
    plt.plot(Freq,np.real(S22))
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (mW)")
    plt.grid()
    
    plt.subplot(428)
    plt.plot(Freq,np.imag(S22))
    plt.xlabel("frecuency (Hz)")
    plt.ylabel("Power (mW)")
    plt.grid()
    
    plt.show()

def SaveFrameCSV(Data,File="FrameDefault.csv"):
    Frame = pd.DataFrame(Data)
    Frame.to_csv(File, header=False,index=False)
    print(f"File {File}, save. ")
    return

def ReadCSVFrame(File="FrameDefault.csv"):
    Frame = pd.read_csv(File,header=None)
    Frame = Frame.to_numpy().astype(complex)
    print(f"File {File}, read. ")
    return Frame

def RawDataGeneration(MainDirectory,Positions):
    
    i = 1
    ruta = os.path.abspath(MainDirectory)

    for position in Positions:
        foldername = f"Measurement_{position}cm"
        folderpath = os.path.join(ruta, foldername)
        filename = f"S21_{position}cm.csv"
        filepath = os.path.join(folderpath, filename)
        data = ReadCSVFrame(filepath)
        if i == 1:
            RawData = np.empty((0, data.shape[0])); i = 0
        RawData = np.vstack((RawData, data[:, 1]))

    print(f"RawData generado")
    SaveFrameCSV(RawData,File=MainDirectory+"/RawData.csv")
    
    return RawData






