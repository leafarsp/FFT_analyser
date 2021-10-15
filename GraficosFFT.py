import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from scipy.signal import find_peaks
from matplotlib.offsetbox import AnchoredText

import matplotlib.ticker as ticker




def main():
    TCC_path = f'C:\\Users\\leafa\\OneDrive - Católica SC\\Educação\\Engenharia elétrica\\Fase 10\\' \
               f'Trabalho de conclusão de curso\\Ensaios\\Ensaio de vibração 2'
    test_folder = f'\\2 - Vazio - Aceleração'
    acc_file_path_excel = f'{TCC_path}{test_folder}\\Teste 2\\ACC_2021-10-4_17_3_21_a44.xlsx'

    g=9.80665

    fft_file_path_excel_eqp = f'{TCC_path}{test_folder}\\Teste 2\\VT2.csv'
    fft_file_path_excel_placa = f'{TCC_path}{test_folder}\\Teste 2\\FFT_2021-10-4_17_3_29_a45.xlsx'

    df_acc_placa = pd.read_excel(f'{acc_file_path_excel}',index_col=0,sheet_name = 'Sheet1')
    df_fft_eqp = pd.read_csv(f'{fft_file_path_excel_eqp}',index_col=0,header=0)
    df_fft_placa = pd.read_excel(f'{fft_file_path_excel_placa}',index_col=0,sheet_name = 'Sheet1')

    fft_beams = 512
    fft_acc_placa = calculateFFT(df_acc_placa,'ACCz',fft_beams)

    f_res = 1.664595
    f_range = np.linspace(0, (fft_beams / 2) * f_res, int(fft_beams / 2))

    df_fft_placa = pd.DataFrame(data=fft_acc_placa,index=f_range, columns=['FFTz'])
    df_fft_placa.index.name = 'Freq'
    #print(df_fft_placa)
    #exit()

    #df_fft_eqp.columns =('X-Axis', 'Ch1 Y-Axis')
    #print(f'df_fft_eqp.head = {df_fft_eqp.head}')

    max_freq = df_fft_placa['FFTz'].index.max()
    max_ampl_eqp = df_fft_eqp.max().values[0]
    max_ampl_placa = df_fft_placa['FFTz'].max()
    #print(f'max_ampl_eqp = //{max_ampl_eqp}//, max_ampl_placa = {max_ampl_placa}')
    #exit()

    df_fft_placa_pu = df_fft_placa['FFTz'] /  max_ampl_placa

    df_fft_eqp_pu = df_fft_eqp / max_ampl_eqp
    df_fft_eqp_pu_slice = df_fft_eqp_pu[:][0:max_freq]
    #print(df_fft_eqp_pu.loc[0:max_freq,'Ch1 Y-Axis'].to_numpy())
    #print(df_fft_eqp_pu_slice.loc[:,'Ch1 Y-Axis'].to_numpy())
    #exit()

    y_axis_eqp = df_fft_eqp_pu_slice.loc[:,'Ch1 Y-Axis'].to_numpy()
    print(df_fft_placa_pu.index.values)

    y_axis_placa = df_fft_placa_pu.values


    x_axis_eqp = df_fft_eqp_pu_slice.index.values
    x_axis_placa = df_fft_placa_pu.index.values

    peaks_eqp, _ = find_peaks(y_axis_eqp, prominence=0.0005, width=0.2)#, threshold=0.001   )
    peaks_placa, _ = find_peaks(y_axis_placa, prominence=0.0001 , threshold=0.0001)
    #y_axis_peaks[peaks_eqp] = y_axis_eqp[peaks_eqp]

    subplot_rows = 1
    subplot_columns = 1

    fig, ((ax1)) = plt.subplots(subplot_rows, subplot_columns)
    #fig.set_size_inches(10, 100)
    #plt.subplot(subplot_rows,subplot_columns,1)
    # plt.plot(df_fft_eqp_pu[:][0:max_freq])

    #print(x_axis_eqp[peaks_eqp])

    df_peaks_eqp = pd.DataFrame(data = y_axis_eqp[peaks_eqp], index=x_axis_eqp[peaks_eqp],columns=['Amplitude_eqp'])
    df_peaks_eqp.index.name = 'Picos de frequência'

    df_peaks_placa = pd.DataFrame(data=y_axis_placa[peaks_placa], index=x_axis_placa[peaks_placa], columns=['Amplitude_placa'])
    df_peaks_placa.index.name = 'Picos de frequência'




    df_peaks_eqp.to_excel('Freq_peaks_eqp2.xlsx')
    df_peaks_placa.to_excel('Freq_peaks_placa2.xlsx')


    ax1.semilogy(df_fft_eqp_pu[:][0:max_freq],label='Equipamento')
    ax1.semilogy(x_axis_eqp[peaks_eqp], y_axis_eqp[peaks_eqp], "xr")

    ax1.set(title='FFT analisador de vibração')
    ax1.grid(True, which="both", axis='y')

    ax1.semilogy(x_axis_placa[peaks_placa], y_axis_placa[peaks_placa], "ob")



    #plt.subplot(subplot_rows, subplot_columns, 2)
    #plt.plot(df_fft_placa_pu)

    ax1.semilogy(df_fft_placa_pu,label='Placa')
    ax1.set(title='FFT placa')
    ax1.grid(True)
    ax1.legend(loc="upper right")

    fig.tight_layout()
    plt.xticks(np.arange(0, 450, step=30))
    plt.title("Espectro de frequências da velocidade por unidade")
    ax1.set(xlabel='Frequência [Hz]', ylabel = 'Velocidade por unidade [m/s]')
    plt.savefig('Gráfico 2.png', bbox_inches='tight')
    plt.show()


    buttonPressed = False
    while not buttonPressed:
        buttonPressed =plt.waitforbuttonpress()




def calculateFFT (data_frame,column_name,num_beams):
    qt_amostras = (data_frame[column_name]).count()
    fft = np.fft.fft(data_frame[column_name],n=num_beams)
    fft = (np.abs(fft))
    print(len(fft))
    fft = fft[0: int(len(fft) / 2)]
    fft = (fft / (qt_amostras / 4))
    return fft










if __name__ == '__main__':
    main()