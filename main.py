import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from datetime import datetime




def main():
    acc_file_path_excel = f'C:\\Users\\leafa\\OneDrive\\Área de Trabalho\\TEMP\\ACC\\ACC7.xlsx'
    fft_file_path_excel = f'C:\\Users\\leafa\\OneDrive\\Área de Trabalho\\TEMP\\FFT\\FFT7.xlsx'
    df_acc = pd.read_excel(f'{acc_file_path_excel}','Sheet1',index_col=0)
    df_fft = pd.read_excel(f'{fft_file_path_excel}','Sheet1',index_col=0)

    print((df_acc['Sample']).count())

    qt_amostras = (df_acc['Sample']).count()
    T_amostragem = max(df_acc['Sample'])

    fft_beams = 256

    fft_acc_z_python = np.fft.fft(df_acc['ACCx'],n=fft_beams)
    #print(fft_acc_z_python)
    #exit(1)

    fft_acc_z_python_mod = np.abs(fft_acc_z_python)
    fft_acc_z_python_mod = fft_acc_z_python_mod[0: int(len(fft_acc_z_python_mod) / 2)]
    fft_acc_z_python_mod = fft_acc_z_python_mod/(qt_amostras/4)

    f_res = (qt_amostras / T_amostragem) / fft_beams
    print (f'frequency resolution = {f_res}Hz')

    f_range = np.linspace(0,(fft_beams/2)*f_res,int(fft_beams/2))
    print(len(f_range))


    #fft_acc_z_python_mod = fft_acc_z_python_mod[0]*

    inv_fft_z = np.fft.ifft(fft_acc_z_python)

    fft_acc_z_placa = df_fft['FFTx']
    acc_z = df_acc['ACCx']

    subplot_rows = 5
    subplot_columns = 3


    plt.subplot(subplot_rows,subplot_columns,1)
    plt.plot(acc_z)

    plt.subplot(subplot_rows,subplot_columns,2)
    plt.plot(fft_acc_z_placa)

    plt.subplot(subplot_rows,subplot_columns,3)
    plt.plot(f_range,fft_acc_z_python_mod)

    plt.subplot(subplot_rows,subplot_columns,4)
    plt.plot(inv_fft_z)

    df_speed_x = integrate_acc_numerically(df_acc['teste'],'teste','Speed_x')
    plt.subplot(subplot_rows,subplot_columns,5)
    plt.plot(df_speed_x['Speed_x'])

    plt.show()

    buttonPressed = False
    while not buttonPressed:
        buttonPressed =plt.waitforbuttonpress()



def integrate_acc_numerically(df, eixo_acc, eixo_speed):
    ret_df = pd.DataFrame(data = None,index = df.index, columns = ['t_s',eixo_speed])
    ret_df.iloc[0][1]=0

    for i in range(1, df.index.size):
        #print(f'i={i} df.iloc[i]={df.iloc[i]} df.iloc[i-1]={df.iloc[i-1]} df.index[i]={df.index[i]} '
              #f'df.index[i-1]={df.index[i-1]} ret_df.iloc[i][1]={ret_df.iloc[i-1][1]}')
        ret_df.iloc[i] = (((df.iloc[i]+df.iloc[i-1])/2) * (df.index[i]-df.index[i-1]))*1000. + ret_df.iloc[i-1][1]

    ret_df.iloc[0][1] = ret_df.iloc[1][1]
    ret_df[eixo_speed] = ret_df[eixo_speed] - ret_df[eixo_speed].mean()
    return ret_df


















if __name__ == '__main__':
    main()