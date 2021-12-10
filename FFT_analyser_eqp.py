import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from datetime import datetime


#Este arquivo é um teste realizado para tentar gerar um gráfico do espectro de frequencia a partir dos dados
#de aceleração do analisador de vibração. O ideal era que fosse possível gerar o mesmo gráfico que o equipamento
#gera, no entanto esse objetovo não foi alcançado.

def main():


    TCC_path = f'C:\\Users\\leafa\\OneDrive - Católica SC\\Educação\\Engenharia elétrica\\Fase 10\\' \
               f'Trabalho de conclusão de curso\\Ensaios\\Ensaio de vibração 2'
    test_folder = f'\\1 - Vazio - Velocidade'
    acc_file_path_excel = f'{TCC_path}{test_folder}\\Teste 1\\VT1_T.csv'

    g=9.80665

    fft_file_path_excel = f'{TCC_path}{test_folder}\\Teste 1\\VT1.csv'

    df_acc = pd.read_csv(f'{acc_file_path_excel}',index_col=0)

    df_fft = pd.read_csv(f'{fft_file_path_excel}',index_col=0)

    #df_fft.convert_objects(convert_numeric=True)



    df_acc['Ch1 Y-Axis'] = df_acc['Ch1 Y-Axis']*g
    df_acc['Ch1 Y-Axis'] = apply_moving_average(df_acc,'Ch1 Y-Axis',5  )

    qt_amostras = (df_acc['Ch1 Y-Axis']).count()
    #T_amostragem = max(df_acc['X-Axis'])
    T_amostragem = df_acc.index.max()

    fft_beams = 3200

    fft_acc_z_python_mod = calculateFFT(df_acc,'Ch1 Y-Axis',fft_beams)


    f_res = (qt_amostras / T_amostragem) / fft_beams
    print (f'frequency resolution = {f_res}Hz')


    f_range = np.linspace(0,(len(fft_acc_z_python_mod))*f_res,int(len(fft_acc_z_python_mod)))

    #inv_fft_z = np.fft.ifft(fft_acc_z_python)

    fft_acc_z_placa = df_fft['Ch1 Y-Axis']
    #print(df_fft)
    acc_z = df_acc.index
    df_speed_x = integrate_acc_numerically(df_acc['Ch1 Y-Axis'], 'Ch1 Y-Axis', 'Speed_z')
    df_speed_x = eliminate_DC_level(df_speed_x,'Speed_z' )

    fft_speed_z_eqp = calculateFFT(df_speed_x,'Speed_z',fft_beams)



    subplot_rows = 2
    subplot_columns = 3

    plt.subplot(subplot_rows,subplot_columns,1)
    plt.plot(df_acc)
    #plt.plot(inv_fft_z)

    plt.subplot(subplot_rows,subplot_columns,2)

    plt.plot(fft_acc_z_placa)



    plt.subplot(subplot_rows,subplot_columns,3)
    plt.plot(f_range,fft_acc_z_python_mod)


    plt.subplot(subplot_rows,subplot_columns,4)

    plt.plot(df_speed_x['Speed_z'])
    plt.subplot(subplot_rows,subplot_columns,5)

    plt.plot(f_range, fft_speed_z_eqp )

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


def calculateFFT (data_frame,column_name,num_beams):
    qt_amostras = (data_frame[column_name]).count()
    fft = np.fft.fft(data_frame[column_name],n=num_beams)
    fft = (np.abs(fft))
    print(len(fft))
    fft = fft[0: int(len(fft) / 2)]
    fft = (fft / (qt_amostras / 4))
    return fft

def eliminate_DC_level (data_frame,column_name):
    mean = data_frame[column_name].mean()
    data_frame[column_name] = data_frame[column_name] - mean
    return data_frame

def apply_moving_average(data_frame,column_name, num_avgs):
    #ret_val=data_frame[column_name].rolling(num_avgs).mean()
    ret_val = data_frame[column_name].ewm(span=num_avgs, adjust=False).mean()

    ret_val.iloc[0:num_avgs-1] = ret_val.iloc[num_avgs]
    print(ret_val.iloc[0:10])
    return ret_val









if __name__ == '__main__':
    main()