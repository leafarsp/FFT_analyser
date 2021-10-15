import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from scipy.signal import find_peaks
from matplotlib.offsetbox import AnchoredText

import matplotlib.ticker as ticker




def main():
    user = 'rafaelb1'
    teste = '4'
    TCC_path = f'C:\\Users\\{user}\\OneDrive - Católica SC\\Educação\\Engenharia elétrica\\Fase 10\\' \
               f'Trabalho de conclusão de curso\\Ensaios\\Ensaio de vibração 2'
    test_folder = f'\\1 - Vazio - Velocidade'

    fft_file_path_excel_eqp1 = f'{TCC_path}{test_folder}\\Teste 1\\VT1.csv'
    df_fft_eqp1 = pd.read_csv(f'{fft_file_path_excel_eqp1}', index_col=0, header=0)
    df_fft_eqp1.rename(columns={'Ch1 Y-Axis':'Teste1'},inplace=True)
    df_fft_eqp1.index.name='Freq'

    fft_file_path_excel_eqp2 = f'{TCC_path}{test_folder}\\Teste 5\\VT5.csv'
    df_fft_eqp2 = pd.read_csv(f'{fft_file_path_excel_eqp2}', index_col=0, header=0)
    df_fft_eqp2.rename(columns={'Ch1 Y-Axis': 'Teste2'},inplace=True)
    df_fft_eqp2.index.name = 'Freq'

    fft_file_path_excel_eqp3 = f'{TCC_path}{test_folder}\\Teste 9\\VT9.csv'
    df_fft_eqp3 = pd.read_csv(f'{fft_file_path_excel_eqp3}', index_col=0, header=0)
    df_fft_eqp3.rename(columns={'Ch1 Y-Axis': 'Teste3'},inplace=True)
    df_fft_eqp3.index.name = 'Freq'

    df_fft_eqp1['Teste2'] = df_fft_eqp2['Teste2']
    df_fft_eqp1['Teste3'] = df_fft_eqp3['Teste3']
    #df_fft_eqp = df_fft_eqp1.merge(df_fft_eqp2).merge(df_fft_eqp3)

    print(df_fft_eqp1)
    fig, ((ax1)) = plt.subplots(1, 1)

    ax1.semilogy(df_fft_eqp1, label={'Teste1','Teste2','Teste3'})
    ax1.grid(True, which="both", axis='both')
    ax1.legend(loc="upper right")
    ax1.set(title='Espectro de frequências da velocidade\nVerificação da repetibilidade entre os testes')
    #ax1.title("Espectro de frequências da velocidade\nVerificação da repetibilidade entre os testes")
    ax1.set(xlabel='Frequência [Hz]', ylabel='Velocidade [m/s]')
    fig.savefig(f'Gráfico sobrepostos equipamento .png', bbox_inches='tight')
    fig.show()















    fft_file_path_excel_placa1 = f'{TCC_path}{test_folder}\\Teste 1\\FFT_2021-10-4_17_1_33_a31.xlsx'
    df_fft_placa1 = pd.read_excel(f'{fft_file_path_excel_placa1}', index_col=0, header=0)
    df_fft_placa1 = pd.DataFrame(data=df_fft_placa1['FFTz'])
    df_fft_placa1.index.name = 'Freq'
    df_fft_placa1.rename(columns={'FFTz':'Teste1'},inplace=True)




    fft_file_path_excel_placa2 = f'{TCC_path}{test_folder}\\Teste 5\\FFT_2021-10-4_17_9_58_a91.xlsx'
    df_fft_placa2 = pd.read_excel(f'{fft_file_path_excel_placa2}', index_col=0, header=0)
    df_fft_placa2 = pd.DataFrame(data=df_fft_placa2['FFTz'])
    df_fft_placa2.index.name = 'Freq'
    df_fft_placa2.rename(columns={'FFTz': 'Teste2'}, inplace=True)

    fft_file_path_excel_placa3 = f'{TCC_path}{test_folder}\\Teste 9\\FFT_2021-10-4_17_17_57_a147.xlsx'
    df_fft_placa3 = pd.read_excel(f'{fft_file_path_excel_placa3}', index_col=0, header=0)
    df_fft_placa3 = pd.DataFrame(data=df_fft_placa3['FFTz'])
    df_fft_placa3.index.name = 'Freq'
    df_fft_placa3.rename(columns={'FFTz': 'Teste3'}, inplace=True)

    df_fft_placa1['Teste2'] = df_fft_placa2['Teste2']
    df_fft_placa1['Teste3'] = df_fft_placa3['Teste3']

    print(df_fft_placa1)

    fig2, ((ax2)) = plt.subplots(1, 1)

    ax2.semilogy(df_fft_placa1, label={'Teste1', 'Teste2', 'Teste3'})
    ax2.grid(True, which="both", axis='both')
    ax2.legend(loc="upper right")
    ax2.set(title='Espectro de frequências da velocidade\nVerificação da repetibilidade entre os testes')
    # ax1.title("Espectro de frequências da velocidade\nVerificação da repetibilidade entre os testes")
    ax2.set(xlabel='Frequência [Hz]', ylabel='Velocidade [m/s]')
    fig2.savefig(f'Gráfico sobrepostos placa .png', bbox_inches='tight')
    fig2.show()



    buttonPressed = False
    while not buttonPressed:
        buttonPressed = plt.waitforbuttonpress()
if __name__ == '__main__':
    main()