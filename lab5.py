import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import pywt # Importar PyWavelets



file_path = "DATA_ECG/ECG_lab5.csv" 
try:
    df = pd.read_csv(file_path)
    tiempo = df.iloc[:, 0].values  
    voltaje = df.iloc[:, 1].values  
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta '{file_path}'. Verifica la ubicación.")
    print("ADVERTENCIA: Usando datos simulados porque no se encontró el archivo.")
    fs_sim = 250 
    duration_sim = 300 
    tiempo = np.linspace(0, duration_sim, int(fs_sim * duration_sim), endpoint=False)
    import heartpy as hp
    voltaje, _ = hp.synth.synthesize(duration=duration_sim, sampling_rate=fs_sim, hr=65)
    # --- FIN SIMULACIÓN ---
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
    import sys
    sys.exit()


# Estimar la frecuencia de muestreo (fs) desde el archivo de tiempo
# ¡¡¡ ADVERTENCIA !!! El cálculo original dio ~12.77 Hz, lo cual es MUY bajo para ECG.
# Verifica tu columna de tiempo. Usaremos este valor bajo pero los resultados no serán fiables.
if 'fs_sim' not in locals(): # Si no estamos usando datos simulados
    fs_estimates = 1 / np.diff(tiempo)
    fs_mean = np.nanmedian(fs_estimates) # Usar mediana es más robusto a outliers que la media
    print(f"Frecuencia de muestreo estimada (mediana): {fs_mean:.2f} Hz")
    # Validar si la fs estimada es razonable
    if fs_mean < 50: # Umbral arbitrario, ajustar según necesidad
         print("\n*** ADVERTENCIA SEVERA: La frecuencia de muestreo estimada es MUY BAJA (<50 Hz). ***")
         print("*** Los resultados del análisis (filtros, picos, HRV, wavelet) NO serán fiables. ***")
         print("*** Por favor, verifica los datos de tiempo en tu archivo CSV o la configuración de adquisición. ***\n")
         # Considera detener la ejecución o usar una fs asumida si estás seguro que la calculada es incorrecta
         # fs_mean = 250 # Ejemplo: Forzar a 250 Hz si sabes que ese era el valor correcto
         # print(f"*** USANDO fs = {fs_mean} Hz ASUMIDA para continuar análisis. ***\n")
else:
    fs_mean = fs_sim # Usar la fs de los datos simulados
    print(f"Usando frecuencia de muestreo simulada: {fs_mean:.2f} Hz")

fs = fs_mean # Frecuencia de muestreo a usar en el resto del código

# Graficar la señal original
plt.figure(figsize=(15, 4))
plt.plot(tiempo, voltaje, label="Señal ECG Original", color="b")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud / Voltaje")
plt.title("Señal ECG Original (5 min)")
plt.legend()
plt.grid(True)
plt.show()


# --- Punto 9c: Pre-procesamiento de la señal ---

print("\n--- Iniciando Punto 9c: Pre-procesamiento ---")

# 1. Filtrado IIR (Butterworth como ejemplo)
nyquist = 0.5 * fs
order = 4 # Orden del filtro

# Filtro Pasa-Altas (remover deriva línea base)
lowcut = 0.5  # Hz
# Asegurarse que lowcut < nyquist
if lowcut >= nyquist:
    print(f"Advertencia: Frecuencia de corte Pasa-Altas ({lowcut} Hz) es mayor o igual a Nyquist ({nyquist} Hz). Omitiendo filtro Pasa-Altas.")
    ecg_filtered_hp = voltaje # No aplicar filtro
else:
    low = lowcut / nyquist
    b_hp, a_hp = butter(order, low, btype='highpass')
    ecg_filtered_hp = filtfilt(b_hp, a_hp, voltaje) # filtfilt evita desfase
    print(f"Aplicado Filtro Pasa-Altas > {lowcut} Hz")

# Filtro Pasa-Bajas (remover ruido alta frecuencia)
highcut = 40.0 # Hz - Puede ser necesario reducirlo si fs es muy baja
# Asegurarse que highcut < nyquist
if highcut >= nyquist:
     print(f"Advertencia: Frecuencia de corte Pasa-Bajas ({highcut} Hz) es mayor o igual a Nyquist ({nyquist} Hz). Ajustando a Nyquist*0.99.")
     highcut = nyquist * 0.99 # Ajustar para evitar error

high = highcut / nyquist
b_lp, a_lp = butter(order, high, btype='lowpass')
ecg_filtered_hplp = filtfilt(b_lp, a_lp, ecg_filtered_hp)
print(f"Aplicado Filtro Pasa-Bajas < {highcut:.2f} Hz")

# Filtro Notch (opcional, si hay ruido de red eléctrica - 50 o 60 Hz)
# ¡¡¡ Con fs=12.77 Hz, Nyquist es ~6.38 Hz, NO SE PUEDE aplicar Notch a 50/60 Hz !!!
f0_notch = 60.0 # Hz (Ajustar a 50 Hz si es necesario)
if f0_notch < nyquist:
    Q = 30.0       # Factor de calidad
    b_notch, a_notch = iirnotch(f0_notch, Q, fs)
    ecg_filtered = filtfilt(b_notch, a_notch, ecg_filtered_hplp)
    print(f"Aplicado Filtro Notch @ {f0_notch} Hz")
    # Ecuación en diferencias (ejemplo para Notch):
    # y[n] = (b[0]/a[0])*x[n] + (b[1]/a[0])*x[n-1] + (b[2]/a[0])*x[n-2] - (a[1]/a[0])*y[n-1] - (a[2]/a[0])*y[n-2]
    print("  Coeficientes Notch (b):", b_notch)
    print("  Coeficientes Notch (a):", a_notch)
else:
    print(f"Omitiendo Filtro Notch @ {f0_notch} Hz (Frecuencia > Nyquist)")
    ecg_filtered = ecg_filtered_hplp # Usar la señal sin Notch

# Visualización de la señal filtrada
plt.figure(figsize=(15, 4))
plt.plot(tiempo, ecg_filtered, label="Señal ECG Filtrada", color="g")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud / Voltaje")
plt.title("Señal ECG Filtrada")
plt.legend()
plt.grid(True)
plt.show()

# 2. Identificación de Picos R
# Ajustar 'height' y 'distance' es CRUCIAL, especialmente con fs baja.
# Distance: Mínima separación entre picos. Si HR max=180 bpm -> 0.33s
min_hr_bpm = 40
max_hr_bpm = 180
min_distance_sec = 60.0 / max_hr_bpm
min_distance_samples = int(min_distance_sec * fs)
if min_distance_samples < 1: min_distance_samples = 1 # Asegurar distancia mínima de 1

# Umbral (height): puede requerir ajuste manual o métodos adaptativos
# Empezamos con un umbral basado en la desviación estándar
peak_height_threshold = np.mean(ecg_filtered) + 0.6 * np.std(ecg_filtered)
# Asegurarse que el umbral no sea demasiado bajo si la señal tiene offset negativo
if peak_height_threshold < np.percentile(ecg_filtered, 75): # Heurística simple
     peak_height_threshold = np.percentile(ecg_filtered, 75)

print(f"Detectando picos R con altura > {peak_height_threshold:.3f} y distancia mínima > {min_distance_samples} muestras ({min_distance_sec:.2f} s)")

peaks_indices, properties = find_peaks(ecg_filtered, height=peak_height_threshold, distance=min_distance_samples)

print(f"Número de picos R detectados: {len(peaks_indices)}")
if len(peaks_indices) < 10: # Chequeo básico de si se detectaron suficientes picos
    print("*** ADVERTENCIA: Muy pocos picos R detectados. Revisa los parámetros de find_peaks (height, distance) y la calidad de la señal filtrada. ***")

# Visualización de picos detectados
plt.figure(figsize=(15, 4))
plt.plot(tiempo, ecg_filtered, label='ECG Filtrada')
if len(peaks_indices) > 0:
    plt.plot(tiempo[peaks_indices], ecg_filtered[peaks_indices], 'ro', label='Picos R Detectados')
else:
    print("No se detectaron picos R con los parámetros actuales.")
plt.title('Detección de Picos R')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud / Voltaje')
plt.legend()
plt.grid(True)
plt.show()

# 3. Cálculo de Intervalos R-R (Necesitamos al menos 2 picos)
if len(peaks_indices) > 1:
    rr_intervals_samples = np.diff(peaks_indices)
    rr_intervals_sec = rr_intervals_samples / fs # Convertir a segundos
    # Tiempo correspondiente a cada intervalo RR (usualmente al final del intervalo)
    rr_times_sec = tiempo[peaks_indices[1:]]

    # Eliminar intervalos RR irrealistas (artefactos o detecciones falsas)
    # Criterio simple: basado en rangos de HR fisiológicos
    min_rr_sec = 60.0 / max_hr_bpm # e.g., 0.333s para 180bpm
    max_rr_sec = 60.0 / min_hr_bpm # e.g., 1.5s para 40bpm
    original_rr_count = len(rr_intervals_sec)
    mask_rr = (rr_intervals_sec >= min_rr_sec) & (rr_intervals_sec <= max_rr_sec)
    rr_intervals_sec = rr_intervals_sec[mask_rr]
    rr_times_sec = rr_times_sec[mask_rr]
    removed_rr_count = original_rr_count - len(rr_intervals_sec)
    if removed_rr_count > 0:
        print(f"Eliminados {removed_rr_count} intervalos RR fuera del rango [{min_rr_sec*1000:.0f} ms, {max_rr_sec*1000:.0f} ms]")

    if len(rr_intervals_sec) > 1:
        # Visualización de la serie de intervalos R-R (Tacograma)
        plt.figure(figsize=(12, 5))
        plt.plot(rr_times_sec, rr_intervals_sec * 1000, marker='o', linestyle='-', label='Intervalos R-R')
        plt.title('Serie de Intervalos R-R (Tacograma)')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Intervalo R-R (ms)')
        plt.grid(True)
        plt.show()
    else:
        print("No hay suficientes intervalos RR válidos para graficar el tacograma.")
        rr_intervals_sec = np.array([]) # Vaciar para evitar errores posteriores
else:
    print("No se detectaron suficientes picos R para calcular intervalos RR.")
    rr_intervals_sec = np.array([]) # Vaciar para evitar errores posteriores

# --- Punto 9d: Análisis de HRV en el dominio del tiempo ---
print("\n--- Iniciando Punto 9d: Análisis Dominio del Tiempo ---")

if len(rr_intervals_sec) > 1:
    mean_rr = np.mean(rr_intervals_sec)
    sdnn = np.std(rr_intervals_sec) # Desviación estándar de los intervalos Normal-a-Normal

    print(f"Intervalo R-R Promedio (Mean RR): {mean_rr * 1000:.2f} ms")
    print(f"Desviación Estándar de Intervalos RR (SDNN): {sdnn * 1000:.2f} ms")
    # Aquí se haría el análisis descrito en la guía, comparando con valores esperados
    # o entre diferentes condiciones si las hubiera. Por ejemplo:
    # Un SDNN bajo puede indicar estrés o poca adaptabilidad autonómica.
    # Un SDNN alto (en reposo) generalmente indica buena función parasimpática.
else:
    print("No hay suficientes datos de intervalos RR para análisis en dominio del tiempo.")

# --- Punto 9e: Aplicación de transformada Wavelet ---
print("\n--- Iniciando Punto 9e: Análisis Wavelet (Tiempo-Frecuencia) ---")

if len(rr_intervals_sec) > 5: # Necesitamos una serie de RR razonable
    # 1. Interpolar la serie RR para tener muestreo uniforme
    # Frecuencia de muestreo para la serie RR interpolada (común: 4 Hz)
    target_fs_rr = 4.0
    # Asegurarse que target_fs_rr sea menor que la fs original / factor sobremuestreo picos
    if target_fs_rr > fs / 2:
        target_fs_rr = fs / 2
        print(f"Ajustando fs de interpolación RR a {target_fs_rr:.2f} Hz (<= fs/2)")

    # Crear rejilla de tiempo uniforme desde el primer hasta el último RR detectado
    time_uniform = np.arange(rr_times_sec[0], rr_times_sec[-1], 1/target_fs_rr)

    if len(time_uniform) > 1:
        rr_interpolated = np.interp(time_uniform, rr_times_sec, rr_intervals_sec)

        # Visualización de la serie RR interpolada
        plt.figure(figsize=(12, 5))
        plt.plot(rr_times_sec, rr_intervals_sec * 1000, 'ro', label='Original RR')
        plt.plot(time_uniform, rr_interpolated * 1000, 'b-', label=f'Interpolado ({target_fs_rr} Hz)')
        plt.title('Serie R-R Original vs. Interpolada')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Intervalo R-R (ms)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2. Calcular la Transformada Wavelet Continua (CWT)
        # Wavelet: 'cmorB-C' (Complex Morlet) es buena para tiempo-frecuencia. Ajustar B y C.
        # B: Ancho de banda (bandwidth), C: Frecuencia central (center frequency)
        wavelet_name = 'cmor1.5-1.0' # (B=1.5, C=1.0) - Una opción común
        # Definir las escalas para cubrir las frecuencias de interés (LF, HF)
        # Frecuencias de interés HRV: LF (0.04-0.15 Hz), HF (0.15-0.4 Hz)
        min_freq_hrv = 0.03 # Ligeramente por debajo de LF
        max_freq_hrv = 0.5  # Ligeramente por encima de HF
        total_scales = 128 # Número de escalas (resolución en frecuencia)
        # Calcular las escalas correspondientes a las frecuencias deseadas
        # La frecuencia central de la wavelet para escala 1 es pywt.central_frequency(wavelet_name)
        # freq = pywt.central_frequency(wavelet_name) * target_fs_rr / scale
        # scale = pywt.central_frequency(wavelet_name) * target_fs_rr / freq
        # Ajuste: usar scale2frequency que ya considera el sampling_period
        scales = np.geomspace(pywt.scale2frequency(wavelet_name, 1)/max_freq_hrv,
                              pywt.scale2frequency(wavelet_name, 1)/min_freq_hrv,
                              num=total_scales) * target_fs_rr # Ajuste por la fs_rr

        print(f"Calculando CWT con wavelet '{wavelet_name}'...")
        sampling_period_rr = 1.0 / target_fs_rr
        coefficients, frequencies = pywt.cwt(rr_interpolated, scales, wavelet_name,
                                             sampling_period=sampling_period_rr)

        # 3. Calcular la potencia (magnitud al cuadrado de los coeficientes)
        power = np.abs(coefficients)**2

        # 4. Visualizar el Espectrograma Wavelet
        plt.figure(figsize=(14, 7))
        # Usar pcolormesh es a veces preferible a contourf para evitar artefactos de interpolación
        plt.pcolormesh(time_uniform, frequencies, power, shading='gouraud', cmap='viridis')
        plt.colorbar(label='Potencia Wavelet $|W(t,f)|^2$')
        plt.ylabel('Frecuencia (Hz)')
        plt.xlabel('Tiempo (s)')
        plt.title(f'Espectrograma Wavelet (CWT) de la Serie R-R (Wavelet: {wavelet_name})')
        # Marcar las bandas LF y HF
        lf_low, lf_high = 0.04, 0.15
        hf_low, hf_high = 0.15, 0.4
        plt.axhspan(lf_low, lf_high, color='red', alpha=0.2, label=f'LF ({lf_low}-{lf_high} Hz)')
        plt.axhspan(hf_low, hf_high, color='cyan', alpha=0.2, label=f'HF ({hf_low}-{hf_high} Hz)')
        # Ajustar escala Y si es necesario (log podría ser útil si el rango es amplio)
        # plt.yscale('log')
        plt.ylim([min_freq_hrv, max_freq_hrv]) # Limitar a frecuencias de interés
        # Añadir leyenda para las bandas (puede requerir ajuste de posición)
        handles, labels = plt.gca().get_legend_handles_labels()
        # Crear patches ficticios para la leyenda de axhspan si no aparecen
        from matplotlib.patches import Patch
        if not any(label.startswith('LF') for label in labels):
             handles.append(Patch(color='red', alpha=0.2))
             labels.append(f'LF ({lf_low}-{lf_high} Hz)')
        if not any(label.startswith('HF') for label in labels):
             handles.append(Patch(color='cyan', alpha=0.2))
             labels.append(f'HF ({hf_low}-{hf_high} Hz)')
        plt.legend(handles, labels, loc='upper right')
        plt.show()

        # 5. Análisis de Bandas LF y HF a lo largo del tiempo
        lf_mask = (frequencies >= lf_low) & (frequencies <= lf_high)
        hf_mask = (frequencies >= hf_low) & (frequencies <= hf_high)

        # Calcular potencia integrada en cada banda para cada instante de tiempo
        power_lf = np.sum(power[lf_mask, :], axis=0) * np.mean(np.diff(frequencies[lf_mask])) # Normalizar por ancho de banda
        power_hf = np.sum(power[hf_mask, :], axis=0) * np.mean(np.diff(frequencies[hf_mask])) # Normalizar

        # Calcular el ratio LF/HF (evitar división por cero)
        # Sumar un epsilon pequeño al denominador o manejar ceros
        epsilon = 1e-10
        lf_hf_ratio = power_lf / (power_hf + epsilon)

        # Visualizar la potencia LF, HF y su ratio
        plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(3, 1, 1)
        plt.plot(time_uniform, power_lf, label='Potencia LF', color='red')
        plt.title('Potencia en Bandas de Frecuencia HRV')
        plt.ylabel('Potencia LF')
        plt.grid(True)
        plt.legend()
        plt.setp(ax1.get_xticklabels(), visible=False) # Ocultar etiquetas X excepto en el último subplot

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(time_uniform, power_hf, label='Potencia HF', color='blue')
        plt.ylabel('Potencia HF')
        plt.grid(True)
        plt.legend()
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        plt.plot(time_uniform, lf_hf_ratio, label='Ratio LF/HF', color='purple')
        plt.ylabel('Ratio LF/HF')
        plt.xlabel('Tiempo (s)')
        plt.grid(True)
        plt.legend()
        # Podrías añadir una línea en y=1 o y=1.5 como referencia si es relevante
        # plt.axhline(1.5, color='k', linestyle='--', alpha=0.5, label='Referencia Ratio')

        plt.suptitle('Evolución Temporal de Potencia en Bandas LF y HF', y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Ajustar layout para título
        plt.show()

        # Análisis crítico (descripción según punto 9e):
        print("\nAnálisis Crítico (Punto 9e):")
        print("- Observa el espectrograma: ¿Hay zonas/tiempos con mayor potencia en la banda LF (roja)? ¿Y en la HF (cyan)?")
        print("- Mira los gráficos de Potencia LF y HF vs Tiempo: ¿Cómo varían a lo largo de los 5 minutos?")
        print("- ¿Hay periodos donde predomina la potencia LF sobre la HF (ratio LF/HF alto)? Esto podría sugerir mayor actividad simpática.")
        print("- ¿Hay periodos donde predomina la potencia HF (ratio LF/HF bajo)? Esto podría sugerir mayor actividad parasimpática (vagal).")
        print("- Dado que la señal fue tomada en reposo (según punto 9b), se esperaría una predominancia relativa de la actividad HF o un ratio LF/HF no muy elevado, pero esto depende mucho del estado del sujeto.")

    else:
        print("No hay suficientes puntos en la serie RR interpolada para análisis CWT.")
else:
    print("No hay suficientes datos de intervalos RR (>5) para análisis wavelet.")


# --- Punto 10: Resultados Esperados / Discusión ---
print("\n--- Reflexiones sobre Punto 10: Resultados Esperados ---")
print("1. Comparación Dominio Tiempo vs. Tiempo-Frecuencia:")
print(f"   - Dominio Tiempo (SDNN={sdnn*1000:.2f} ms si calculado): Da una medida global de la variabilidad total en los 5 min.")
print("   - Dominio Tiempo-Frecuencia (Wavelet): Muestra CÓMO cambia la distribución de potencia entre LF y HF a lo largo del tiempo.")
print("     Por ejemplo, el SDNN podría ser el mismo en dos grabaciones, pero una podría tener fluctuaciones rápidas entre LF y HF y la otra no; el wavelet lo mostraría.")

print("\n2. Relación LF/HF con Actividad Autonómica:")
print("   - Banda LF (0.04-0.15 Hz): Se asocia con influencias simpáticas y parasimpáticas (más simpáticas, control barorreflejo).")
print("   - Banda HF (0.15-0.4 Hz): Se asocia principalmente con la modulación parasimpática (vagal) ligada a la respiración (arritmia sinusal respiratoria).")
print("   - Ratio LF/HF: A menudo se interpreta como un índice del balance simpático-vagal (valores altos -> predominio simpático; valores bajos -> predominio parasimpático). Esta interpretación debe ser cautelosa.")
print("   - Discusión: ¿Los patrones de potencia LF y HF observados en el espectrograma y gráficos temporales son consistentes con una condición de reposo?")

print("\n3. Transmisión del Conocimiento (GitHub):")
print("   - La guía menciona [fuente: 36, 37] la importancia de poder explicar el código desarrollado (por ejemplo, en un repositorio público).")
print("   - Asegúrate de entender cada paso del código, los parámetros elegidos (filtros, picos, wavelet), y cómo interpretar los resultados.")
print("   - Comenta bien tu código final para que otros (y tú en el futuro) puedan entenderlo.")

print("\n--- Fin del Análisis ---")