## LABORATORIO #5 PDS - Variabilidad de la Frecuencia Cardiaca usando la Transformada Wavelet
El objetivo de este repositorio es analizar la variabilidad de la frecuencia cardíaca (HRV) utilizando la transformada wavelet para identificar cambios en las frecuencias características 
y analizar la dinámica temporal de la señal cardíaca.

## Información a tener en cuenta 
En este codigo se realizó el analizis de una muestra de  ECG y sus cambios de frecuencia cuando un indviduo se somete de un estado de reposo a un estado de perturbación.Por esta razón se deben tener en cuenta los siguientes conceptos e información.

## Variabilidad en la frecuencia cardiaca (HVR)
Es una medida de las fluctuaciones naturales en los intervalos de tiempo entre latidos consecutivos, específicamente entre los picos ¨R-R¨ del electrocardiograma (ECG). Estas variaciones, que pueden ser de apenas milisegundos, reflejan la capacidad del sistema nervioso autónomo para adaptarse a demandas internas y externas. Una ¨HRV alta¨ sugiere un buen equilibrio entre la rama simpática (activación) y parasimpática (relajación), mientras que una ¨HRV baja¨ puede indicar estrés, fatiga o problemas cardiovasculares.  

La HRV se analiza mediante métodos de dominio del tiempo como la desviación estándar de intervalos R-R, SDNN o de dominio de frecuencia descomponiendo las oscilaciones en bandas como LF (low frecuency), HF (higth fracuency) y LF/HF, que reflejan actividad simpática y parasimpática. También se usan métodos no lineales, como la entropía aproximada, para evaluar la complejidad del sistema. Estas métricas ayudan a evaluar el estado de salud, la recuperación deportiva e incluso el estrés mental. 

## Frecuencias de interes de un ECG
En un ECG, las frecuencias de interés asociadas a la actividad del sistema nervioso autónomo (simpático y parasimpático) se analizan principalmente en el dominio espectral de la variabilidad de la frecuencia cardíaca (HRV), las más representativas pueden ser. 
- LF (Low Frequency, 0.04-0.15 Hz): Refleja una mezcla de influencia simpática (activación) y, en menor medida, parasimpática. Se asocia con la modulación de la presión arterial y el sistema nervioso simpático.  
- HF (High Frequency, 0.15-0.4 Hz): Está estrechamente ligada a la actividad parasimpática (vagal), sincronizada con la respiración (arritmia sinusal respiratoria). Un aumento en HF indica mayor tono vagal y relajación.  
- LF/HF: Esta proporción se usa como indicador del balance simpático-vaga, un ejemplo el  estrés agudo aumenta LF/HF, mientras el reposo lo reduce.  

## Transformada Wavelet
En actividad simpática (ejercicio o estrés), predominan las oscilaciones en ¨LF¨ (frecuencias bajas), reduciendose ¨HF¨ por inhibición vagal. En estado parasimpático (reposo, sueño), aumenta HF y disminuye LF. Estas frecuencias permiten se permiten evaluar usando métodos como la ¨transformada de Fourier¨ o ¨análisis de Wavelet¨ en señales de HRV.

La transformada wavelet es una herramienta matemática utilizada para descomponer una señal en componentes de diferentes frecuencias y escalas, permitiendo analizar su comportamiento tanto en el dominio del tiempo como en el de la frecuencia. A diferencia de la transformada de Fourier, que utiliza funciones sinusoidales infinitas, la transformada wavelet emplea funciones localizadas en el tiempo (wavelets o "ondeletas"), lo que la hace especialmente útil para estudiar señales no estacionarias o con transitorios.  

Existen dos tipos principales. La transformada wavelet continua (CWT), que analiza la señal en un rango continuo de escalas y traslaciones, y la transformada wavelet discreta (DWT), que utiliza escalas y traslaciones discretas para facilitar su implementación computacional.

La transformada wavelet es fundamental en el procesamiento de señales de electrocardiograma (ECG) debido a su capacidad para analizar componentes de alta y baja frecuencia de manera simultánea. En señales ECG, permite identificar y separar eficientemente elementos clave como los complejos QRS, las ondas P y T, así como el ruido de fondo (como artefactos musculares, interferencia de línea base o ruido electromagnético). A diferencia de la transformada de Fourier, que solo proporciona información en frecuencia, la wavelet preserva la localización temporal de los eventos, lo que es crucial para detectar anomalías como arritmias o extrasístoles.  

Además, la transformada wavelet discreta (DWT) se utiliza en algoritmos de compresión de señales ECG, ya que permite representar la información con menos coeficientes sin perder detalles clínicamente relevantes. También es útil en la eliminación de ruido mediante técnicas de umbralización, donde se descartan coeficientes wavelet asociados a interferencias.

## Tipos de transformadas Wavelet 
Las transformadas wavelet aplicadas a señales biológicas se clasifican principalmente en tres.
- *transformada wavelet continua (CWT)*
- *transformada wavelet discreta (DWT)* 
- *transformada wavelet estacionaria (SWT)*

 La *CWT* es útil para analizar señales no estacionarias como EEG, EMG o ECG, ya que proporciona una representación detallada en tiempo y escala, ideal para detectar eventos transitorios o cambios sutiles en frecuencias. Sin embargo, su elevado costo computacional la hace menos práctica para aplicaciones en tiempo real.  

 La *DWT* es la más utilizada en procesamiento de señales biomédicas debido a su eficiencia y facilidad de implementación. Mediante bancos de filtros, descompone señales como ritmos cardíacos o actividad cerebral en sub-bandas de frecuencia, permitiendo tareas como eliminación de ruido (denoising), extracción de características o detección de anomalías. Variantes como la *transformada wavelet packet (WPT)* ofrecen mayor flexibilidad al descomponer tanto las bandas de alta como de baja frecuencia, siendo útil en estudios de sueño o diagnóstico de epilepsia.  

La *SWT*evita el efecto de submuestreo y preserva mejor la información temporal en señales como la respiración o la presión arterial, y las *wavelets complejas*, que permiten analizar la fase y amplitud en oscilaciones cerebrales. Estas herramientas son esenciales en aplicaciones clínicas, como monitoreo de pacientes, diagnóstico automatizado e investigación en neurociencia, donde se requiere un equilibrio entre resolución temporal y espectral.

# Procedimiento toma de datos ECG
Para este laboratorio analizamos la variabilidad de la frecuencia cardíaca (HRV) utilizando la transformada wavelet para identificar cambios en las frecuencias, pero antes de pasar al punto digital necesitamos centrarnos en la toma ECG que realizamos en el laboratorio.
![alt text](<White Minimalist Modern Recruitment Process Flowchart.png>)
- *paso 1:* realizamos un codigo en matlab y por medio de una stm32, un cable de comunicación serial y un sensor de electro cardiograma para obtener un cambio en la toma de ECG al exponer al individuo a tres tipos de sucesos.
- *paso 2:* en el primer suceso se expuso al sujeto de prueba a un instante de total calma donde el sistema para simpatico prebalece en su actividad caerdiaca, ya que no esta ante una amenaza y se hizo con el fin de simular un estado en el que el sujeto de prueba estaria durmiendo.
- *paso 3:* el sujeto se expuso a un segundo momento en el cual establecia una cenversación cotidiana con el sujeto de prueba, el instante simulaba un momento cotidiano de la vida del sujeto donde tenia que estar pendiente de responder pero no se encontraba en un estado de respuesta huida.
- *paso 4:* el sujeto se expuso a un tercer instante en el cuel se le expuso a un video de terror el cual acciono su sentidoi de respuesta huida provocado por el sistema simpatico, generando que tiviera ciertos movomientos involuntarios devido al trauma provocado. 
- *paso 5:* se le pidio al individuo tranquilidad y hacer un trabajjo de respiración para volver a un etado de calma.
![alt text](<Imagen de WhatsApp 2025-04-29 a las 15.43.16_f3ee027f.jpg>) Esta imagen muestra el montaje y la posición del sujeto de prueba, el examen de los tres momentos cotidianos se hicieron en un estado pasivo del sujeto, es decir no tuvo que moverse de la silla para los tres instantes de tiempo.

# Codigo
## STM32
    #include "main.h"
    #include "usb_device.h"
    #include "usbd_cdc_if.h"
    #include <stdio.h>  // Para sprintf
    #include <string.h> // Para strlen

En primer lugar para capturar la señal ha du usar un microcontrolador o un sistema embebido en este caso sera la STM32 en primer lugar definimos las librerias a usar y en este caso usaremos el protocolo USB para poder compartir los datos de la STM32 a matlab por un ADC

Para este protocolo incluimos la libreria usb_decive.h y usbd_cdc_if.h las cuales nos permitiran seguir el protolo para poder hacer la comunicacion serial, ademas de otras librerias las cuales nos permitiran compartir datos.


    uint32_t ADC[2]={0,0};
    int bits;
    char buffer[64];

Declaramos algunas variables en esta el uint refiriendose a la declaracion de 32 bits en el conversor analogo digital en este caso tenemos a A0 y A1 esto por si alguno no funciona correctamente, ademas contamos con un entero en bits y un chat llamado buffer 

    int main(void)
    {

    /* USER CODE BEGIN 1 */

    /* USER CODE END 1 */

    /* MCU Configuration--------------------------------------------------------*/

    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* USER CODE BEGIN Init */

    /* USER CODE END Init */

    /* Configure the system clock */
    SystemClock_Config();

    /* USER CODE BEGIN SysInit */

    /* USER CODE END SysInit */

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    MX_USB_DEVICE_Init();
    MX_TIM3_Init();
    /* USER CODE BEGIN 2 */

Ahora dentro dle main esta sera la configuracion incial tanto para el ADC 1, timer 3, el usb, serial wire y el crystal ceramic resonator esta configuracion nos permitira hacer uso de estas herramientas siendo la configuracion por default

        
    /* USER CODE END 2 */
        HAL_ADC_Start_DMA(&hadc1, ADC, 2);
        HAL_TIM_Base_Start(&htim3);
    /* Infinite loop */
    /* USER CODE BEGIN WHILE */
        uint8_t result;
        
Ahora inicializamos el ADC esto con un start y lo hacemos con el ADC 1, con el nombre de la variable que lo va a almacenar en 32 bits en este caso ADC y la cantidad de canales en este caso 2, inicializamos el timmer 3 el cual se dara en el ADC para poder enviar los datos en una cierta cantidad de tiempo esto sera util para la frecuencia de muestreo o fs.


    while (1)
    {
        /* USER CODE END WHILE */
            sprintf(buffer, "%lu\r\n", ADC[0]);  // \r\n para mejor visualización en terminal
        do {
            result = CDC_Transmit_FS((uint8_t*)buffer, strlen(buffer));
        } while (result == USBD_BUSY);  // Esperar si el USB está ocupado

        HAL_Delay(1);
        /* USER CODE BEGIN 3 */

Ahora vamos a imprimir los datos en un char buffer de 64 este lo vamos a enviar con un espacio dato por dato y seran los del ADC canal 0, osea A0.

Luego de este haremos un do en donde permitira entrar sin condicion donde el resultado dado como variable de 8 bits sera igual a el protocolo CDC_Transmit_Fs donde buffer se convertira en una variable de 8 bits y contara la cantidad de datos recordando que buffer ya tiene los valores del ADC[0].

Luego de este se hace el while para salir del do en donde si la USB esta ocupada este enviara el dato y esperara 1 ms, esto nos ayudara a enviar 1 dato cada 1 ms, osea que la fs sera de 1Khz, debido a que envia 1 datos cada cierto periodo osea cada 1 ms.

    }
    /* USER CODE END 3 */
    }

Esto sera muy util en stm para poder caturar los datos del ADC y con ello enviarlo de la forma o manera adecuada.

## MATLAB

En primer lugar vamos a obtener la señal de un ADC de la stm y con ello vamos a capturar la señal en matlab esto con comunicacion serial.


    % ======= CONFIGURACIÓN SERIAL =======
    serialPort = 'COM9';      % Ajusta al puerto correspondiente
    baudRate = 9600;
    duration = 300;            % Duración total (segundos)
    outputFile = 'ECG.csv';

Definimos el puerto serial en este caso sera COM9, el boundrate de la stm 9600 la duracion de la señal sera de 300 y lo guardara en un archivo .csv

    % ======= CONECTAR SERIAL =======
    s = serialport(serialPort, baudRate);
    configureTerminator(s, "LF");  % Si STM envía líneas con salto de línea

Conceta al puerto serial donde rectifica el puerto y la conexion del MX de la stm32, luego de ello verifica que la stm este enviando con lineas de salto dato por dato.

    % ======= VARIABLES =======
    timeVec = [];
    signalVec = [];

Definimos las varianles time y signal en donde el tiempo se dara en el eje x y el voltaje se dara en y

    % ======= CONFIGURAR GRÁFICA =======
    figure('Name', 'Señal de ECG', 'NumberTitle', 'off');
    h = plot(NaN, NaN);
    xlabel('Tiempo (s)');
    ylabel('Voltaje (v)');
    title('Señal de ECG');
    xlim([0, 15]);
    ylim([0, 3.3]);  % Ajusta según el rango de tu ADC
    grid on;

configuramos la grafica donde la figura que va a ir pintando tendra en x tiempo (s), en y voltaje (v), el titulo de la grafica sera Señal de ECG, y tendremos unos limites en este caso la STM32 solo podra enviar hasta 3.3 V por lo cual sera el limite en Y y en x sera de 15 segundos esto con el fin de observar de forma correcta la señal.

    % ======= INICIAR LECTURA =======
    disp('Iniciando adquisición...');
    startTime = datetime('now');

    while seconds(datetime('now') - startTime) < duration
        if s.NumBytesAvailable > 0
            dataStr = readline(s);
            value = str2double(dataStr); % value sera el valor en tiempo real de la STM32
            voltage = (value * 3.3) / 4095; %Pasar a volteos
            t = seconds(datetime('now') - startTime);

Inicia la adquisicion de datos y rectifica que todo este o se encuentre acorde a lo esperado ya con esto lo que va a realizar sera mantener una ventana de tiempo durante la duracion con un while, donde vamos a obtener en valor del voltaje el cual sera el valor de bits * 3.3 /4095 y el tiempo en segundos sera un datatime del ahora menos el inicial.

            if ~isnan(voltage)
                timeVec = [timeVec; t];
                signalVec = [signalVec; voltage];
                
                % Mantener solo últimos 15 segundos
                idx = timeVec >= (t - 60);
                set(h, 'XData', timeVec(idx), 'YData', signalVec(idx));
                xlim([max(0, t - 15), max(15, t)]);
                drawnow;
            end
        end
    end

Esto permitira graficar la ventana en tiempo real ms por cada dato lo que nos permitira reconstruir de forma adecuada la frecuencia de muestreo, ahora en la ventana para que se continue moviendo mantendra los ultimos 15 segundos, donde el limite para la ventana sera ese pero se observara de la forma adecuada.

    % ======= GUARDAR LOS DATOS =======
    disp('Adquisición finalizada. Guardando archivo...');
    T = table(timeVec, signalVec, 'VariableNames', {'Tiempo_s', 'Valor'});
    writetable(T, outputFile);
    disp(['Datos guardados en: ', outputFile]);

    % ======= CERRAR SERIAL =======
    clear s;

Se guardan los datos en un archivo csv y con ello nos permitira luego graficarlos en python y con ello poder procesar la señal.
## PYTHON

Ahora deberemos procesar la señal anteriormente capturada.

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt, find_peaks
    import pywt # Importar PyWavelets

En primer lugar vamos a importar algunas librerias en este caso pandas para los calculos con numpy para las graficas usamos el plt, para los filtros el cual sera un IIR usaremos butter, y para la transformada de wavelets usaremos pywt

    file_path = "DATA_ECG/ECG_lab5.csv" 

    df = pd.read_csv(file_path)
    tiempo = df.iloc[:, 0].values  
    voltaje = df.iloc[:, 1].values  

    fs_estimates = 1 / np.diff(tiempo)
    fs_mean = np.nanmedian(fs_estimates) # Usar mediana es más robusto a outliers que la media
    print(f"Frecuencia de muestreo estimada es de: {fs_mean:.2f} Hz")

    fs = fs_mean # Es decir que Fs sera igual a la frecuencia de muestreo estimada

Ahora vamos a capturar el archivo en este caso se llama ECG_lab5 luego de ello vamos a leer el archivo y capturaremos dos variables una en x sera tiempo y otra en y que sera el voltaje, ahora calcularemos la frecuencia de muestreo estimada con 1/T donde la frecuencia de muestreo estimada fue de 703 Hz lo cual se encuentra muy por debajo de los 1Khz que se esperaban capturar esto puede ser debido a que uno que otro dato no se enviaba de la forma adecuada, y luego fs sera igual a fs_mean la que se acaba de calcular.

    # Graficar la señal original
    plt.figure(figsize=(15, 4))
    plt.plot(tiempo, voltaje, label="Señal ECG Original", color="r")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud / Voltaje (v) ")
    plt.title("Señal ECG Original (1 min)")
    plt.legend()
    plt.grid(True)
    plt.show()

Graficaremos la señal original dada anteriormente la cual debe ser la equivalente a la de matlab.


![alt text](<images/ECG_origin.png>)
La gráfica de electromiografía (EMG) muestra dos momentos claramente diferenciados. En el primer instante, correspondiente al reposo, la señal se presenta con una amplitud baja y estable, reflejando la mínima actividad muscular propia de un estado de relajación. Las oscilaciones son casi imperceptibles, sin picos abruptos, lo que indica que el sujeto no está realizando contracciones voluntarias. Este patrón típico de reposo confirma la ausencia de estímulos externos o esfuerzos por parte del paciente. 

Al introducir el video como estímulo, la gráfica experimenta un cambio notable: aparecen oscilaciones de mayor amplitud y frecuencia, con picos repentinos que sugieren activación muscular en respuesta al contenido visual. Estos pulsos pueden asociarse a microcontracciones involuntarias, ya sea por sorpresa, tensión o empatía con lo observado, tam bien se concluye que los valores negativos se dan por ruido muscular al momento de moverse por reflejo.

Ahora vamos a filtrar la señal con un filtro IIR donde usaremos un Butterworth para ello deberemos cumplir con el teorema de nyquist


    # 1. Filtrado IIR (Butterworth como ejemplo)
    nyquist = 0.5 * fs #Para que cumpla el teorema de Nyquist
    order = 5 # Orden del filtro

Dicho teorema dice que la mitad de la fs en este caso que sera de 351,5 Hz debera ser mayor al lowcut y al highcut osea mayor a el filtro el cual ha de aplicarse

    # Filtro IIR pasa bandas empezamos con un pasa altos y luego un pasa bajos

Vamos a hacer un filtro pasa-bandas de 0.5Hz a 100 Hz debido a que es la frecuencia cardiaca que normalmente utiliza el corazon para comunicarse de forma fisiologica.

    # Filtro Pasa-Altos
    lowcut = 0.5  # Frecuencia de corte del filtro pasa altos
    # Asegurar que el Teorema de Nyquist se cumpla
    if lowcut >= nyquist:
        print(f"Advertencia: Frecuencia de corte Pasa-Altos ({lowcut} Hz) es mayor o igual a Nyquist ({nyquist} Hz). omitir el filtro.")
        ecg_filtered_hp = voltaje # No aplicar filtro
    else:
        low = lowcut / nyquist
        b_hp, a_hp = butter(order, low, btype='highpass')
        ecg_filtered_hp = filtfilt(b_hp, a_hp, voltaje) # Realiza el filtro
        print(f"Aplicado Filtro Pasa-Altos de {lowcut} Hz")

Filtro pasa-altos de 0.5 Hz el cual es menor a 351,5 Hz osea que cumple el teorema de Nyquist donde primero se va a hacer un if con el fin de observar si se cumple o no con el teorema de Nyquist debido a que si no se cumple no se podra aplicar el filtro ya con esto se imprime un:

Aplicado Filtro Pasa-Altos de 0.5 Hz

    # Filtro Pasa-Bajos 
    highcut = 100.0 
    # Asegurarse que highcut < nyquist
    if highcut >= nyquist:
        print(f"Advertencia: Frecuencia de corte Pasa-Bajas ({highcut} Hz) es mayor o igual a Nyquist ({nyquist} Hz). Ajustando a Nyquist*0.99 para cumplir el teorema.")
        highcut = nyquist * 0.99 # Ajusta el filtro pasa bajos

    high = highcut / nyquist
    b_lp, a_lp = butter(order, high, btype='lowpass')
    ecg_filtered = filtfilt(b_lp, a_lp, ecg_filtered_hp)
    print(f"Aplicado Filtro Pasa-Bajos de o menor {highcut:.2f} Hz")

Ahora deberemos realizar un pasa-bajos en donde primero se verifica que cumpla con el teorema de Nyquist si es asi hara el filtro sin ningun problema si no lo cumple este se acoplara a la frecuencia mas cercana para poder realizar el filtro ejemplo tengo que la mitad de Fs es 50 Hz y el filtro es de 60Hz entonces lo que hara ser multiplicar a Nyquist *0,99 y con ello se acomplara el filtro pasa bajos a lo mas cercano que se pueda realizar el filtro, y con ello va a imprimir un:  

Aplicado Filtro Pasa-Bajos de o menor 100 Hz

    ## Se hicieron un filtro de 0.5 Hz el pasa altos y un filtro pasa bajos de 100 Hz



    # Visualización de la señal filtrada
    plt.figure(figsize=(15, 4))
    plt.plot(tiempo, ecg_filtered, label="Señal ECG Filtrada", color="g")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud / Voltaje")
    plt.title("Señal ECG Filtrada")
    plt.legend()
    plt.grid(True)
    plt.show()

Ahora se va a observar la señal filtrada donde se espera que ya no exista un offset y la señal inicie desde 0 debido a que esta se daba ya que tenia una parte AC.

![alt text](<images/ECG_filtered.png>)
El offset en una señal filtrada de ECG se refiere a una componente de continua no deseada que puede alterar el análisis. Este offset puede surgir por factores como la polarización de los electrodos, interferencias externas o características propias del sistema de adquisición. Al aplicar filtros (como un pasa-altos o un filtro de línea base), se elimina esta deriva para centrar la señal alrededor de cero, facilitando la detección de eventos cardíacos como ondas P, QRS o T.

    # Graficar señal original vs filtrada
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, voltaje, label="Señal Original", alpha=0.5, color="gray")
    plt.plot(tiempo, ecg_filtered, label="Señal Filtrada", color="green")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Voltaje (V)")
    plt.title("Señal EMG antes y después del filtrado")
    plt.legend()
    plt.grid(True)
    plt.show()

Ahora se van a comparar en una grafica ambas señales donde se puede observar que si el offset se fue del AC y si se realizo el filtro de la manera adecuada.

![alt text](<images/ECG_comparate_filtered.png>)


    # Identificación de Picos R
    # Ajusta la altura y la distancia entre picos R.
    # Distance: Mínima separación entre picos. Si HR max=180 bpm -> 0.33s
    min_hr_bpm = 40 # Esto sera los BMPS minima entre picos la cual sera 40 BPM osea 1,5 segundos
    max_hr_bpm = 180 # Este sera los BMPS maxima entre picos la cual sera de 0,33 segundos
    min_distance_sec = 60.0 / max_hr_bpm #Distancia minima en segundos porque es 60/BMP la cual sera de 0,33 segundos o 333 ms
    min_distance_samples = int(min_distance_sec * fs) #Convierte la distancia minima a tiempo para la grafica
    if min_distance_samples < 1: min_distance_samples = 1 # Asegurar distancia mínima de 1

Ahora vamos a detectar los picos R-R para esto debemos tener en cuentra la distancia minima de separacion de picos en donde esta sera de 0,33 s en este caso lo tomaremos como 180 bpm, y el maximo como 40 bpms o 1,5 segundos ahora bien la distancia en segundos sera igual a 60/ bpm por lo que entre que mayor sea bpm menos segundos son y viceversa, la distancia se tomara como los 0,33 s por la fs lo que sera la distancia minima en la grafica asi mismo que la distancia de esta ha de ser 1.

    # Ahora adaptamos el Umbral pues sera el o la altura maxima que dara el pico o el R
    # Empezamos con un umbral basado en la desviación estándar
    # Se hace con la desviacion estandarf porque debe mantener todos los picos R como cierta altura media 
    peak_height_threshold = np.mean(ecg_filtered) + 0.6 * np.std(ecg_filtered) # Se coge la nueva señal ya filtrada para hayar los picos R
    # Asegurarse que el umbral no sea demasiado bajo si la señal tiene offset negativo
    if peak_height_threshold < np.percentile(ecg_filtered, 75): # Heurística simple
        peak_height_threshold = np.percentile(ecg_filtered, 75)

Con esto vamos a tener en cuenta el umbral osea entiendase como los picos mas altos, osea R-R ya con esto hacemos una desviacion estandar de las muestras para poder observar los R-R mas altos sin dejar de lado a los mas puqueños que alcanzan el umbral para esto usamos la desviacion, como una altura media ya cin esto vamos a detectar los picos R-R ya con la señal filtrada como se observa en la grafica.


    print(f"Detectando picos R con altura > {peak_height_threshold:.3f} y distancia mínima > {min_distance_samples} muestras ({min_distance_sec:.2f} s)")

    peaks_indices, properties = find_peaks(ecg_filtered, height=peak_height_threshold, distance=min_distance_samples)

    print(f"Número de picos R detectados: {len(peaks_indices)}")

Imprimimos la cantidad de picos R detectados en los 300 segundos en este cado fue de:

Detectando picos R con altura > 0.219 y distancia mínima > 4 muestras (0.33 s)
Número de picos R detectados: 353

Donde cada pico R se toma como un ciclo cardiaco donde la frecuencia cardiaca se asume que con los picos detectados fue de 70.6


    # Visualización de picos detectados
    plt.figure(figsize=(15, 4))
    plt.plot(tiempo, ecg_filtered, label='ECG Filtrada')
    plt.plot(tiempo[peaks_indices], ecg_filtered[peaks_indices], 'ro', label='Picos R Detectados')
    plt.title('Detección de Picos R')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud / Voltaje')
    plt.legend()
    plt.grid(True)
    plt.show()

![alt text](<images/ECG_picosR.png>)

Ya con esto observamos la señal filtrada y los picos R-R que se detectan recordando que la R-R no es pareja sino que esta R puede aumentar y variar.


    # Se calculan los intervalor R-R para eso necesitamos 2 picos o 2 R ya dados en la grafica 
    if len(peaks_indices) > 1: #debe ser mayor a 1 porque el minimo de R-R son 2 
        rr_intervals_samples = np.diff(peaks_indices) # R-R de los intervalos va a ser igual a la diferencia de los picos
        rr_intervals_sec = rr_intervals_samples / fs # Convertimos los picos R-R a segundos para poder graficarlos 
        rr_times_sec = tiempo[peaks_indices[1:]]     # Tiempo correspondiente a cada intervalo RR

Ahora vamos a calcular los intervalos R-R osea vamos a tomar el tiempo entre dos picos R, en donde el intervalo minimo es de 0.33 segundos osea que un pico a otro pico va a darse con una distancia de 0.33 segundos minimo, esta va a variar y va a ser el tiempo correspondiente a cada intervalo R-R

De la señal de ECG analizada, podemos concluir que los intervalos R-R reflejan la variabilidad de la frecuencia cardíaca del paciente, donde el intervalo mínimo registrado (0.33 segundos) corresponde a una frecuencia cardíaca máxima de aproximadamente 180 lpm (calculada como 60/0.33). Esta variación en los intervalos R-R se puede decir que es normal.


        if len(rr_intervals_sec) > 1:
            # Visualización de la serie de intervalos R-R (Tacograma)
            plt.figure(figsize=(12, 5))
            plt.plot(rr_times_sec, rr_intervals_sec * 1000, marker='o', linestyle='-', label='Intervalos R-R')
            plt.title('Serie de Intervalos R-R ')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Intervalo R-R (ms)')
            plt.grid(True)
            plt.show()

Ahora se grafica los intervalos R-R y con esta grafica ahora vamos a realizar la transformada de wavelet y el HRV

![alt text](<images/ECG_intervalosR.png>)

Se observan los 353 picos R-R y con este en Y se dara el intervalo entre cada pico siendo en ms, donde el minimo eran 330 ms o 0.33 s, como se observa en la grafica de pico a pico hay una diferencia de tiempo alrededor de 400 a 1500 ms.

De la señal de ECG analizada, se concluye que se registraron 353 intervalos R-R con una variabilidad significativa, donde el intervalo mínimo fue de 330 ms (equivalente a ~180 lpm) y los intervalos fluctuaron entre 400 ms y 1500 ms, reflejando frecuencias cardíacas desde ~40 lpm hasta ~150 lpm. Esta amplia variación sugiere una elevada variabilidad de la frecuencia cardíaca (VFC), que puede ser por los tres escenarios de tiemppo que tuvimos de una modulación autonomica en el instante de total relajación y otro instante perturvando al sujeto de prueba con un video. 



    # ---  Análisis de HRV en el dominio del tiempo ---
    mean_rr = np.mean(rr_intervals_sec) ## COGER EL INTERVALO DE DATOS DE RR
    sdnn = np.std(rr_intervals_sec) # Desviación estándar de los intervalos R-R

    print(f"Intervalo R-R Promedio : {mean_rr * 1000:.2f} ms") # Promedio de los intervalos R-R en este caso debe ser similar a 700 a 900 ms
    print(f"Desviación Estándar de Intervalos RR : {sdnn * 1000:.2f} ms") #Calculo de desviacion estandar

Ahora realizaremos el analisis en el dominio del tiempo mediante la variacion de las frecuencias esto por medio de HRV para este tomaremos la nueva grafica dada por los picos R-R, y se calcularan el promedio de ms y la desviacion estandar.

Intervalo R-R Promedio : 839.75 ms
Desviación Estándar de Intervalos RR : 213.23 ms



    # --- Aplicación de transformada Wavelet ---

    if len(rr_intervals_sec) > 10: # Necesitamos una serie de RR razonable
        # 1. Interpolar la serie RR para tener muestreo uniforme
        # Frecuencia de muestreo para la serie RR interpolada (común: 4 Hz)
        target_fs_rr = 4.0
        # Asegurarse que target_fs_rr sea menor que la fs original / factor sobremuestreo picos
        if target_fs_rr > fs / 2:
            target_fs_rr = fs / 2
            print(f"Ajustando fs de interpolación RR a {target_fs_rr:.2f} Hz (<= fs/2)")

Ahora aplicaremos la transformada de wavelet donde deberemos de tener una seria de R-R dado por los datos anteriormente capturados razonable entiendase como una muestra mayor a 10 datos, luego de ello vamos a ajustar la interpolacion de los datos RR, osea que los datos sean mayores a mas de 2 datos para poder hacer la transformada de wavelet.



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

Observamos los daos inerpolados donde se espera que sean los mismos datos sea muy similar la grafica a la R-R anterior con la diferencia que se observe un punto en cada dato tomado como se observa en la imagen:


![alt text](<images/ECG_interpolate_RR.png>)

Ahora bien se observa la señal original en puntos de los R-R original de la señal original y el interpolado muy similar donde se hace una pequeña atenuacion. 
            



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

Vamos a hacer la transformada de wavelet para este debemos tener en cuenta que esta misma se va a desglosar en varios puntos para ello el primero sera el ancho de banda entendiendose que el ancho de banda frecuencia LF (0.04-0.15 Hz) y HF (0.15-0.4 Hz), que están relacionadas con el sistema nervioso autónomo, en este caso el Corazon, o musculo cardiaco.

Ahora definimos estas bandas de frecuencia se elige la wavelet Morlet compleja con parámetros:
B = 1.5 → control del ancho de banda.
C = 1.0 → frecuencia central.
Esta hara un análisis tiempo-frecuencia, como la HRV, para el sistema nervios autonomo, asi mismo definimos la escala de LF y HF en este caso:
LF: 0.04–0.15 Hz
HF: 0.15–0.4 Hz

Ahora traduciremos la frecuencia que se encontraba en escala con 
pywt.scale2frequency(), para luego generar un espacio de escalas de la frecuencia en maximas y minimas con np.geomspace() y por ultimo multiplicaremos por la frecuencia de muestreo para darla en el tiempo real. 

y ahora calculamos el CWT:

Calculando CWT con wavelet 'cmor1.5-1.0'...



            # 3. Calcular la potencia (magnitud al cuadrado de los coeficientes)
            power = np.abs(coefficients)**2

Calculamos la potencia de los coeficientes dado que es elevado al cuadrado.

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

Ahora vamos a imprimir o a visualizar el espectro de wavelet como se observa en la siguiente grafica:

![alt text](<images/ECG_wavelet.png>)
El espectro wavelet de la señal de ECG revela información detallada sobre las componentes temporales y frecuenciales de la actividad cardíaca, permitiendo identificar las ondas características (P, QRS, T) en escalas específicas: los complejos QRS (alta frecuencia, 10-25 Hz) se distinguen en escalas más finas, mientras que las ondas P y T (baja frecuencia, ~0.5-5 Hz) aparecen en escalas gruesas. 


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

Por ultimo vamos a separar las bandas LH y HF bandas de frecuencia para el espectro de un sistema nervioso autonomo para observar en mayor medida la transformada de wavelet como se observa en la siguiente grafica:

![alt text](<images/ECG_LH_HF.png>)
La gráfica muestra la evolución temporal de la potencia en las bandas LF y HF, junto con el ratio LF/HF (un indicador del equilibrio autonómico). Los valores de potencia HF (0.00–0.06) superan a los de LF (0.00–0.03) en ciertos intervalos, sugiriendo un predominio parasimpático (vagal) en esos momentos. Sin embargo, el ratio LF/HF presenta oscilaciones marcadas (desde 0 hasta 250), lo que podría indicar episodios de activación simpática (ratio alto) o transiciones bruscas en el estado autonómico. 

        else:
            print("No hay suficientes puntos en la serie RR interpolada para análisis CWT.")
    else:
        print("No hay suficientes datos de intervalos RR (>10) para análisis wavelet.")

Se compuerba de que la cantidad de datos R-R dados despues por la interpolaralizacion sea fiable osea mayor a 10.


# Requisitos
- Python 3.11
- Math lab
- Circuito ECG
- Cable de comunicación serial 
- Internet 
- Contenido grafico fuerte

## Bibliografia 
- Mallat, S. (2009). A Wavelet Tour of Signal 
(Fundamentos matemáticos de la transformada wavelet)

- Addison, P. S. (2017). The Illustrated Wavelet .
(Aplicaciones de wavelets en señales biomédicas, como el ECG)
- Clifford, G. D., Azuaje, F., & McSharry, P. E. (2006). Advanced Methods and Tools for ECG Data Analysis. Artech House.
- Alcaraz, R., & Rieta, J. J. (2010). A review on wavelet analysis of the ECG for arrhythmia detection.
- Singh, O., & Sunkaria, R. K. (2017). Wavelet-based denoising of ECG signal for HRV analysis. 
## Contactos 
- Jose Daniel Porras est.jose.dporras@unimilitar.edu.co
- Jhonathan David Guevara Ramirez est.jhonathan.guev@unimilitar.edu.co


