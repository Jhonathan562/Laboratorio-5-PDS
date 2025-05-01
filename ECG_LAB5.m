% ======= CONFIGURACIÓN SERIAL =======
serialPort = 'COM9';      % Ajusta al puerto correspondiente
baudRate = 9600;
duration = 300;            % Duración total (segundos)
outputFile = 'ECG.csv';

% ======= CONECTAR SERIAL =======
s = serialport(serialPort, baudRate);
configureTerminator(s, "LF");  % Si STM envía líneas con salto de línea

% ======= VARIABLES =======
timeVec = [];
signalVec = [];

% ======= CONFIGURAR GRÁFICA =======
figure('Name', 'Señal de ECG', 'NumberTitle', 'off');
h = plot(NaN, NaN);
xlabel('Tiempo (s)');
ylabel('Valor');
title('Señal de ECG');
xlim([0, 15]);
ylim([0, 3.3]);  % Ajusta según el rango de tu ADC
grid on;

% ======= INICIAR LECTURA =======
disp('Iniciando adquisición...');
startTime = datetime('now');

while seconds(datetime('now') - startTime) < duration
    if s.NumBytesAvailable > 0
        dataStr = readline(s);
        value = str2double(dataStr); % value sera el valor en tiempo real de la STM32
        voltage = (value * 3.3) / 4095; %Pasar a volteos
        t = seconds(datetime('now') - startTime);

        if ~isnan(voltage)
            timeVec = [timeVec; t];
            signalVec = [signalVec; voltage];
            
            % Mantener solo últimos 60 segundos
            idx = timeVec >= (t - 60);
            set(h, 'XData', timeVec(idx), 'YData', signalVec(idx));
            xlim([max(0, t - 15), max(15, t)]);
            drawnow;
        end
    end
end

% ======= GUARDAR LOS DATOS =======
disp('Adquisición finalizada. Guardando archivo...');
T = table(timeVec, signalVec, 'VariableNames', {'Tiempo_s', 'Valor'});
writetable(T, outputFile);
disp(['Datos guardados en: ', outputFile]);

% ======= CERRAR SERIAL =======
clear s;