# Transformer-based Restoration of Images Degraded by Adverse Weather Conditions
Código base tomado de : https://github.com/jeya-maria-jose/TransWeather y https://github.com/lzdnsb/Restoration-of-Images-Degraded-by-Adverse-Weather-Conditions/tree/main/TransWeather 

En esta carpeta se encuentran los ficheros para transformar en onnx, cuantificar, aplica Knowledge Distillation, entrenar y evaluar para modelos de pytorch y onnx.

## Ficheros test
Dos ficheros de test, guardan imágenes y ficheros con tiempos y métricas (mse, psnr, ssim):
1. test_metrics: para ficheros pytorch. 
	- Con --test_batch_size el tamaño de batch, por defecto a 1
	- Con --model el tipo de modelo a cargar, 0 original, 1 KD, 2 Light, 3 Encoder, 4 decoder
	
2. test_onnx: para los ficheros onnx.
	- Con --test_batch_size el tamaño de batch, por defecto a 1
	- Con --model elegir entre "og" para modelo original en onnx, fp16 o int8

## Ficheros train
Ficheros train, se guarda fichero log, gráficas de loss y el modelo después de cada época:

1. train_kd: Entrena el student con técnica KD. 
2. train_encoder_kd y train_decoder_kd: Entrenamiento solo del encoder, se congelan todas las capas que no tengan "Tenc" o "Tdec" en el nombre, 200 épocas con early stopping de paciencia 10 épocas y margen 0.001. Mirar parámetros en el fichero, algunos de los importantes:
	-- learning_rate
	-- train_batch_size
	-- num_epochs
## ONNX
Convierte el modelo indicado a onnx y aplica cuantificación, descomentar llamadas a funciones dependiendo de lo que se quiera ejecutar.

## Video
Fichero video_test.
Pulsar tecla "q" para cerrar los videos
	- Con --video indicar nombre del video (solo {nombre}.mp4, la ruta es siempre ../videos
Igual que en test_metrics para lo demás:
	- Con --model se elige qué modelos cargar: 0 para original, 1 para KD, 2 para ligero, 3 encoder, 4 decoder.
Por defecto es 0, originales

