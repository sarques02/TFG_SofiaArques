# Multi-weather Image Restoration via Domain Translation

In this folder you will find the codes for ONNX conversion, quantization, Knowledge Distillation training and testing for both pytorch models and ONNX models.
The base network and code are taken from: https://github.com/pwp1208/Domain_Translation_Multi-weather_Restoration

## Ficheros test
Dos ficheros de test, los dos guardan las imágenes de resultado y generan fichero txt con métricas y tiempos al terminar de evaluar todas las imágenes

1. test_metrics: Evaluación de los .pth
 
- Con --models se elige qué modelos cargar: 
			0 para los originales, 1 para los KD, 2 para los ligeros.
			Por defecto es 0, originales
   
- Con --factor se indica el factor por el que se divide. 
			Por defecto 4 si es kd o ligero, 1 si es original (se pone solo para los originales)
   
- Con --dataset se indica el path al dataset, por defecto smaller_test
		
2. test_onnx: Evaluación de los .onnx
		Cambiar variable type_net para usar modelos originales en onnx (og), fp16, int8, KD o pruning

## Ficheros train
Dos ficheros de entrenamiento, guardan los modelos y crean un log con métricas y tiempo.

1. train_kd: Entrenamiento por Knowledge Distillation
Hace falta carpeta "datasets" con train y test dentro. 
Tiene Early stopping con paciencia 5 épocas y margen (min delta) 0.001 si se quiere cambiar, buscar variables dentro del código.
- Con --network se decide qué red entrenar: 1 niebla, 2 lluvia o 3 nieve
- Con --factor el factor por el que reducir
- Con --batch_size el batch_size de entrenamiento y validación
		
## ONNX
Fichero onnx_quantification: convierte a onnx y cuantifica. Descomentar llamadas a funciones de debajo del todo del código dependiendo de lo que se quiera ejecutar.

## Video
Fichero video_test.
Pulsar tecla "q" para cerrar los videos
- Con --video indicar nombre del video (solo {nombre}.mp4, la ruta es siempre ../videos
Igual que en test_metrics para lo demás:
- Con --models se elige qué modelos cargar: 0 para los originales, 1 para los KD, 2 para los ligeros.
		Por defecto es 0, originales
- Con --factor se indica el factor por el que se divide. 
		Por defecto 4 si es kd o ligero, 1 si es original (se pone solo para los originales)
