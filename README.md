# Generación de imágenes: GAN y difusión latente


Actividad R2-A5-S14 Generación (Moodle 73208).


## Contenido publicado


- Karen_DCGAN_MNIST.ipynb: prueba de DCGAN sobre MNIST ejecutada en Colab.
- Karen_DCGAN_Rostros_LFW.ipynb: cuaderno para ampliar el experimento facial.
- Karen_Diffusers_LoRA_Pokemon.ipynb: flujo LoRA preparado; todavía no hay una ejecución terminada.
- gan_core.py y experimento_digitos.py: GAN introductoria en NumPy.


## Estado de ejecución


La GAN introductoria se ejecutó con cinco semillas. El MMD² medio pasó de 0,4137 a 0,0265; la referencia real-real fue 0,0086. Esta métrica resume distancia entre distribuciones con el kernel del script y no demuestra por sí sola calidad visual ni diversidad.


La DCGAN de MNIST se ejecutó en CPU con 1.000 imágenes y dos épocas. Las pérdidas D/G fueron 1,3230/0,6983 y 1,1338/0,7804. Es una demostración breve, no un resultado facial.


La prueba de LFW sí se ejecutó en Colab con CPU: 256 de los 1.288 rostros disponibles tras el filtro, 7 identidades, 32 × 32 píxeles, una época y 8 lotes. Las pérdidas finales fueron D=1,3776 y G=0,6053. La prueba tomó cerca de 30 segundos. Es una comprobación exploratoria y no demuestra convergencia ni calidad; no se calculó FID. Las métricas están publicadas en resultados_lfw/metricas.json; el script reproducible y el registro ampliado están en el paquete local de entrega.


LoRA no se ejecutó: aunque la autenticación de Hugging Face funcionó, Colab rechazó la asignación de GPU por límites de uso. El cuaderno no contiene pesos ni imágenes producidos por ese ajuste. Nunca se debe guardar HF_TOKEN en el repositorio.


## Reproducción local


Para la GAN introductoria, instalar Python, NumPy, Matplotlib y scikit-learn, y ejecutar:


```bash
python experimento_digitos.py
```


La prueba LFW requiere PyTorch, scikit-learn y Pillow. El script `resultados_lfw/run_lfw_trial.py` guarda una cuadrícula y el registro de métricas; su primera ejecución descarga los datos. Para ejecutar LoRA se necesita una sesión Colab con GPU disponible y acceso al modelo base. Revisar las licencias antes de redistribuir datos, imágenes o adaptadores.
