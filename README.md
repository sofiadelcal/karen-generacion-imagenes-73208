# Generación de imágenes: GAN y difusión latente

Actividad R2-A5-S14 Generación (Moodle 73208).

## Contenido publicado

- Karen_DCGAN_MNIST.ipynb: prueba de DCGAN en MNIST ejecutada en Colab.
- gan_core.py y experimento_digitos.py: GAN introductoria en NumPy.
- Las métricas y figuras publicadas corresponden a la GAN introductoria de dígitos 8 × 8.

## Estado de ejecución

La GAN introductoria se ejecutó con cinco semillas. El MMD² medio pasó de 0,4137 a 0,0265; la referencia real-real fue 0,0086. Esta métrica resume distancia entre distribuciones con el kernel del script y no demuestra por sí sola calidad visual ni diversidad.

La DCGAN de MNIST se ejecutó en CPU con 1.000 imágenes y dos épocas. Las pérdidas D/G fueron 1,3230/0,6983 y 1,1338/0,7804. Es una corrida breve de demostración, no un resultado facial; los mosaicos y el checkpoint generados en Colab aún no están publicados en este repositorio.

La práctica con LFW y el ajuste de Stable Diffusion con LoRA siguen pendientes de ejecución. Los cuadernos preparados y el informe distinguen ese estado y no presentan resultados no medidos. LoRA requiere GPU y acceso autorizado al modelo. Nunca se debe guardar HF_TOKEN en el repositorio.

## Reproducción local

Con Python, NumPy, Matplotlib y scikit-learn instalados, ejecutar:

python experimento_digitos.py

El script usa las semillas 204–208 y regenera las métricas y figuras de la GAN introductoria. Revisar licencias de datos y modelos antes de distribuir imágenes o adaptadores.
