# pytabular

Herramientas para extraer tablas desde imágenes utilizando un pipeline de
preprocesamiento, detección de texto, agrupamiento y armado de DataFrames.

## Requisitos

```bash
pip install -r requirements.txt
```

## Demo (Streamlit)

1. Instala las dependencias (idealmente dentro de un entorno virtual):

   ```bash
   pip install -r requirements.txt
   ```

2. Asegúrate de contar con el modelo `frozen_east_text_detection.pb` en la
   carpeta `src/`. Si no lo tienes, la app intentará descargarlo la primera vez
   que se ejecute (≈90 MB). En entornos sin acceso a internet descárgalo
   manualmente desde el repositorio de `opencv_extra` y colócalo allí.
   Para poblar la tabla automáticamente instala **Tesseract** (binario del SO +
   `pytesseract`) o, alternativamente, `easyocr` (`pip install easyocr`). El
   pipeline usará Tesseract si está disponible y caerá en EasyOCR como backup.

3. Lanza la demo con Streamlit (se puede hacer desde la raíz del repo o desde
   la carpeta `src/`):

   ```bash
   streamlit run src/demo.py
   ```

   También puedes ejecutar `python src/demo.py`; el script arranca el servidor de
   Streamlit automáticamente, sin necesidad de pasar argumentos como `-i` ni rutas
   de entrada.

4. Abre el enlace local que aparece en la consola, arrastra y suelta una o más
   imágenes de tablas y ajusta los parámetros desde la barra lateral para
   inspeccionar cada etapa del pipeline (deskew, detección, clustering y tabla
   final). Cada procesamiento expone botones de descarga tanto para la tabla
   (CSV/XLSX) como para el dataframe completo de clusters (centro, tamaño,
   puntaje, texto y asignación de fila/columna), además de un paquete ZIP con
   todas las visualizaciones y artefactos generados en cada paso.

## Licencia

Distribuido bajo la licencia MIT incluida en el repositorio.
