# app.py — WebApp en Python (Streamlit) para extraer tablas desde imágenes/PDF
# -------------------------------------------------------------------------
# Características principales:
# - Subir una o varias imágenes (o páginas de PDF convertidas a imagen externamente).
# - Deskew (des-rotación) automático con Hough.
# - Detección de bloques de texto con EAST (OpenCV DNN) + NMS.
# - Agrupación de bloques en filas/columnas por alineación (tolerancias ajustables).
# - Estructuración de tabla y armado de DataFrame (auto/usuario define #filas/#columnas).
# - OCR opcional por bloque con Tesseract (si está instalado) para llenar la tabla.
# - Descarga de resultados: CSV y Excel. También CSV de bloques detectados.
# - Panel de depuración con overlays (Hough, cajas EAST, clusters fila/columna).
# -------------------------------------------------------------------------
# Requisitos (requirements.txt sugerido):
# streamlit>=1.33.0
# opencv-python-headless>=4.9.0.80
# numpy>=1.24.0
# pillow>=10.0.0
# pandas>=2.1.0
# openpyxl>=3.1.0
# scipy>=1.11.0
# pytesseract>=0.3.10      # opcional (requiere instalación del binario Tesseract en el SO)
# pdf2image>=1.17.0        # opcional si deseas convertir PDF a imágenes antes de subir
# -------------------------------------------------------------------------

import os
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st

from image_utils import ImageUtils
from pipeline import PipelineConfig, TableExtractionPipeline
from ocr import TESSERACT_OK


st.set_page_config(page_title="TABULAR:APP — OCR de Tablas (Python)", layout="wide")

st.title("📄➡️📊 TABULAR:APP — De foto/PDF a tabla (Python)")

with st.sidebar:
    st.header("Parámetros")
    st.markdown("**Deskew (Hough)**")
    canny1 = st.slider("Canny th1", 0, 200, 50, 1)
    canny2 = st.slider("Canny th2", 0, 300, 150, 1)
    hough_th = st.slider("Hough threshold", 10, 300, 120, 5)

    st.markdown("**EAST**")
    score_th = st.slider("Score mínimo", 0.1, 0.9, 0.5, 0.05)
    nms_th = st.slider("NMS IoU", 0.1, 0.9, 0.3, 0.05)

    st.markdown("**Agrupación**")
    row_tol = st.slider("Tolerancia filas (px)", 2, 50, 12, 1)
    col_tol = st.slider("Tolerancia columnas (px)", 5, 80, 24, 1)
    prefer_left = st.checkbox("Preferir alineación por borde izquierdo (columnas)", value=True)

    st.markdown("**Post-procesamiento**")
    merge_row_tol = st.slider("Tolerancia vertical para fusionar cajas (px)", 2, 40, 10, 1)
    merge_gap_tol = st.slider("Separación horizontal máxima para fusionar (px)", 2, 80, 18, 1)

    st.markdown("**Tabla**")
    auto_shape = st.checkbox("Estimar automáticamente filas/columnas", value=True)
    manual_rows = st.number_input("Filas (manual)", min_value=1, max_value=200, value=6)
    manual_cols = st.number_input("Columnas (manual)", min_value=1, max_value=50, value=4)

    do_ocr = st.checkbox("Hacer OCR para llenar celdas (Tesseract)", value=False)
    tess_lang = st.text_input("Idioma Tesseract (lang)", value="eng")

    st.markdown("**Depuración**")
    show_hough = st.checkbox("Ver overlay Hough", value=False)
    show_east = st.checkbox("Ver cajas EAST", value=True)
    show_groups = st.checkbox("Ver grupos filas/columnas", value=True)

st.markdown("---")

uploaded_files = st.file_uploader(
    "Sube una o varias imágenes (JPG/PNG). Para PDF, conviértelo a imágenes previamente.",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    pipeline = TableExtractionPipeline()

    all_results = []
    zip_buffer = io.BytesIO()
    from zipfile import ZipFile

    zf = ZipFile(zip_buffer, mode="w")

    for up in uploaded_files:
        st.subheader(f"🖼️ {up.name}")
        img_pil = Image.open(up).convert("RGB")
        img_bgr = ImageUtils.pil_to_cv2(img_pil)

        cfg = PipelineConfig(
            canny1=canny1,
            canny2=canny2,
            hough_thresh=hough_th,
            score_thresh=score_th,
            nms_thresh=nms_th,
            row_tol=row_tol,
            col_tol=col_tol,
            prefer_left_alignment=prefer_left,
            merge_row_tol=merge_row_tol,
            merge_gap_tol=merge_gap_tol,
            do_ocr=do_ocr,
            ocr_lang=tess_lang,
            manual_rows=None if auto_shape else int(manual_rows),
            manual_cols=None if auto_shape else int(manual_cols),
        )

        result, dbg = pipeline.process(img_bgr, config=cfg, debug=True)
        deskewed = result.deskewed
        df = result.table
        dbg_h = dbg.get("deskew", {})
        dbg_e = dbg.get("east", {})
        st.caption(f"Ángulo estimado: {result.angle_deg:.2f}°")

        row_groups = result.row_groups
        col_groups = result.col_groups
        assignments = result.assignments
        boxes_final = [
            (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
            for row in result.boxes_df.itertuples()
        ]

        # ---------------- Visualizaciones de depuración ----------------
        cols = st.columns(3)
        with cols[0]:
            st.image(img_pil, caption="Original", use_column_width=True)
        with cols[1]:
            st.image(ImageUtils.cv2_to_pil(deskewed), caption="Deskewed", use_column_width=True)
        with cols[2]:
            if show_hough and "hough_overlay" in dbg_h:
                st.image(ImageUtils.cv2_to_pil(dbg_h["hough_overlay"]), caption="Hough overlay", use_column_width=True)
            else:
                st.empty()

        overlay = deskewed.copy()
        if show_east and "east_overlay" in dbg_e:
            overlay = dbg_e["east_overlay"].copy()
        if show_groups:
            # Dibujar grupos con colores por fila y columna
            # Fila: borde verde; Columna: etiqueta texto
            # Nota: Streamlit no permite colores aleatorios fácilmente sin estado; usamos paleta simple.
            for r, grp in enumerate(row_groups):
                color = (0, int(50 + 200 * (r / max(1, len(row_groups) - 1))), 0)
                for idx in grp:
                    x1, y1, x2, y2 = boxes_final[idx]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(overlay, f"r{r}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            for c, grp in enumerate(col_groups):
                color = (int(50 + 200 * (c / max(1, len(col_groups) - 1))), 0, 0)
                for idx in grp:
                    x1, y1, x2, y2 = boxes_final[idx]
                    cv2.putText(
                        overlay,
                        f"c{c}",
                        (x1, min(overlay.shape[0] - 2, y2 + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        st.image(ImageUtils.cv2_to_pil(overlay), caption="Detecciones / Grupos", use_column_width=True)

        st.markdown("### Tabla estimada")
        st.dataframe(df, use_container_width=True)

        # Exportar CSV y Excel
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        excel_bytes = excel_buf.getvalue()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ Descargar CSV",
                data=csv_bytes,
                file_name=f"{os.path.splitext(up.name)[0]}_table.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "⬇️ Descargar Excel",
                data=excel_bytes,
                file_name=f"{os.path.splitext(up.name)[0]}_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # Guardar CSV de bloques detectados
        blocks_df = result.boxes_df.copy()
        blocks_df.insert(0, "image_name", up.name)
        blocks_csv = blocks_df.to_csv(index=False).encode("utf-8")
        with c3:
            st.download_button(
                "⬇️ Bloques detectados (CSV)",
                data=blocks_csv,
                file_name=f"{os.path.splitext(up.name)[0]}_blocks.csv",
                mime="text/csv",
            )

        # Agregar a ZIP
        zf.writestr(f"{os.path.splitext(up.name)[0]}_table.csv", csv_bytes)
        zf.writestr(f"{os.path.splitext(up.name)[0]}_table.xlsx", excel_bytes)
        zf.writestr(f"{os.path.splitext(up.name)[0]}_blocks.csv", blocks_csv)

        all_results.append(
            {
                "name": up.name,
                "angle": result.angle_deg,
                "n_boxes": len(boxes_final),
                "n_rows": result.n_rows,
                "n_cols": result.n_cols,
            }
        )

    zf.close()
    st.markdown("---")
    st.subheader("📦 Descarga masiva")
    st.download_button(
        "⬇️ Descargar todo (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="tabular_app_outputs.zip",
        mime="application/zip",
    )

    st.markdown("### Resumen del procesamiento")
    st.dataframe(pd.DataFrame(all_results))
else:
    st.info("Sube una imagen de una tabla escaneada o fotografiada para empezar.")

st.markdown("---")
with st.expander("💡 Notas y consejos"):
    st.markdown(
        """
        - **EAST** localiza texto incluso sin líneas de tabla visibles. Ajusta *Score mínimo* y *NMS* si ves demasiadas/pocas cajas.
        - **Filas / Columnas**: las tolerancias controlan lo estricto de la alineación. Filas deben ser más sensibles en Y (tolerancia menor).
        - **Manual vs Auto**: si la estimación no coincide con tu tabla, desactiva *Auto* y fija nº de filas/columnas.
        - **OCR**: activa Tesseract si lo tienes instalado para llenar celdas. También puedes exportar los **bloques detectados** y hacer OCR fuera.
        - Para **PDF**, conviértelo a imágenes (p. ej. con *pdf2image*) antes de subir.
        """
    )
