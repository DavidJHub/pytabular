"""Streamlit demo that visualizes each step of the table extraction pipeline."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for candidate in (APP_DIR, ROOT_DIR):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from image_utils import ImageUtils
from pipeline import PipelineConfig, TableExtractionPipeline


BoxTuple = Tuple[int, int, int, int]


def _ensure_color(image: cv2.Mat) -> cv2.Mat:
    """Return a 3-channel BGR image for visualization."""

    if image is None:
        raise ValueError("Cannot normalize a None image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _iter_boxes(boxes: Sequence[BoxTuple] | pd.DataFrame) -> Iterable[BoxTuple]:
    """Yield boxes as integer tuples regardless of the container type."""

    if isinstance(boxes, pd.DataFrame):
        for row in boxes.itertuples():
            yield (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
    else:
        for box in boxes:
            x1, y1, x2, y2 = box
            yield int(x1), int(y1), int(x2), int(y2)


def _draw_boxes(
    image: cv2.Mat,
    boxes: Sequence[BoxTuple] | pd.DataFrame,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> cv2.Mat:
    """Overlay rectangles for the provided boxes on top of the reference image."""

    canvas = _ensure_color(image)
    overlay = canvas.copy()
    for x1, y1, x2, y2 in _iter_boxes(boxes):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    return overlay


def draw_cluster_boxes(image: cv2.Mat, boxes_df: pd.DataFrame) -> cv2.Mat:
    """Visualize cluster assignments (row, col) using deterministic colors."""

    canvas = _ensure_color(image)
    overlay = canvas.copy()

    for row in boxes_df.itertuples():
        pt1 = (int(row.x1), int(row.y1))
        pt2 = (int(row.x2), int(row.y2))
        row_idx = getattr(row, "row")
        col_idx = getattr(row, "col")
        if row_idx is None or col_idx is None:
            color = (180, 180, 180)
        else:
            seed = int(row_idx) * 53 + int(col_idx) * 97
            color = (
                60 + (seed * 29) % 196,
                60 + (seed * 47) % 196,
                60 + (seed * 71) % 196,
            )
        cv2.rectangle(overlay, pt1, pt2, color, 2)
        if row_idx is not None and col_idx is not None:
            label = f"{row_idx},{col_idx}"
            text_origin = (pt1[0] + 2, pt1[1] + 18)
            cv2.putText(
                overlay,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
    return overlay


def _boxes_to_pil(
    image: cv2.Mat,
    boxes: Sequence[BoxTuple] | pd.DataFrame,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> Image.Image | None:
    if not boxes:
        return None
    overlay = _draw_boxes(image, boxes, color=color, thickness=thickness)
    return ImageUtils.cv2_to_pil(overlay)


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _boxes_df_to_csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# Streamlit bootstrap expects to own the runtime, so the UI lives in
# ``_render_app``. ``run_app()`` detects if we were launched via ``streamlit run``
# or ``python src/demo.py`` and either renders directly or bootstraps the
# Streamlit server programmatically.


def _render_app() -> None:
    st.set_page_config(page_title="Pipeline demo", layout="wide")
    st.title("📊 Table Extraction Pipeline — Demo paso a paso")
    st.write(
        "Sube una o varias imágenes de tablas y visualiza cómo el pipeline las "
        "deskewea, detecta el texto, agrupa los cuadros y construye la tabla final."
    )

    with st.sidebar:
        st.header("Parámetros del pipeline")
        canny1 = st.slider("Canny th1", 0, 200, 50, 1)
        canny2 = st.slider("Canny th2", 0, 300, 150, 1)
        hough_th = st.slider("Hough threshold", 10, 300, 120, 5)
        score_th = st.slider("Score mínimo EAST", 0.1, 0.9, 0.5, 0.05)
        nms_th = st.slider("NMS IoU", 0.1, 0.9, 0.3, 0.05)
        merge_row_tol = st.slider("Tolerancia vertical unión cajas", 2, 40, 10, 1)
        merge_gap_tol = st.slider("Separación horizontal unión", 2, 80, 18, 1)
        row_tol = st.slider("Tolerancia filas (agrupación)", 2, 50, 12, 1)
        col_tol = st.slider("Tolerancia columnas (agrupación)", 5, 80, 24, 1)
        prefer_left = st.checkbox("Priorizar alineación por borde izquierdo", value=True)
        do_ocr = st.checkbox("Ejecutar OCR (Tesseract)", value=False)
        tess_lang = st.text_input("Idioma OCR", value="eng")
        auto_shape = st.checkbox("Estimar filas/columnas automáticamente", value=True)
        manual_rows = st.number_input("Filas manual", min_value=1, max_value=200, value=6)
        manual_cols = st.number_input("Columnas manual", min_value=1, max_value=50, value=4)

    uploaded_files = st.file_uploader(
        "Selecciona imágenes de tablas (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Sube al menos una imagen para ejecutar el pipeline.")
        st.stop()

    @st.cache_resource(show_spinner=False)
    def _load_pipeline() -> TableExtractionPipeline:
        try:
            return TableExtractionPipeline()
        except FileNotFoundError:
            st.error(
                "❌ No fue posible inicializar el detector EAST. "
                "Descarga manualmente `frozen_east_text_detection.pb` y colócalo en la carpeta `src/`."
            )
            st.stop()

    pipeline = _load_pipeline()

    for uploaded in uploaded_files:
        st.markdown("---")
        st.header(f"🖼️ {uploaded.name}")

        image = Image.open(uploaded).convert("RGB")
        image_bgr = ImageUtils.pil_to_cv2(image)

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

        with st.spinner("Ejecutando pipeline…"):
            result, debug_payload = pipeline.process(image_bgr, config=cfg, debug=True)

        st.caption(
            f"Ángulo estimado: {result.angle_deg:.2f}° | Celdas: {result.n_rows}×{result.n_cols}"
        )

        # Preprocesamiento
        st.subheader("1. Preprocesamiento")
        pre_cols = st.columns(4)
        original_pil = image
        deskewed_pil = ImageUtils.cv2_to_pil(result.deskewed)
        pre_cols[0].image(original_pil, caption="Original", use_column_width=True)
        pre_cols[1].image(deskewed_pil, caption="Deskewed", use_column_width=True)

        deskew_debug = debug_payload.get("deskew", {})
        edges_img = deskew_debug.get("edges")
        hough_overlay = deskew_debug.get("hough_overlay")

        edges_pil = (
            ImageUtils.cv2_to_pil(_ensure_color(edges_img)) if edges_img is not None else None
        )
        hough_pil = (
            ImageUtils.cv2_to_pil(_ensure_color(hough_overlay))
            if hough_overlay is not None
            else None
        )

        if edges_img is not None:
            pre_cols[2].image(edges_pil, caption="Canny / bordes", use_column_width=True)
        else:
            pre_cols[2].empty()

        if hough_overlay is not None:
            pre_cols[3].image(hough_pil, caption="Overlay Hough", use_column_width=True)
        else:
            pre_cols[3].empty()

        # Detección
        st.subheader("2. Detección de texto")
        raw_boxes = debug_payload.get("raw_boxes", [])
        merged_boxes = debug_payload.get("merged_boxes", [])
        final_boxes = debug_payload.get("final_boxes", [])

        det_cols = st.columns(3)
        raw_img = _boxes_to_pil(result.deskewed, raw_boxes, color=(0, 0, 255))
        merged_img = _boxes_to_pil(result.deskewed, merged_boxes, color=(40, 220, 40))
        final_img = _boxes_to_pil(result.deskewed, final_boxes, color=(255, 180, 0))

        if raw_img is not None:
            det_cols[0].image(raw_img, caption="Cajas EAST crudas", use_column_width=True)
        else:
            det_cols[0].info("Sin cajas detectadas")

        if merged_img is not None:
            det_cols[1].image(merged_img, caption="Post-fusión (solapes)", use_column_width=True)
        else:
            det_cols[1].empty()

        if final_img is not None:
            det_cols[2].image(final_img, caption="Cajas finales a agrupar", use_column_width=True)
        else:
            det_cols[2].empty()

        # Clustering
        st.subheader("3. Clustering de filas/columnas")
        clusters_img = draw_cluster_boxes(result.deskewed, result.boxes_df)
        clusters_pil = ImageUtils.cv2_to_pil(clusters_img)
        st.image(clusters_pil, caption="Asignación (fila, columna)", use_column_width=True)

        # Tabla final
        st.subheader("4. Tabla resultante")
        st.dataframe(result.table, use_container_width=True)

        csv_bytes = result.table.to_csv(index=False).encode("utf-8")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            result.table.to_excel(writer, index=False, sheet_name="Sheet1")
        boxes_csv = _boxes_df_to_csv(result.boxes_df)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ Descargar tabla CSV",
                data=csv_bytes,
                file_name=f"{uploaded.name}_table.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "⬇️ Descargar tabla Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{uploaded.name}_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c3:
            st.download_button(
                "⬇️ Cajas detectadas (CSV)",
                data=boxes_csv,
                file_name=f"{uploaded.name}_boxes.csv",
                mime="text/csv",
            )

        stage_images = {
            "original": original_pil,
            "deskewed": deskewed_pil,
            "edges": edges_pil,
            "hough_overlay": hough_pil,
            "east_raw": raw_img,
            "east_merged": merged_img,
            "east_final": final_img,
            "clusters": clusters_pil,
        }

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED) as zf:
            stem = Path(uploaded.name).stem
            for label, pil_image in stage_images.items():
                if pil_image is None:
                    continue
                zf.writestr(f"{stem}_{label}.png", _pil_to_png_bytes(pil_image))
            zf.writestr(f"{stem}_table.csv", csv_bytes)
            zf.writestr(
                f"{stem}_table.xlsx",
                excel_buffer.getvalue(),
            )
            zf.writestr(f"{stem}_boxes.csv", boxes_csv)

        zip_bytes = zip_buffer.getvalue()

        st.download_button(
            "⬇️ Descargar paquete (ZIP)",
            data=zip_bytes,
            file_name=f"{Path(uploaded.name).stem}_resultados.zip",
            mime="application/zip",
        )

    with st.expander("Ver dataframe de cajas"):
        st.dataframe(result.boxes_df, use_container_width=True)


def run_app() -> None:
    """Render the Streamlit demo, bootstrapping Streamlit when run via ``python``."""

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ModuleNotFoundError:
        # Extremely old Streamlit versions: fall back to plain rendering which will
        # instruct the user to launch via ``streamlit run``.
        get_script_run_ctx = lambda: None  # type: ignore

    ctx = get_script_run_ctx()
    if ctx is None:
        # Ejecutado como ``python src/demo.py``: levantamos un proceso ``streamlit
        # run`` asegurándonos de que ningún argumento legacy (como ``--image``)
        # quede en ``sys.argv``.
        script_path = str(Path(__file__).resolve())
        cmd = [sys.executable, "-m", "streamlit", "run", script_path]

        # Forzamos modo headless si no está configurado para que el comando sea
        # reproducible en servidores remotos / contenedores.
        env = os.environ.copy()
        env.setdefault("STREAMLIT_SERVER_HEADLESS", "1")
        env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")

        try:
            completed = subprocess.run(cmd, check=True, env=env)
        except FileNotFoundError as exc:  # pragma: no cover - entorno roto
            raise RuntimeError(
                "No se encontró el ejecutable de Streamlit. Instala el paquete con "
                "`pip install streamlit` y vuelve a intentar."
            ) from exc

        # Si el proceso hijo terminó correctamente no seguimos renderizando en
        # este intérprete.
        if completed.returncode == 0:
            return

        raise RuntimeError(
            "La ejecución de Streamlit finalizó con un código de error. Revisa los "
            "logs anteriores para más detalles."
        )

    _render_app()


if __name__ == "__main__":
    run_app()
