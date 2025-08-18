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
import time
import cv2
import math
import base64
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict

import streamlit as st

# Intentamos importar pytesseract si está disponible
try:
    import pytesseract
    TESSERACT_OK = True
except Exception:
    TESSERACT_OK = False

# ----------------------------- Utilidades gráficas -----------------------------

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    if len(img_bgr.shape) == 2:
        return Image.fromarray(img_bgr)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# ----------------------------- Deskew con Hough -------------------------------

def hough_deskew(image_bgr: np.ndarray, canny1: int = 50, canny2: int = 150,
                  hough_thresh: int = 100, debug: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Estima el ángulo dominante de líneas y rota la imagen para deskew.
    Devuelve imagen rotada y dict con debug.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny1, canny2)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)

    angle_deg = 0.0
    angles = []
    if lines is not None:
        for rho_theta in lines:
            rho, theta = rho_theta[0]
            # Convertimos a grados alrededor del eje horizontal
            deg = (theta * 180 / np.pi)
            # Normalizar a rango [-90, 90)
            deg = (deg + 90) % 180 - 90
            angles.append(deg)
        if len(angles) > 0:
            # Usamos la mediana para robustez
            angle_deg = float(np.median(angles))

    # Rotación
    (h, w) = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    dbg = {
        "angle_deg": angle_deg,
        "edges": edges,
        "lines": lines,
    }
    if debug:
        overlay = image_bgr.copy()
        if lines is not None:
            for rho_theta in lines[:100]:  # limitar para visual
                rho, theta = rho_theta[0]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
        dbg["hough_overlay"] = overlay

    return rotated, dbg

# ------------------------ Detección de texto con EAST -------------------------

EAST_URL = "https://github.com/opencv/opencv_extra/raw/4.x/testdata/dnn/text/frozen_east_text_detection.pb"


def ensure_east_model(model_path: str = "frozen_east_text_detection.pb"):
    if os.path.exists(model_path):
        return model_path
    # Intento de descarga si hay internet; si falla, el usuario deberá colocar el .pb manualmente
    try:
        import urllib.request
        st.info("Descargando modelo EAST (~90MB) la primera vez…")
        urllib.request.urlretrieve(EAST_URL, model_path)
        st.success("Modelo EAST descargado.")
    except Exception as e:
        st.warning(
            "No fue posible descargar el modelo EAST automáticamente. "
            "Descárgalo manualmente y colócalo como 'frozen_east_text_detection.pb' en la carpeta del app.\n" + str(e)
        )
    return model_path


def decode_east(scores: np.ndarray, geometry: np.ndarray, score_thresh: float = 0.5) -> Tuple[List[Tuple[int,int,int,int]], List[float]]:
    """Decodifica salidas de EAST a cajas y puntajes.
    Retorna lista de cajas [x, y, w, h] y lista de confidences.
    """
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            score = scoresData[x]
            if score < score_thresh:
                continue

            # offset factor 4 (EAST salida stride=4)
            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
            endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, w, h))
            confidences.append(float(score))

    return rects, confidences


def nms_boxes(boxes: List[Tuple[int,int,int,int]], scores: List[float], nms_thresh: float = 0.3) -> List[int]:
    """Non-Maximum Suppression: devuelve índices mantenidos."""
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)

    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_np.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def detect_text_blocks_east(image_bgr: np.ndarray, model_path: str,
                            score_thresh: float = 0.5, nms_thresh: float = 0.3,
                            debug: bool = False) -> Tuple[List[Tuple[int,int,int,int]], List[float], Dict]:
    """Detecta bloques de texto con EAST y devuelve cajas en coords originales."""
    (H, W) = image_bgr.shape[:2]
    newW, newH = (int(math.ceil(W / 32) * 32), int(math.ceil(H / 32) * 32))
    rW = newW / float(W)
    rH = newH / float(H)

    blob_img = cv2.resize(image_bgr, (newW, newH))
    blob = cv2.dnn.blobFromImage(blob_img, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net = cv2.dnn.readNet(model_path)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    rects, confidences = decode_east(scores, geometry, score_thresh)
    keep = nms_boxes(rects, confidences, nms_thresh)

    boxes = []
    scores_kept = []
    for i in keep:
        (x, y, w, h) = rects[i]
        # Re-escalar a la imagen original
        startX = int(x / rW)
        startY = int(y / rH)
        endX = int((x + w) / rW)
        endY = int((y + h) / rH)
        # Clip
        startX = max(0, startX); startY = max(0, startY)
        endX = min(W - 1, endX); endY = min(H - 1, endY)
        boxes.append((startX, startY, endX, endY))
        scores_kept.append(confidences[i])

    dbg = {"scores": scores, "geometry": geometry}

    if debug:
        overlay = image_bgr.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        dbg["east_overlay"] = overlay

    return boxes, scores_kept, dbg

# ------------------------- Agrupar filas y columnas ---------------------------

@dataclass
class ClusteredBox:
    x1: int; y1: int; x2: int; y2: int; score: float

try:
    from dataclasses import dataclass
except Exception:
    # Fallback muy raro, pero por si acaso
    def dataclass(cls):
        return cls


def merge_intersecting_boxes(boxes: List[Tuple[int,int,int,int]], iou_thresh: float = 0.0) -> List[Tuple[int,int,int,int]]:
    """Une cajas que se intersectan (>0 intersección) o con IoU>umbral."""
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=np.int32)
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = np.zeros(len(boxes_np), dtype=bool)
        for i in range(len(boxes_np)):
            if used[i]:
                continue
            x1a, y1a, x2a, y2a = boxes_np[i]
            cur = np.array([x1a, y1a, x2a, y2a])
            for j in range(i+1, len(boxes_np)):
                if used[j]:
                    continue
                x1b, y1b, x2b, y2b = boxes_np[j]
                xx1 = max(x1a, x1b); yy1 = max(y1a, y1b)
                xx2 = min(x2a, x2b); yy2 = min(y2a, y2b)
                iw = max(0, xx2 - xx1)
                ih = max(0, yy2 - yy1)
                inter = iw * ih
                if inter > 0:
                    # Unir
                    cur = np.array([
                        min(cur[0], x1b), min(cur[1], y1b), max(cur[2], x2b), max(cur[3], y2b)
                    ])
                    used[j] = True
                    merged = True
            used[i] = True
            new_boxes.append(tuple(map(int, cur.tolist())))
        boxes_np = np.array(new_boxes, dtype=np.int32)
    return [tuple(map(int, b)) for b in boxes_np]


def group_rows_columns(boxes: List[Tuple[int,int,int,int]],
                        row_tol: int = 12, col_tol: int = 24,
                        prefer_left_alignment: bool = True) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Agrupa índices de cajas en filas (alineación en Y) y en columnas (alineación por X izq.).
    - row_tol: tolerancia estricta (píxeles) para considerar la misma fila (más sensible en altura).
    - col_tol: tolerancia algo mayor para columnas por alineación de x1.
    Devuelve (rows_groups, cols_groups), cada uno lista de listas de índices.
    """
    if not boxes:
        return [], []

    # Centroides y bordes izquierdos
    xs1 = np.array([b[0] for b in boxes])
    ys1 = np.array([b[1] for b in boxes])
    xs2 = np.array([b[2] for b in boxes])
    ys2 = np.array([b[3] for b in boxes])
    cy = (ys1 + ys2) / 2.0
    cx = (xs1 + xs2) / 2.0

    # --- Filas por cy ---
    order_y = np.argsort(cy)
    row_groups = []
    current = [int(order_y[0])]
    for k in range(1, len(order_y)):
        i_prev = int(order_y[k-1])
        i_cur = int(order_y[k])
        if abs(cy[i_cur] - cy[i_prev]) <= row_tol:
            current.append(i_cur)
        else:
            row_groups.append(sorted(current, key=lambda idx: boxes[idx][0]))
            current = [i_cur]
    row_groups.append(sorted(current, key=lambda idx: boxes[idx][0]))

    # --- Columnas ---
    # izquierdas x1 (prefer_left_alignment=True) o centroides cx
    key_x = xs1 if prefer_left_alignment else cx
    order_x = np.argsort(key_x)
    col_groups = []
    current = [int(order_x[0])]
    for k in range(1, len(order_x)):
        j_prev = int(order_x[k-1])
        j_cur = int(order_x[k])
        if abs(key_x[j_cur] - key_x[j_prev]) <= col_tol:
            current.append(j_cur)
        else:
            # ordenar por y
            col_groups.append(sorted(current, key=lambda idx: (boxes[idx][1] + boxes[idx][3]) / 2.0))
            current = [j_cur]
    col_groups.append(sorted(current, key=lambda idx: (boxes[idx][1] + boxes[idx][3]) / 2.0))

    return row_groups, col_groups

# ------------------------ Construcción de tabla (DataFrame) -------------------

def assign_cells_from_groups(boxes: List[Tuple[int,int,int,int]],
                             row_groups: List[List[int]], col_groups: List[List[int]],
                             manual_rows: int = None, manual_cols: int = None) -> Tuple[Dict[int, Tuple[int,int]], int, int]:
    """
    Asigna a cada box (por índice) una celda (r,c) usando grupos.
    Si manual_rows/cols se establecen, re-binning según número deseado.
    Retorna mapping index->(r,c) y shape (n_rows, n_cols).
    """
    if not boxes:
        return {}, 0, 0

    # Mapas por índice -> row_id/col_id basados en posición en grupos ordenados
    row_map = {}
    for r, grp in enumerate(row_groups):
        for idx in grp:
            row_map[idx] = r

    col_map = {}
    for c, grp in enumerate(col_groups):
        for idx in grp:
            col_map[idx] = c

    n_rows_est = len(row_groups)
    n_cols_est = len(col_groups)

    # Re-binning si manual
    if manual_rows is not None and manual_rows > 0 and manual_rows != n_rows_est:
        # Re-ordenar por cy y asignar bins equiespaciados
        cy = np.array([(boxes[i][1] + boxes[i][3]) / 2.0 for i in range(len(boxes))])
        order = np.argsort(cy)
        bin_edges = np.linspace(0, len(order), manual_rows+1).astype(int)
        row_map = {}
        for r in range(manual_rows):
            for pos in order[bin_edges[r]:bin_edges[r+1]]:
                row_map[int(pos)] = r
        n_rows_est = manual_rows

    if manual_cols is not None and manual_cols > 0 and manual_cols != n_cols_est:
        key_x = np.array([boxes[i][0] for i in range(len(boxes))])
        order = np.argsort(key_x)
        bin_edges = np.linspace(0, len(order), manual_cols+1).astype(int)
        col_map = {}
        for c in range(manual_cols):
            for pos in order[bin_edges[c]:bin_edges[c+1]]:
                col_map[int(pos)] = c
        n_cols_est = manual_cols

    assignment = {}
    for i in range(len(boxes)):
        r = row_map.get(i, 0)
        c = col_map.get(i, 0)
        assignment[i] = (int(r), int(c))

    return assignment, n_rows_est, n_cols_est


def ocr_text_from_box(image_bgr: np.ndarray, box: Tuple[int,int,int,int], lang: str = "eng") -> str:
    (x1, y1, x2, y2) = box
    roi = image_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if roi.size == 0:
        return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if not TESSERACT_OK:
        return ""
    try:
        cfg = "--psm 6"
        txt = pytesseract.image_to_string(gray, lang=lang, config=cfg)
        return txt.strip()
    except Exception:
        return ""


def build_dataframe_from_assignments(image_bgr: np.ndarray,
                                     boxes: List[Tuple[int,int,int,int]],
                                     scores: List[float],
                                     assignment: Dict[int, Tuple[int,int]],
                                     n_rows: int, n_cols: int,
                                     do_ocr: bool = True,
                                     lang: str = "eng") -> pd.DataFrame:
    table = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    # En cada celda podemos concatenar múltiples bloques si caen en el mismo (ordenados por x)
    cell_blocks: Dict[Tuple[int,int], List[Tuple[int, str]]] = {}

    texts_cache = {}
    for i, box in enumerate(boxes):
        r, c = assignment[i]
        if do_ocr:
            if i not in texts_cache:
                texts_cache[i] = ocr_text_from_box(image_bgr, box, lang=lang)
            txt = texts_cache[i]
        else:
            txt = ""
        x1, _, x2, _ = box
        cell_blocks.setdefault((r, c), []).append(((x1 + x2) / 2.0, txt))

    for (r, c), items in cell_blocks.items():
        items_sorted = sorted(items, key=lambda t: t[0])
        table[r][c] = " ".join([t[1] for t in items_sorted if t[1].strip() != ""]).strip()

    df = pd.DataFrame(table)
    return df

# ------------------------------- Streamlit UI --------------------------------

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
    type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    model_path = ensure_east_model()

    all_results = []
    zip_buffer = io.BytesIO()
    from zipfile import ZipFile
    zf = ZipFile(zip_buffer, mode="w")

    for up in uploaded_files:
        st.subheader(f"🖼️ {up.name}")
        img_pil = Image.open(up).convert("RGB")
        img_bgr = pil_to_cv2(img_pil)

        # 1) Deskew
        deskewed, dbg_h = hough_deskew(img_bgr, canny1, canny2, hough_th, debug=True)
        st.caption(f"Ángulo estimado: {dbg_h['angle_deg']:.2f}°")

        # 2) EAST
        boxes_east, scores_east, dbg_e = detect_text_blocks_east(deskewed, model_path, score_th, nms_th, debug=True)
        boxes_merged = merge_intersecting_boxes(boxes_east)

        # 3) Grupos filas/columnas
        row_groups, col_groups = group_rows_columns(boxes_merged, row_tol=row_tol, col_tol=col_tol, prefer_left_alignment=prefer_left)

        # 4) Asignación celdas
        if auto_shape:
            assign, n_rows, n_cols = assign_cells_from_groups(boxes_merged, row_groups, col_groups)
        else:
            assign, n_rows, n_cols = assign_cells_from_groups(
                boxes_merged, row_groups, col_groups,
                manual_rows=int(manual_rows), manual_cols=int(manual_cols)
            )

        # 5) DataFrame
        df = build_dataframe_from_assignments(deskewed, boxes_merged, scores_east, assign, n_rows, n_cols, do_ocr=do_ocr, lang=tess_lang)

        # ---------------- Visualizaciones de depuración ----------------
        cols = st.columns(3)
        with cols[0]:
            st.image(img_pil, caption="Original", use_column_width=True)
        with cols[1]:
            st.image(cv2_to_pil(deskewed), caption="Deskewed", use_column_width=True)
        with cols[2]:
            if show_hough and "hough_overlay" in dbg_h:
                st.image(cv2_to_pil(dbg_h["hough_overlay"]), caption="Hough overlay", use_column_width=True)
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
                color = (0, int(50 + 200 * (r / max(1, len(row_groups)-1))), 0)
                for idx in grp:
                    x1, y1, x2, y2 = boxes_merged[idx]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(overlay, f"r{r}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            for c, grp in enumerate(col_groups):
                color = (int(50 + 200 * (c / max(1, len(col_groups)-1))), 0, 0)
                for idx in grp:
                    x1, y1, x2, y2 = boxes_merged[idx]
                    cv2.putText(overlay, f"c{c}", (x1, min(overlay.shape[0]-2, y2+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        st.image(cv2_to_pil(overlay), caption="Detecciones / Grupos", use_column_width=True)

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
            st.download_button("⬇️ Descargar CSV", data=csv_bytes, file_name=f"{os.path.splitext(up.name)[0]}_table.csv", mime="text/csv")
        with c2:
            st.download_button("⬇️ Descargar Excel", data=excel_bytes, file_name=f"{os.path.splitext(up.name)[0]}_table.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Guardar CSV de bloques detectados
        blocks_df = pd.DataFrame([
            {
                "image_name": up.name,
                "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                "score": scores_east[min(i, len(scores_east)-1)] if len(scores_east) else None,
                "row": assign[i][0], "col": assign[i][1]
            }
            for i, b in enumerate(boxes_merged)
        ])
        blocks_csv = blocks_df.to_csv(index=False).encode("utf-8")
        with c3:
            st.download_button("⬇️ Bloques detectados (CSV)", data=blocks_csv, file_name=f"{os.path.splitext(up.name)[0]}_blocks.csv", mime="text/csv")

        # Agregar a ZIP
        zf.writestr(f"{os.path.splitext(up.name)[0]}_table.csv", csv_bytes)
        zf.writestr(f"{os.path.splitext(up.name)[0]}_table.xlsx", excel_bytes)
        zf.writestr(f"{os.path.splitext(up.name)[0]}_blocks.csv", blocks_csv)

        all_results.append({
            "name": up.name,
            "angle": dbg_h.get("angle_deg", 0.0),
            "n_boxes": len(boxes_merged),
            "n_rows": n_rows,
            "n_cols": n_cols,
        })

    zf.close()
    st.markdown("---")
    st.subheader("📦 Descarga masiva")
    st.download_button("⬇️ Descargar todo (ZIP)", data=zip_buffer.getvalue(), file_name="tabular_app_outputs.zip", mime="application/zip")

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
