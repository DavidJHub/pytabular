"""Command line demo for the table extraction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from pipeline import PipelineConfig, TableExtractionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo interactiva de extracción de tablas")
    parser.add_argument(
        "--image",
        "-i",
        nargs="+",
        required=True,
        help="Ruta(s) de imagen con tablas",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="Directorio donde se almacenarán los resultados",
    )
    parser.add_argument("--score", type=float, default=0.5, help="Umbral de score para EAST")
    parser.add_argument("--nms", type=float, default=0.3, help="Umbral IoU para NMS")
    parser.add_argument("--row-tol", type=int, default=12, help="Tolerancia vertical para agrupar filas")
    parser.add_argument("--col-tol", type=int, default=24, help="Tolerancia horizontal para agrupar columnas")
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=18,
        help="Separación horizontal máxima (px) para unir cajas consecutivas",
    )
    parser.add_argument(
        "--merge-row",
        type=int,
        default=10,
        help="Tolerancia vertical (px) para decidir si dos cajas pertenecen a la misma línea",
    )
    parser.add_argument("--prefer-left", action="store_true", help="Priorizar alineación por el borde izquierdo")
    parser.add_argument("--ocr", action="store_true", help="Activar OCR con Tesseract si está disponible")
    parser.add_argument("--lang", default="eng", help="Idioma para OCR")
    parser.add_argument("--manual-rows", type=int, default=None, help="Forzar número de filas en la tabla")
    parser.add_argument("--manual-cols", type=int, default=None, help="Forzar número de columnas en la tabla")
    return parser.parse_args()


def load_image(path: Path) -> cv2.Mat:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return image


def run_demo(image_paths: List[Path], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TableExtractionPipeline()

    summaries = []
    for img_path in image_paths:
        image = load_image(img_path)
        cfg = PipelineConfig(
            score_thresh=args.score,
            nms_thresh=args.nms,
            row_tol=args.row_tol,
            col_tol=args.col_tol,
            prefer_left_alignment=args.prefer_left,
            merge_gap_tol=args.merge_gap,
            merge_row_tol=args.merge_row,
            do_ocr=args.ocr,
            ocr_lang=args.lang,
            manual_rows=args.manual_rows,
            manual_cols=args.manual_cols,
        )

        result, _ = pipeline.process(image, config=cfg, debug=False)

        base_name = img_path.stem
        image_output_dir = output_dir / base_name
        image_output_dir.mkdir(exist_ok=True, parents=True)

        deskewed_path = image_output_dir / f"{base_name}_deskewed.png"
        cv2.imwrite(str(deskewed_path), result.deskewed)

        table_csv = image_output_dir / f"{base_name}_table.csv"
        table_xlsx = image_output_dir / f"{base_name}_table.xlsx"
        boxes_csv = image_output_dir / f"{base_name}_boxes.csv"

        result.table.to_csv(table_csv, index=False)
        with pd.ExcelWriter(table_xlsx, engine="openpyxl") as writer:
            result.table.to_excel(writer, index=False, sheet_name="Sheet1")

        result.boxes_df.to_csv(boxes_csv, index=False)

        summaries.append(
            {
                "image": str(img_path),
                "deskew_angle": result.angle_deg,
                "boxes": len(result.boxes_df),
                "rows": result.n_rows,
                "cols": result.n_cols,
                "table_csv": str(table_csv),
                "table_xlsx": str(table_xlsx),
                "boxes_csv": str(boxes_csv),
                "deskewed_image": str(deskewed_path),
            }
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summaries, fp, ensure_ascii=False, indent=2)

    print("\nResumen del procesamiento:")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    image_paths = [Path(p) for p in args.image]
    run_demo(image_paths, args.output, args)


if __name__ == "__main__":
    main()
