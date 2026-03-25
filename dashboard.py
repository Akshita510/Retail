"""
Streamlit UI: upload catalog Excel, run analysis, view rows with images and flags.

Run: python -m poetry run streamlit run dashboard.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.excel_io import list_sheet_names_bytes, load_excel_bytes, parse_image_urls
from retail_analyzer.pipeline import analyze_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Catalog intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="stApp"] {
                font-family: 'Outfit', system-ui, sans-serif !important;
            }
            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 3rem;
                max-width: 1280px;
            }
            .hero-wrap {
                background: linear-gradient(135deg, #0f766e 0%, #115e59 48%, #134e4a 100%);
                border-radius: 16px;
                padding: 1.75rem 2rem 1.85rem;
                margin-bottom: 1.75rem;
                box-shadow: 0 12px 40px -12px rgba(15, 118, 110, 0.45);
            }
            .hero-wrap h1 {
                color: #ecfdf5 !important;
                font-weight: 700;
                font-size: 1.85rem;
                letter-spacing: -0.02em;
                margin: 0 0 0.35rem 0;
                border: none;
            }
            .hero-sub {
                color: #99f6e4;
                font-size: 1rem;
                font-weight: 400;
                line-height: 1.5;
                margin: 0;
                opacity: 0.95;
            }
            .panel-label {
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #78716c;
                margin-bottom: 0.65rem;
            }
            div[data-testid="stExpander"] {
                background: #ffffff;
                border: 1px solid #e7e5e4 !important;
                border-radius: 12px !important;
                margin-bottom: 0.65rem;
                box-shadow: 0 1px 3px rgba(28, 25, 23, 0.04);
            }
            div[data-testid="stExpander"] details {
                border: none !important;
            }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e7e5e4;
                border-radius: 12px;
                padding: 0.65rem 0.85rem;
                box-shadow: 0 1px 2px rgba(28, 25, 23, 0.04);
            }
            div[data-testid="stMetric"] label {
                font-size: 0.8rem !important;
            }
            .stButton > button[kind="primary"] {
                border-radius: 10px;
                font-weight: 600;
                padding: 0.5rem 1.25rem;
                width: 100%;
            }
            section[data-testid="stFileUploader"] {
                border: 2px dashed #d6d3d1;
                border-radius: 12px;
                padding: 0.5rem;
                background: #ffffff;
            }
            section[data-testid="stFileUploader"]:hover {
                border-color: #0f766e;
                background: #f0fdfa;
            }
            div[data-testid="stImage"] img {
                border-radius: 10px !important;
                border: 1px solid #e7e5e4 !important;
                box-shadow: 0 4px 14px -4px rgba(28, 25, 23, 0.12);
            }
            div[data-testid="stCaption"] {
                font-size: 0.72rem !important;
                color: #78716c !important;
                word-break: break-all;
                line-height: 1.35 !important;
            }
            .badge {
                display: inline-block;
                padding: 0.15rem 0.5rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .badge-dup { background: #ccfbf1; color: #115e59; }
            .badge-ok { background: #e7e5e4; color: #57534e; }
            .badge-warn { background: #ffedd5; color: #9a3412; }
            .detail-grid { font-size: 0.9rem; line-height: 1.65; color: #44403c; }
            .detail-grid strong { color: #1c1917; }
            hr.soft { border: none; border-top: 1px solid #e7e5e4; margin: 1.25rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero() -> None:
    st.markdown(
        """
        <div class="hero-wrap">
            <h1>Catalog intelligence</h1>
            <p class="hero-sub">
                Find visually similar listings and data anomalies from your retail spreadsheet—results stay in the browser; no export required.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _sheet_names(data: bytes) -> list[str]:
    return list_sheet_names_bytes(data)


def _result_stats(result: pd.DataFrame) -> tuple[int, int, int, int]:
    """rows in dup groups, unique dup group ids, rows with flags, total rows."""
    total = len(result)
    dup_mask = result["similar_image_group_id"] >= 0
    rows_dup = int(dup_mask.sum())
    n_groups = int(result.loc[dup_mask, "similar_image_group_id"].nunique()) if rows_dup else 0
    af = result["anomaly_flags"].fillna("").astype(str).str.strip()
    rows_flag = int((af.str.len() > 0).sum())
    return rows_dup, n_groups, rows_flag, total


def main() -> None:
    _inject_styles()
    _hero()

    up = st.file_uploader(
        "Drop your workbook here",
        type=["xlsx"],
        help="Supports multi-image cells (comma-separated URLs).",
    )
    if not up:
        st.markdown('<p class="panel-label">Get started</p>', unsafe_allow_html=True)
        st.info("Upload an **.xlsx** file to configure the sheet, map columns, and run analysis.")
        return

    data = up.getvalue()
    file_key = f"{up.name}:{len(data)}"
    if st.session_state.get("_upload_key") != file_key:
        st.session_state["_upload_key"] = file_key
        st.session_state.pop("analysis_result", None)

    try:
        sheets = _sheet_names(data)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read workbook: {e}")
        return

    st.markdown('<p class="panel-label">Analysis setup</p>', unsafe_allow_html=True)
    cfg_row = st.columns((1.1, 1.1, 1), gap="medium")
    with cfg_row[0]:
        sheet = st.selectbox("Sheet", options=sheets, index=0)
    with cfg_row[1]:
        skip_images = st.checkbox("Skip images (rules only)", value=False)
    with cfg_row[2]:
        st.caption(f"**File:** {up.name}")

    sheet_arg: str | int = sheet
    df0 = load_excel_bytes(data, sheet=sheet_arg)
    cols = [str(c) for c in df0.columns]

    map_row = st.columns(3, gap="medium")
    with map_row[0]:
        img_default = next(
            (c for c in cols if "image" in c.lower() and "url" in c.lower()),
            cols[0] if cols else "",
        )
        image_column = st.selectbox(
            "Image URL column",
            options=cols,
            index=cols.index(img_default) if img_default in cols else 0,
        )
    with map_row[1]:
        price_opts = ["(none)"] + cols
        pc = st.selectbox("Price column", options=price_opts, index=0)
        price_column = None if pc == "(none)" else pc
    with map_row[2]:
        desc_guess = next(
            (c for c in cols if "product" in c.lower() and "name" in c.lower()),
            cols[0] if cols else "",
        )
        desc_opts = ["(auto)"] + cols
        di = desc_opts.index(desc_guess) if desc_guess in desc_opts else 0
        dc = st.selectbox("Description column", options=desc_opts, index=di)
        description_column = None if dc == "(auto)" else dc

    adv = st.columns((1.4, 1), gap="medium")
    with adv[0]:
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.86,
            step=0.01,
            help="Higher values mean stricter “same product” grouping.",
        )
    with adv[1]:
        cache_dir = st.text_input("Image cache folder", value=".retail_analyzer_cache")

    run = st.button("Run analysis", type="primary", use_container_width=True)

    if run:
        cfg = AnalyzerConfig(
            image_similarity_threshold=float(similarity_threshold),
            image_neighbor_k=15,
            cache_dir=Path(cache_dir),
        )
        with st.spinner("Running analysis… (first run may download the CLIP model)"):
            try:
                result = analyze_dataframe(
                    df0,
                    image_column=image_column,
                    price_column=price_column,
                    description_column=description_column,
                    skip_images=skip_images,
                    config=cfg,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("analyze_dataframe failed")
                st.error(str(e))
                return
        st.session_state["analysis_result"] = result
        st.session_state["analysis_image_col"] = image_column

    if "analysis_result" not in st.session_state:
        st.markdown('<hr class="soft">', unsafe_allow_html=True)
        st.markdown('<p class="panel-label">Preview</p>', unsafe_allow_html=True)
        st.dataframe(df0.head(20), use_container_width=True, hide_index=True)
        return

    result = st.session_state["analysis_result"]
    img_col = st.session_state.get("analysis_image_col", image_column)
    _render_results(result, img_col)


def _render_results(result: pd.DataFrame, image_column: str) -> None:
    rows_dup, n_groups, rows_flag, total = _result_stats(result)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.markdown('<p class="panel-label">Results</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        st.metric("Rows analyzed", f"{total:,}")
    with m2:
        st.metric("Visual duplicate rows", f"{rows_dup:,}", help="Rows in a similar-image group")
    with m3:
        st.metric("Duplicate groups", f"{n_groups:,}")
    with m4:
        st.metric("Rows with flags", f"{rows_flag:,}")

    filt = st.columns((1, 1, 1.4), gap="medium")
    with filt[0]:
        show_dup = st.toggle("Only duplicate groups", value=False)
    with filt[1]:
        show_anom = st.toggle("Only flagged rows", value=False)
    with filt[2]:
        q = st.text_input("Search in table", placeholder="Filter by any visible text…")

    view = result.copy()
    if show_dup:
        view = view[view["similar_image_group_id"] >= 0]
    if show_anom:
        af = view["anomaly_flags"].fillna("").astype(str).str.strip()
        view = view[af.str.len() > 0]
    if q.strip():
        mask = view.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False)).any(axis=1)
        view = view[mask]

    show_cols = [c for c in view.columns if not str(c).startswith("_")]
    st.dataframe(
        view[show_cols],
        use_container_width=True,
        height=min(520, 72 + 36 * min(len(view), 14)),
        hide_index=True,
    )

    st.markdown('<p class="panel-label">Row gallery</p>', unsafe_allow_html=True)
    st.caption("Expand a row for full fields and product images.")

    display_ix = list(view.index)
    for i, ix in enumerate(display_ix):
        row = view.loc[ix]
        gid = row.get("similar_image_group_id", -1)
        flags = str(row.get("anomaly_flags", "") or "").strip()
        try:
            gid_i = int(gid)
        except (TypeError, ValueError):
            gid_i = -1

        if gid_i >= 0:
            badge = f'<span class="badge badge-dup">Group {gid_i}</span>'
        elif flags:
            badge = '<span class="badge badge-warn">Flagged</span>'
        else:
            badge = '<span class="badge badge-ok">Clean</span>'

        label = f"Row {i + 1} · index {ix}"
        with st.expander(f"{label}", expanded=False):
            st.markdown(badge, unsafe_allow_html=True)
            if flags:
                st.markdown(f"**Flags:** `{flags}`")

            detail_cols = [
                c
                for c in view.columns
                if c != image_column and not str(c).startswith("_")
            ]
            parts = []
            for c in detail_cols[:16]:
                val = row[c]
                if pd.isna(val):
                    val = "—"
                parts.append(f"<strong>{c}</strong>: {val}")
            st.markdown(
                '<div class="detail-grid">' + "<br>".join(parts) + "</div>",
                unsafe_allow_html=True,
            )

            urls = parse_image_urls(row.get(image_column))
            if not urls:
                st.caption("No image URLs in this cell.")
                continue

            n = min(len(urls), 8)
            ncols = min(4, max(1, n))
            img_cols = st.columns(ncols)
            for j in range(n):
                u = urls[j]
                cap = u[:100] + ("…" if len(u) > 100 else "")
                with img_cols[j % ncols]:
                    try:
                        st.image(u, caption=cap, use_container_width=True)
                    except Exception:  # noqa: BLE001
                        st.caption("Could not load image.")


if __name__ == "__main__":
    main()
