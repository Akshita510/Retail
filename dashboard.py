"""
Streamlit UI: upload catalog Excel, then run either duplication or anomaly analysis (one at a time).

Run: python -m poetry run streamlit run dashboard.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from retail_analyzer.config import AnalyzerConfig
from retail_analyzer.excel_io import (
    guess_category_column,
    guess_description_column,
    guess_image_column,
    guess_price_column,
    load_excel_bytes,
)
from retail_analyzer.merge import dataframe_to_excel_bytes, merge_visual_duplicate_rows
from retail_analyzer.retail_anomaly_detection import ANOMALY_HELP_BULLETS
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
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="stApp"] { font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important; }
            .block-container { padding-top: 1rem; padding-bottom: 2.5rem; max-width: 1100px; }
            .hero {
                background: linear-gradient(135deg, #0d9488 0%, #0f766e 45%, #155e75 100%);
                border-radius: 20px;
                padding: 1.5rem 1.75rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 16px 48px -20px rgba(13, 148, 136, 0.55);
                position: relative; overflow: hidden;
            }
            .hero::after {
                content: ""; position: absolute; right: -40px; top: -40px; width: 180px; height: 180px;
                border-radius: 50%; background: rgba(255,255,255,0.08); pointer-events: none;
            }
            .hero h1 {
                color: #f0fdfa !important; font-weight: 700; font-size: 1.65rem; margin: 0 0 0.4rem 0;
                letter-spacing: -0.03em; border: none;
            }
            .hero p { color: #ccfbf1; margin: 0; font-size: 0.95rem; line-height: 1.5; opacity: 0.95; }
            .hero-anom {
                background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 50%, #4c1d95 100%) !important;
                box-shadow: 0 16px 48px -20px rgba(91, 33, 182, 0.45) !important;
            }
            .card-visual {
                background: linear-gradient(180deg, #fafaf9 0%, #f5f5f4 100%);
                border: 1px solid #e7e5e4; border-radius: 16px; padding: 1rem 1.15rem; margin: 0.5rem 0 1rem 0;
            }
            .section-title {
                font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
                color: #a8a29e; margin: 1.25rem 0 0.6rem 0;
            }
            div[data-testid="stExpander"] {
                background: #fafaf9; border: 1px solid #e7e5e4 !important; border-radius: 14px !important;
                margin-bottom: 0.5rem;
            }
            div[data-testid="stMetric"] {
                background: #fff; border: 1px solid #e7e5e4; border-radius: 14px; padding: 0.75rem 1rem;
            }
            .action-bar {
                background: #f5f5f4; border: 1px solid #e7e5e4; border-radius: 14px; padding: 1rem 1.1rem; margin: 1rem 0;
            }
            div[data-testid="stImage"] img {
                border-radius: 12px !important; border: 1px solid #e7e5e4 !important;
            }
            .badge {
                display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px;
                font-size: 0.72rem; font-weight: 600;
            }
            .badge-dup { background: #ccfbf1; color: #0f766e; }
            .badge-anom { background: #ffedd5; color: #9a3412; }
            .badge-both { background: #ede9fe; color: #5b21b6; }
            .detail-grid { font-size: 0.88rem; line-height: 1.6; color: #44403c; }
            hr.soft { border: none; border-top: 1px solid #e7e5e4; margin: 1rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(mode: str | None = None) -> None:
    if mode is None:
        st.markdown(
            """
            <div class="hero">
                <h1>✨ Catalog intelligence</h1>
                <p>Two tools in one: find duplicate product images, or scan your sheet for retail data issues. Pick a mode below — only one runs at a time.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    titles = {
        "duplication": (
            "🖼️ Duplication analysis",
            "Same image URL + similar description & category (cosine ≥ 0.8). Groups get a duplicate_group_id.",
        ),
        "anomaly": (
            "📊 Anomaly analysis",
            "Image–text mismatches, price outliers (±3σ by category), missing fields, and invalid formats.",
        ),
    }
    title, sub = titles.get(mode, ("Analysis", ""))
    extra = " hero-anom" if mode == "anomaly" else ""
    st.markdown(
        f"""
        <div class="hero{extra}">
            <h1>{title}</h1>
            <p>{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _needs_review_mask(result: pd.DataFrame, mode: str) -> pd.Series:
    gid_col = "duplicate_group_id" if "duplicate_group_id" in result.columns else "similar_image_group_id"
    dup = result[gid_col] >= 0 if gid_col in result.columns else pd.Series(False, index=result.index)
    if "anomaly_flag" in result.columns:
        anom = result["anomaly_flag"].apply(lambda x: bool(x) if not isinstance(x, str) else str(x).lower() in ("true", "yes", "1"))
    else:
        af = result["anomaly_flags"].fillna("").astype(str).str.strip()
        anom = af.str.len() > 0
    if mode == "duplication":
        return dup
    if mode == "anomaly":
        return anom
    return dup | anom


def _row_numbers(df: pd.DataFrame) -> dict[object, int]:
    return {ix: i + 1 for i, ix in enumerate(df.index)}


def _sync_dup_editor_state(result: pd.DataFrame) -> None:
    run_id = st.session_state.get("analysis_run_id", 0)
    if st.session_state.get("_dup_run_id") != run_id:
        st.session_state["dup_sheet_df"] = result.copy()
        st.session_state["_dup_run_id"] = run_id


def _duplication_summary_table(ed: pd.DataFrame) -> pd.DataFrame:
    row_no = _row_numbers(ed)
    gid_col = "duplicate_group_id" if "duplicate_group_id" in ed.columns else "similar_image_group_id"
    dup = ed[ed[gid_col] >= 0].copy()
    if len(dup) == 0:
        return pd.DataFrame(columns=["Row No", "Match strength", "Summary"])
    out = pd.DataFrame(
        {
            "Row No": [row_no[ix] for ix in dup.index],
            "Match strength": dup["similarity_score"].values,
            "Summary": dup["similarity_comment"].fillna("").astype(str).values,
        }
    )
    return out


def _clear_editor_session() -> None:
    st.session_state.pop("dup_sheet_df", None)
    st.session_state.pop("_dup_run_id", None)
    st.session_state.pop("anom_sheet_df", None)
    st.session_state.pop("_anom_run_id", None)
    st.session_state.pop("analysis_run_id", None)
    st.session_state.pop("merged_excel_bytes", None)
    st.session_state.pop("merged_excel_name", None)


@st.dialog("Merge duplicate groups into one row per group")
def merge_duplicate_dialog() -> None:
    """Uses the current sheet (including any edits) to build a merged catalog."""
    ed = st.session_state.get("dup_sheet_df")
    img_col = st.session_state.get("analysis_image_col")
    if ed is None or not img_col:
        st.error("No data to merge.")
        return
    if img_col not in ed.columns:
        st.error(f"Image column «{img_col}» is not in the sheet.")
        return
    gid_col = "duplicate_group_id" if "duplicate_group_id" in ed.columns else "similar_image_group_id"
    if gid_col not in ed.columns:
        st.error("Missing duplicate group column. Run analysis again.")
        return
    dup = ed[ed[gid_col] >= 0]
    n_g = int(dup[gid_col].nunique()) if len(dup) else 0
    st.markdown(
        f"**{len(dup)} rows** in **{n_g} duplicate group(s)** will be merged into **{n_g} rows**. "
        "Images are combined into one cell; differing text fields are joined with ` | `."
    )
    st.caption("Rows that are not in a duplicate group stay as they are.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("Approve & build Excel", type="primary", use_container_width=True):
            merged = merge_visual_duplicate_rows(ed, img_col)
            st.session_state["merged_excel_bytes"] = dataframe_to_excel_bytes(
                merged, image_column=img_col
            )
            st.session_state["merged_excel_name"] = st.session_state.get(
                "_merge_dup_filename", "merged_catalog.xlsx"
            )
            st.rerun()


DEFAULT_CACHE_DIR = Path(".retail_analyzer_cache")


def _anomaly_issue_counts(ed: pd.DataFrame) -> pd.Series:
    from collections import Counter

    c: Counter[str] = Counter()
    col = "anomaly_type" if "anomaly_type" in ed.columns else "anomaly_flags"
    if col not in ed.columns:
        return pd.Series(dtype=float)
    for raw in ed[col].fillna(""):
        for part in str(raw).split(";"):
            p = part.strip()
            if p:
                c[p] += 1
    if not c:
        return pd.Series(dtype=float)
    s = pd.Series(dict(c)).sort_values(ascending=False)
    s.index = s.index.map(lambda x: x.replace("_", " ").title())
    return s


def _result_stats(result: pd.DataFrame, mode: str) -> tuple[int, int, int, int, int]:
    total = len(result)
    gid_col = "duplicate_group_id" if "duplicate_group_id" in result.columns else "similar_image_group_id"
    dup_mask = result[gid_col] >= 0 if gid_col in result.columns else pd.Series(False, index=result.index)
    rows_dup = int(dup_mask.sum())
    n_groups = int(result.loc[dup_mask, gid_col].nunique()) if rows_dup else 0
    if "anomaly_flag" in result.columns:
        rows_flag = int(
            result["anomaly_flag"].apply(
                lambda x: bool(x) if not isinstance(x, str) else str(x).lower() in ("true", "yes", "1")
            ).sum()
        )
    else:
        af = result["anomaly_flags"].fillna("").astype(str).str.strip()
        rows_flag = int((af.str.len() > 0).sum())
    review = int(_needs_review_mask(result, mode).sum())
    return total, review, rows_dup, n_groups, rows_flag


def main() -> None:
    _inject_styles()
    mode = st.session_state.get("analysis_mode")

    if mode is None:
        _hero(None)
        st.markdown('<p class="section-title">Start here</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Duplication analysis", type="primary", use_container_width=True):
                st.session_state["analysis_mode"] = "duplication"
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("_upload_key", None)
                _clear_editor_session()
                st.rerun()
        with c2:
            if st.button("Anomaly analysis", type="primary", use_container_width=True):
                st.session_state["analysis_mode"] = "anomaly"
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("_upload_key", None)
                _clear_editor_session()
                st.rerun()
        st.caption("Pick one workflow, then upload your workbook. Duplication and anomaly detection do not run together.")
        return

    assert mode in ("duplication", "anomaly")
    _hero(mode)
    back_c, _ = st.columns((1, 4))
    with back_c:
        if st.button("← Change analysis type"):
            st.session_state.pop("analysis_mode", None)
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("_upload_key", None)
            _clear_editor_session()
            st.rerun()

    if mode == "duplication":
        up = st.file_uploader(
            "Workbook (.xlsx)",
            type=["xlsx"],
            help="Uses the first sheet. Image URLs are detected automatically from column names.",
        )
        similarity_threshold = st.slider("Image similarity threshold", 0.5, 1.0, 0.86, 0.01)
        if not up:
            st.caption("Upload a file to continue.")
            return

        data = up.getvalue()
        file_key = f"{up.name}:{len(data)}:dup:{similarity_threshold}"
        if st.session_state.get("_upload_key") != file_key:
            st.session_state["_upload_key"] = file_key
            st.session_state.pop("analysis_result", None)
            _clear_editor_session()

        try:
            df0 = load_excel_bytes(data, sheet=0)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            return

        cols = [str(c) for c in df0.columns]
        img_col = guess_image_column(cols)
        if not img_col:
            st.error(
                "Could not detect an image column automatically. "
                "Rename a column to include something like image, photo, url, or link."
            )
            return

        st.caption(f"File: **{up.name}** · first sheet · image column: **{img_col}**")

        if st.button("Run duplication analysis", type="primary", use_container_width=True):
            cfg = AnalyzerConfig(
                image_similarity_threshold=float(similarity_threshold),
                image_neighbor_k=15,
                cache_dir=DEFAULT_CACHE_DIR,
                enable_context_anomalies=False,
            )
            with st.spinner("Analyzing…"):
                try:
                    result = analyze_dataframe(
                        df0,
                        image_column=img_col,
                        price_column=None,
                        description_column=None,
                        skip_images=False,
                        skip_anomalies=True,
                        config=cfg,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception("analyze_dataframe failed")
                    st.error(str(e))
                    return
            st.session_state["analysis_run_id"] = (st.session_state.get("analysis_run_id") or 0) + 1
            st.session_state["analysis_result"] = result
            st.session_state["analysis_image_col"] = img_col
            st.session_state["dup_upload_name"] = up.name
            st.session_state.pop("merged_excel_bytes", None)
            st.session_state.pop("merged_excel_name", None)
            base = Path(up.name).stem if up.name else "catalog"
            st.session_state["_merge_dup_filename"] = f"{base}_merged.xlsx"

    else:
        up = st.file_uploader(
            "Workbook (.xlsx)",
            type=["xlsx"],
            help="Uses the first sheet. Columns are inferred from headers.",
        )
        if not up:
            st.caption("Upload a file to continue.")
            return

        data = up.getvalue()
        file_key = f"{up.name}:{len(data)}:anom"
        if st.session_state.get("_upload_key") != file_key:
            st.session_state["_upload_key"] = file_key
            st.session_state.pop("analysis_result", None)
            _clear_editor_session()

        try:
            df0 = load_excel_bytes(data, sheet=0)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            return

        cols = [str(c) for c in df0.columns]
        img_col = guess_image_column(cols)
        price_col = guess_price_column(cols)
        desc_col = guess_description_column(cols)
        cat_col = guess_category_column(cols)
        st.caption(
            f"File: **{up.name}** · first sheet · "
            f"image: **{img_col or '—'}** · price: **{price_col or '—'}** · "
            f"description: **{desc_col or '—'}** · category: **{cat_col or '—'}**"
        )

        if st.button("Run anomaly analysis", type="primary", use_container_width=True):
            cfg = AnalyzerConfig(
                image_similarity_threshold=0.86,
                image_neighbor_k=15,
                cache_dir=DEFAULT_CACHE_DIR,
                enable_context_anomalies=False,
            )
            with st.spinner("Analyzing…"):
                try:
                    result = analyze_dataframe(
                        df0,
                        image_column=img_col,
                        price_column=price_col,
                        description_column=desc_col,
                        skip_images=True,
                        skip_anomalies=False,
                        config=cfg,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception("analyze_dataframe failed")
                    st.error(str(e))
                    return
            st.session_state["analysis_run_id"] = (st.session_state.get("analysis_run_id") or 0) + 1
            st.session_state["analysis_result"] = result
            st.session_state["analysis_image_col"] = img_col

    if "analysis_result" not in st.session_state:
        return

    result = st.session_state["analysis_result"]
    if mode == "duplication":
        upload_name = str(st.session_state.get("dup_upload_name") or "catalog.xlsx")
        _render_duplication_results(result, upload_name)
    else:
        _render_anomaly_results(result)


def _sync_anom_editor_state(result: pd.DataFrame) -> None:
    run_id = st.session_state.get("analysis_run_id", 0)
    if st.session_state.get("_anom_run_id") != run_id:
        st.session_state["anom_sheet_df"] = result.copy()
        st.session_state["_anom_run_id"] = run_id


def _render_duplication_results(result: pd.DataFrame, upload_name: str) -> None:
    _sync_dup_editor_state(result)
    ed = st.session_state["dup_sheet_df"]
    total, n_review, _rows_dup, n_groups, _ = _result_stats(result, "duplication")

    st.markdown('<p class="section-title">Summary</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Rows in file", f"{total:,}")
    with m2:
        st.metric("Rows in duplicate groups", f"{n_review:,}")
    with m3:
        st.metric("Duplicate groups", f"{n_groups:,}")

    st.markdown('<p class="section-title">Duplicate rows</p>', unsafe_allow_html=True)
    st.caption("Row numbers, match strength, and a plain-language explanation for each duplicate.")
    summary = _duplication_summary_table(ed)
    st.dataframe(summary, use_container_width=True, hide_index=True, height=min(360, 80 + 32 * max(len(summary), 1)))

    st.markdown('<p class="section-title">Full sheet</p>', unsafe_allow_html=True)
    st.caption("Edit cells below; download exports this version.")
    run_id = int(st.session_state.get("analysis_run_id") or 0)
    edited = st.data_editor(
        ed,
        key=f"dup_ed_{run_id}",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        height=min(520, 120 + 28 * min(len(ed), 18)),
    )
    st.session_state["dup_sheet_df"] = edited

    out_bytes = dataframe_to_excel_bytes(
        st.session_state["dup_sheet_df"],
        image_column=st.session_state.get("analysis_image_col"),
    )
    base = Path(upload_name).stem if upload_name else "catalog"
    dl1, dl2, dl3 = st.columns((1, 1, 1))
    with dl1:
        st.download_button(
            "Download current sheet (.xlsx)",
            data=out_bytes,
            file_name=f"{base}_edited.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        has_dups = n_groups > 0
        if st.button(
            "Merge duplicate groups…",
            use_container_width=True,
            disabled=not has_dups,
            help="Opens a confirmation dialog, then you can download one row per duplicate group.",
        ):
            merge_duplicate_dialog()
    with dl3:
        if st.session_state.get("merged_excel_bytes"):
            st.download_button(
                "Download merged catalog (.xlsx)",
                data=st.session_state["merged_excel_bytes"],
                file_name=st.session_state.get("merged_excel_name", "merged_catalog.xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.caption("Merged file appears here after you approve in the dialog.")


def _render_anomaly_results(result: pd.DataFrame) -> None:
    _sync_anom_editor_state(result)
    ed = st.session_state["anom_sheet_df"]
    total, _, _, _, rows_flag = _result_stats(result, "anomaly")
    row_no = _row_numbers(ed)

    st.markdown('<p class="section-title">Summary</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Rows in file", f"{total:,}")
    with m2:
        st.metric("Rows flagged", f"{rows_flag:,}")
    with m3:
        pct = f"{100.0 * rows_flag / total:.1f}%" if total else "0%"
        st.metric("Share flagged", pct)

    counts = _anomaly_issue_counts(ed)
    if len(counts) > 0:
        st.markdown(
            '<p class="section-title" style="margin-top:0.5rem">Issue mix</p>',
            unsafe_allow_html=True,
        )
        st.caption("How often each rule fired (a row can match more than one).")
        st.bar_chart(counts)

    st.markdown('<p class="section-title">Flagged rows</p>', unsafe_allow_html=True)
    mask = _needs_review_mask(ed, "anomaly")
    flagged = ed.loc[mask].copy()
    if len(flagged) == 0:
        st.success("No anomalies detected with the current rules.")
    else:
        view = pd.DataFrame(
            {
                "Row No": [row_no[ix] for ix in flagged.index],
                "anomaly_flag": flagged["anomaly_flag"].tolist(),
                "anomaly_type": (
                    flagged["anomaly_type"].tolist()
                    if "anomaly_type" in flagged.columns
                    else [""] * len(flagged)
                ),
                "reason": (
                    flagged["reason"].tolist()
                    if "reason" in flagged.columns
                    else flagged["anomaly_reason"].tolist()
                ),
                "anomaly_score": flagged["anomaly_score"].tolist(),
            }
        )
        st.dataframe(view, use_container_width=True, hide_index=True)

    with st.expander("What do we check?"):
        st.markdown("\n".join(f"- {line}" for line in ANOMALY_HELP_BULLETS))

    st.markdown('<p class="section-title">Full sheet</p>', unsafe_allow_html=True)
    st.caption("Edit cells below; download exports this version.")
    run_id = int(st.session_state.get("analysis_run_id") or 0)
    edited = st.data_editor(
        ed,
        key=f"anom_ed_{run_id}",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        height=min(520, 120 + 28 * min(len(ed), 18)),
    )
    st.session_state["anom_sheet_df"] = edited

    out_bytes = dataframe_to_excel_bytes(
        st.session_state["anom_sheet_df"],
        image_column=st.session_state.get("analysis_image_col"),
    )
    st.download_button(
        "Download Excel (.xlsx)",
        data=out_bytes,
        file_name="catalog_anomaly_edited.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
