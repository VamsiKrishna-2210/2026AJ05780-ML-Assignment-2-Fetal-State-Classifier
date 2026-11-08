"""Score a test CSV with the five saved fetal-state classifiers.

The page loads the joblib files in model/. Fitting happens only in train_models.py.
"""

import base64
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from common import (
    CLASS_ORDER,
    FEATURES,
    MODEL_DIR,
    MODEL_FILENAMES,
    TARGET,
    TEST_PATH,
    aligned_probabilities,
    compute_metrics,
    prepare_table,
)

PAGE_CSS = """
<style>
.stApp, .stApp * { font-family: "Inter", sans-serif !important; }
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * {
  font-family: "Material Symbols Rounded" !important;
}
.block-container {
  max-width: 1500px;
  padding-top: 5.5rem;
  padding-left: 3rem;
  padding-right: 3rem;
}
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.main-title {
  font-size: 2.35rem;
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1.1;
  margin: 0.4rem 0 2rem;
  color: #17212b;
}
.note { color: #5c6d7a; font-size: 0.95rem; line-height: 1.55; margin: 0.35rem 0 1.1rem; }
.model-note {
  margin: 0 0 0.55rem;
  color: #5c6d7a;
  font-size: 0.95rem;
  line-height: 1.4;
}
.section {
  font-size: 1.02rem;
  font-weight: 650;
  margin: 1.35rem 0 0.6rem;
}
.rank {
  display: grid;
  grid-template-columns: 26px 1fr;
  gap: 8px;
  box-sizing: border-box;
  background: #ffffff;
  border: 1px solid #d5dee6;
  border-radius: 8px;
  padding: 10px 12px;
  margin: 0;
}
.rank.is-on { background: #1e3344; border-color: #1e3344; color: #f4f7f9; }
.rank h3 { margin: 0 0 5px; font-size: 0.98rem; font-weight: 650; }
.rank-no { font-size: 0.78rem; color: #6b7c89; padding-top: 2px; }
.rank.is-on .rank-no { color: #b7c6d1; }
.ms { display: flex; flex-wrap: wrap; gap: 3px 12px; }
.ms .m { font-variant-numeric: tabular-nums; font-size: 0.8rem; }
.ms i {
  font-style: normal;
  display: block;
  color: #6b7c89;
  font-size: 0.64rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.rank.is-on .ms i { color: #b7c6d1; }
.ms .best { font-weight: 750; color: #0f5c4c; }
.rank.is-on .ms .best { color: #f2c14e; }
.grid { border-collapse: collapse; width: 100%; }
.grid th, .grid td {
  border: 1px solid #d5dee6;
  padding: 11px 8px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.grid thead th, .grid tbody th {
  font-size: 0.78rem;
  font-weight: 650;
  color: #3e5160;
}
.grid tbody th { text-align: left; background: #f7f9fb; }
.grid td.diag { background: #1e3344; color: #f4f7f9; font-weight: 700; }
.grid td.off { background: #ffffff; }
.states {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
.state {
  background: #ffffff;
  border: 1px solid #d5dee6;
  border-top: 3px solid #8aa0b2;
  border-radius: 8px;
  padding: 8px 10px 10px;
}
.state.normal { border-top-color: #2f6f4e; }
.state.suspect { border-top-color: #b5811a; }
.state.pathologic { border-top-color: #a33b32; }
.state h3 { margin: 0; font-size: 0.95rem; }
.state .big { margin: 2px 0 6px; font-weight: 650; }
.state p { margin: 0; font-size: 0.84rem; line-height: 1.35; }
.state .small { margin-top: 6px; color: #5c6d7a; }
.fileline { margin: 0.2rem 0 1rem; font-size: 0.95rem; line-height: 1.5; }
.empty {
  background: #ffffff;
  border: 1px dashed #c5d0da;
  border-radius: 8px;
  padding: 0.9rem 1rem;
  color: #3e5160;
}
div[class*="st-key-hit_"] { position: relative; margin: 0 0 18px; }
div[class*="st-key-hit_"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
div[class*="st-key-hit_"] .stElementContainer { margin: 0 !important; }
div[class*="st-key-hit_"] div[class*="st-key-card_"] {
  position: absolute !important;
  inset: 0;
  z-index: 2;
  height: auto !important;
}
div[class*="st-key-hit_"] div[class*="st-key-card_"] > div,
div[class*="st-key-hit_"] div[class*="st-key-card_"] button {
  height: 100% !important;
}
div[class*="st-key-hit_"] button {
  width: 100%;
  background: transparent !important;
  border: none !important;
  border-radius: 8px !important;
  color: transparent !important;
  box-shadow: none !important;
  cursor: pointer;
  padding: 0 !important;
}
div[class*="st-key-hit_"] button p { color: transparent !important; }
div[class*="st-key-hit_"] button:hover { background: rgba(23, 33, 43, 0.07) !important; }
div[class*="st-key-hit_"] button:focus-visible {
  outline: 2px solid #1e3344;
  outline-offset: 2px;
}
div[data-testid="stHorizontalBlock"]:has(.model-note) {
  align-items: flex-end !important;
}
div[data-testid="stHorizontalBlock"]:has(.detail-panel) {
  align-items: flex-start !important;
  column-gap: 3rem !important;
}
div[data-testid="stHorizontalBlock"]:has(.detail-panel) > div[data-testid="stColumn"] {
  padding-top: 0 !important;
}
div[data-testid="stColumn"]:has(.detail-panel) {
  border-left: 1px solid #cfd8e0;
  padding-left: 1.7rem !important;
}
div[data-testid="stHorizontalBlock"]:has(.detail-panel) h2.section {
  margin: 0 0 0.45rem !important;
  line-height: 1.25;
}
</style>
"""

FONT_FILES = {
    400: "inter-400.woff2",
    500: "inter-500.woff2",
    600: "inter-600.woff2",
    700: "inter-700.woff2",
}


@st.cache_data
def inter_face_css():
    folder = Path(__file__).resolve().parent / "assets"
    rules = []
    for weight, filename in FONT_FILES.items():
        encoded = base64.b64encode((folder / filename).read_bytes()).decode("ascii")
        rules.append(
            "@font-face{"
            "font-family:'Inter';"
            "font-style:normal;"
            f"font-weight:{weight};"
            "font-display:swap;"
            f"src:url(data:font/woff2;base64,{encoded}) format('woff2');"
            "}"
        )
    return "<style>" + "".join(rules) + "</style>"


def choose_model(name):
    st.session_state.model_pick = name


def model_slug(name):
    return name.lower().replace(" ", "_")

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MODEL_NOTES = {
    "Logistic Regression": "Linear boundary. Class weights keep the two smaller states in the fit.",
    "Decision Tree": "Splits on the raw measurements. Depth stops at 6.",
    "KNN": "Vote of the 11 nearest tracings after scaling.",
    "Naive Bayes": "Reads every measurement as independent of the others.",
    "Support Vector Machine": "RBF kernel with class weights. AUC uses the probability scores.",
}


def show(fragment):
    st.html(fragment)


def format_metric(value):
    if pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


@st.cache_resource
def load_models():
    loaded = {}
    missing = []
    for name, filename in MODEL_FILENAMES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
        else:
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            "Saved model files are missing: "
            + ", ".join(missing)
            + ". Run python train_models.py, then open this page again."
        )
    return loaded


def score_frame(models, table):
    rows = []
    details = {}
    for name, estimator in models.items():
        predicted = estimator.predict(table[FEATURES])
        proba = aligned_probabilities(estimator, table[FEATURES], CLASS_ORDER)
        scores = compute_metrics(table[TARGET], predicted, proba, CLASS_ORDER)
        scores["ML Model Name"] = name
        rows.append(scores)
        details[name] = {
            "predicted": predicted,
            "report": classification_report(
                table[TARGET],
                predicted,
                labels=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            ),
            "matrix": confusion_matrix(table[TARGET], predicted, labels=CLASS_ORDER),
        }
    comparison = pd.DataFrame(rows)[["ML Model Name", *METRIC_NAMES]]
    return comparison, details


def lead_sentence(name, scores, matrix, rank, model_count):
    accuracy = scores["Accuracy"]
    pathologic_index = CLASS_ORDER.index("Pathologic")
    pathologic_total = int(matrix[pathologic_index].sum())
    pathologic_hit = int(matrix[pathologic_index, pathologic_index])
    missed = pathologic_total - pathologic_hit
    opening = (
        f"{name} gets {accuracy:.1%} of these rows right "
        f"and sits {rank} of {model_count} by F1."
    )
    if pathologic_total == 0:
        return opening
    if missed == 0:
        return f"{opening} Every Pathologic tracing in the file is caught."
    if missed == 1:
        return f"{opening} 1 of {pathologic_total} Pathologic tracings is missed."
    return f"{opening} {missed} of {pathologic_total} Pathologic tracings are missed."


def best_values(comparison):
    best = {}
    for name in METRIC_NAMES:
        column = comparison[name]
        best[name] = float(column.max()) if column.notna().any() else None
    return best


def render_rank_board(comparison, chosen):
    ordered = comparison.sort_values(
        ["F1", "MCC"], ascending=False
    ).reset_index(drop=True)
    best = best_values(comparison)
    for place, record in enumerate(ordered.itertuples(index=False), start=1):
        model_name = str(record[0])
        css = "rank is-on" if model_name == chosen else "rank"
        cells = []
        for offset, name in enumerate(METRIC_NAMES, start=1):
            value = record[offset]
            marked = (
                best[name] is not None
                and not pd.isna(value)
                and abs(float(value) - best[name]) < 1e-12
            )
            mark = " best" if marked else ""
            cells.append(
                f"<span class='m{mark}'><i>{html.escape(name)}</i>{format_metric(value)}</span>"
            )
        with st.container(key=f"hit_{model_slug(model_name)}"):
            show(
                f"<article class='{css}'>"
                f"<div class='rank-no'>{place}</div>"
                f"<div><h3>{html.escape(model_name)}</h3>"
                f"<div class='ms'>{''.join(cells)}</div></div></article>"
            )
            st.button(
                model_name,
                key=f"card_{model_slug(model_name)}",
                on_click=choose_model,
                args=(model_name,),
                use_container_width=True,
            )


def count_grid(matrix):
    header = "".join(f"<th>{html.escape(label)}</th>" for label in CLASS_ORDER)
    body = []
    for row, actual in enumerate(CLASS_ORDER):
        cells = [f"<th>{html.escape(actual)}</th>"]
        for column in range(len(CLASS_ORDER)):
            kind = "diag" if row == column else "off"
            cells.append(f"<td class='{kind}'>{int(matrix[row, column])}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table class='grid'><thead><tr><th></th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
        + "<p class='note'>Rows are the true state. Columns are the label this model assigned.</p>"
    )


def state_cards(report, matrix):
    cards = []
    for row, label in enumerate(CLASS_ORDER):
        stats = report[label]
        moved = []
        for column, predicted in enumerate(CLASS_ORDER):
            count = int(matrix[row, column])
            if row != column and count:
                moved.append(f"{count} called {predicted}")
        detail = ", ".join(moved) if moved else "No row of this state was sent elsewhere."
        cards.append(
            f"<div class='state {label.lower()}'>"
            f"<h3>{html.escape(label)}</h3>"
            f"<p class='big'>{int(stats['support'])} rows</p>"
            f"<p>Precision {stats['precision']:.4f}</p>"
            f"<p>Recall {stats['recall']:.4f}</p>"
            f"<p>F1 {stats['f1-score']:.4f}</p>"
            f"<p class='small'>{html.escape(detail)}</p>"
            "</div>"
        )
    return "<div class='states'>" + "".join(cards) + "</div>"


def misread_rows(table, predicted):
    view = table[FEATURES + [TARGET]].copy()
    view.insert(0, "Predicted", predicted)
    view.insert(0, "Actual", table[TARGET].to_numpy())
    mismatch = view["Actual"].to_numpy() != view["Predicted"].to_numpy()
    wrong = view.iloc[mismatch].copy()
    return wrong[
        ["Actual", "Predicted", "LB", "ASTV", "ALTV", "DP", "Mean"]
    ].rename(
        columns={
            "LB": "Baseline",
            "ASTV": "Abn. short-term %",
            "ALTV": "Abn. long-term %",
            "DP": "Prolonged decel.",
            "Mean": "Histogram mean",
        }
    )


def main():
    st.set_page_config(page_title="Fetal state classifier", layout="wide")
    show(inter_face_css())
    show(PAGE_CSS)

    names = list(MODEL_FILENAMES)
    if "model_pick" not in st.session_state:
        st.session_state.model_pick = names[0]

    try:
        models = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    header = st.container()
    body = st.container()
    footer = st.container()

    with footer:
        show("<h2 class='section'>Test CSV</h2>")
        with st.container(border=True):
            file_col, option_col = st.columns([1.15, 1])
            with file_col:
                upload = st.file_uploader("Test CSV", type=["csv"])
            with option_col:
                use_included = st.checkbox(
                    "Score test_data.csv if no file is uploaded",
                    value=True,
                )
                if TEST_PATH.exists():
                    st.download_button(
                        "Get test_data.csv",
                        data=TEST_PATH.read_bytes(),
                        file_name="test_data.csv",
                        mime="text/csv",
                    )
                with st.expander("Columns required"):
                    st.caption(
                        "FetalState, plus LB, AC, FM, UC, DL, DS, DP, ASTV, MSTV, "
                        "ALTV, MLTV, Width, Min, Max, Nmax, Nzeros, Mode, Mean, "
                        "Median, Variance, Tendency. Labels are Normal, Suspect, Pathologic."
                    )

    with header:
        show('<h1 class="main-title">Fetal state classifier</h1>')
        pick_col, note_col = st.columns([0.8, 1.8], gap="large")
        with pick_col:
            chosen = st.selectbox("Model", names, key="model_pick")
        with note_col:
            show(f'<p class="model-note">{html.escape(MODEL_NOTES[chosen])}</p>')

    if upload is not None:
        source_name = upload.name
        raw = pd.read_csv(upload)
    elif use_included and TEST_PATH.exists():
        source_name = TEST_PATH.name
        raw = pd.read_csv(TEST_PATH)
    else:
        raw = None

    with body:
        if raw is None:
            show(
                "<div class='empty'>Add a test CSV at the bottom of the page, "
                "or leave the included split switched on.</div>"
            )
            return

        table, error = prepare_table(raw)
        if error:
            st.error(error)
            return

        comparison, details = score_frame(models, table)
        selected = details[chosen]
        selected_scores = comparison.loc[comparison["ML Model Name"] == chosen].iloc[0]
        f1_order = comparison.sort_values("F1", ascending=False)["ML Model Name"].tolist()
        rank = f1_order.index(chosen) + 1
        counts = table[TARGET].value_counts()
        mix = ", ".join(
            f"{int(counts.get(label, 0))} {label}" for label in CLASS_ORDER
        )

        bench, detail = st.columns([0.92, 1.35], gap="large")
        with bench:
            show("<h2 class='section'>All five models</h2>")
            show(
                "<p class='note'>Ordered by F1. Click a card to open that model. "
                "A marked figure is the best of the five on this file. "
                "Precision, recall and F1 are macro. AUC is one-vs-rest.</p>"
            )
            render_rank_board(comparison, chosen)
        with detail:
            show(f"<h2 class='section detail-panel'>{html.escape(chosen)}</h2>")
            show(
                "<p class='fileline'>"
                + html.escape(f"{len(table)} rows from {source_name}. {mix}.")
                + "</p>"
            )
            show(
                "<p class='note'>"
                + html.escape(
                    lead_sentence(
                        chosen, selected_scores, selected["matrix"], rank, len(models)
                    )
                )
                + "</p>"
            )
            show("<h2 class='section'>Confusion matrix</h2>")
            show(count_grid(selected["matrix"]))
            show("<h2 class='section'>Classification report</h2>")
            show(state_cards(selected["report"], selected["matrix"]))

            wrong = misread_rows(table, selected["predicted"])
            show("<h2 class='section' style='margin-top:0.9rem'>Misread tracings</h2>")
            if wrong.empty:
                show("<p class='note'>Actual and predicted labels match on every row.</p>")
            else:
                preview = wrong.head(6)
                show(
                    f"<p class='note'>Showing {len(preview)} of {len(wrong)}. "
                    "Baseline is heart rate in bpm.</p>"
                )
                st.dataframe(preview, use_container_width=True, hide_index=True)

            scored = table.copy()
            scored["PredictedFetalState"] = selected["predicted"]
            st.download_button(
                "Download predictions",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
