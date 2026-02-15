# --- Section 1: Imports and Setup ---
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
import os
import warnings
import logging
import random
import math
from collections import defaultdict
import glob

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- Section 2: Helpers & Model Logic ---

# --- Per-category PP caps (auto from data; no hard-coded values) ---
def _load_category_caps(use_quantile: bool = False, q_upper: float = 0.99) -> dict:
    """
    Load per-category caps in this order:
      1) category_pp_caps.pkl (persisted during training)
      2) Compute from training data (max or quantile) and cache to pkl
      3) Return {} if nothing available (no capping)
    """
    # 1) Try precomputed caps
    try:
        caps = joblib.load("category_pp_caps.pkl")
        if isinstance(caps, dict) and caps:
            return caps
    except Exception:
        pass

    # 2) Derive from training file if available
    candidates = []
    env_tf = os.getenv("CLARIO_TRAIN_FILE")
    if env_tf:
        candidates.append(env_tf)
    candidates += glob.glob("./*train*.xlsx") + glob.glob("./*train*.csv")
    candidates += glob.glob("./*dataset*.xlsx") + glob.glob("./*dataset*.csv")
    candidates += glob.glob("./data/*train*.xlsx") + glob.glob("./data/*train*.csv")
    candidates += glob.glob("./data/*dataset*.xlsx") + glob.glob("./data/*dataset*.csv")

    for path in candidates:
        try:
            if not os.path.exists(path):
                continue
            if path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            elif path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                continue

            # normalize cols
            low = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
            cat_col = next((low[k] for k in ("category", "cat", "item_category") if k in low), None)
            tgt_col = next((low[k] for k in ("client_pp", "ideal_pp", "pp", "client_per_pax", "per_pax") if k in low), None)
            if not cat_col or not tgt_col:
                continue

            tmp = df[[cat_col, tgt_col]].copy()
            tmp.columns = ["category", "target"]
            tmp["category"] = tmp["category"].map(canon_category)
            grp = tmp.groupby("category")["target"]
            caps = (grp.quantile(q_upper) if use_quantile else grp.max()).to_dict()
            joblib.dump(caps, "category_pp_caps.pkl")
            return caps
        except Exception:
            continue

    return {}


def enforce_pp_cap(per_pax: float, category: str) -> float:
    """Clamp per-pax to the learned cap for this category (if available)."""
    caps = _load_category_caps()
    cap = caps.get(canon_category(category))
    return min(per_pax, cap) if cap is not None else per_pax


# --- Shared-safe helpers (client-agnostic, fine to reuse) ---

def norm(s):
    return s.strip().lower() if isinstance(s, str) else s


def safe_transform(le, value, default_idx=0):
    """Return encoder index if present; otherwise a safe default (or 'other' if exists)."""
    if value in getattr(le, "classes_", []):
        return le.transform([value])[0]
    if "other" in getattr(le, "classes_", []):
        return le.transform(["other"])[0]
    return default_idx


def case_insensitive_item_lookup(name, known_items):
    """Return the exact-cased known item if lowercased matches; else original."""
    lower_to_exact = {k.lower(): k for k in known_items}
    return lower_to_exact.get(name.lower(), name)


def same_cat(a, b):
    return (a is not None) and (b is not None) and (norm(a) == norm(b))


def canon_category(s):
    # Canonical labels
    CANON = {
        "indian bread": "indian bread", "indian breads": "indian bread",
        "veg dry": "veg dry",
        "veg curry": "veg gravy", "gravy veg": "veg gravy", "veg gravy": "veg gravy",
        "healty rice": "healty rice",
        "flavour rice": "flavoured rice", "flavoured rice": "flavoured rice",
        "white rice": "white rice", "steamed rice": "white rice",
        "dal": "dal",
        "sambar": "sambar/rasam", "sambar/rasam": "sambar/rasam",
        "rasam": "sambar/rasam",
    }
    return CANON.get(norm(s), norm(s))


def canon_mealday(s):
    M = {
        "veg": "veg",
        "nonveg": "nonveg", "non-veg": "nonveg", "non veg": "nonveg",
    }
    return M.get(norm(s), norm(s))


def get_fallback_item_from_subcat(sub_category, item_le, item_to_subcat):
    s_sub = norm(sub_category)
    for item, subcat in item_to_subcat.items():
        if norm(subcat) == s_sub and item in item_le.classes_:
            return item
    return None


def build_category_to_subcats(item_to_cat, item_to_subcat):
    cat_to_subs = defaultdict(set)
    for item, cat in item_to_cat.items():
        sub = item_to_subcat.get(item)
        if cat and sub:
            cat_to_subs[norm(cat)].add(norm(sub))
    return {k: sorted(v) for k, v in cat_to_subs.items()}


def custom_round(value):
    return math.floor(value / 0.005) * 0.005


def bumped_ceil_og_prop(per_pax, is_nonveg_biryani=False):
    bump = 0.035 if is_nonveg_biryani else 0.01
    return math.ceil((per_pax + bump) * 100) / 100.0


def aggressive_rounding(value):
    """Round to the nearest 0.005 for aggressive portion planning."""
    return round(value * 200) / 200.0


def mg_rounder(x):
    """Round to the nearest multiple of 5 for MG values."""
    return int(round(x / 5.0)) * 5


def clean_format(df, float_cols):
    for col in float_cols:
        df[col] = df[col].map(lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if isinstance(x, float) else x)
    return df


@st.cache_data
def get_nonveg_categories_from_data():
    df = pd.read_excel(
        r'C:\Users\Darshan.Pawar\Downloads\embedded code\Clario\Compass Clario\Clario_Wastage_dataset.xlsx'
    )
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for c in ["meal_type", "category"]:
        if c in df.columns:
            if c == "category":
                df[c] = df[c].map(canon_category)
            else:
                df[c] = df[c].map(norm)
    df = df[df.get("meal_type", "") == "non veg"]
    return sorted(df["category"].dropna().unique())


def train_model():
    st.info("Training per-pax quantity model…")
    df = pd.read_excel(
        r'C:\Users\Darshan.Pawar\Downloads\embedded code\Clario\Compass Clario\Clario_Wastage_dataset.xlsx'
    )

    # 1) Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2) Normalize key columns (no day_type / holiday_type)
    for c in ["menu_items", "sub_category", "category", "meal_day", "meal_type"]:
        if c in df.columns:
            if c == "category":
                df[c] = df[c].map(canon_category)
            elif c == "meal_day":
                df[c] = df[c].map(canon_mealday)
            else:
                df[c] = df[c].map(norm)

    # 3) Weekday normalized
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name().map(norm)

    # 4) Context list
    df["menu_items_list"] = df.groupby("date")["menu_items"].transform(list)

    # 5) Save mappings
    item_to_subcat = df.set_index("menu_items")["sub_category"].to_dict()
    item_to_cat = df.set_index("menu_items")["category"].to_dict()
    joblib.dump(item_to_subcat, "item_to_subcat.pkl")
    joblib.dump(item_to_cat, "item_to_cat.pkl")
    joblib.dump(build_category_to_subcats(item_to_cat, item_to_subcat), "cat_to_subs.pkl")

    # 6) Label encoders (no day_type / holiday_type)
    le_map = {}
    for col in ["weekday", "menu_items", "sub_category", "category", "meal_day"]:
        le = LabelEncoder()
        df[f"{col}_idx"] = le.fit_transform(df[col])
        joblib.dump(le, f"{col}_encoder.pkl")
        le_map[col] = le

    # 7) Context sequence indices
    item_to_idx = {item: idx for idx, item in enumerate(le_map["menu_items"].classes_)}
    pad_idx = len(item_to_idx)
    df["context_seq"] = df["menu_items_list"].map(
        lambda ml: [item_to_idx[i] for i in ml if i in item_to_idx][:10]
        + [pad_idx] * (10 - len([i for i in ml if i in item_to_idx][:10]))
    )

    # 8) Features & target (no day_type / holiday_type)
    X = [
        df["menu_items_idx"],
        np.array(df["context_seq"].tolist()),
        df["sub_category_idx"],
        df["category_idx"],
        df["weekday_idx"],
        df["meal_day_idx"],
    ]
    y = df["ideal_pp"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(list(zip(*X)), y, test_size=0.2, random_state=42)
    X_train = list(map(lambda arr: np.array(arr), zip(*X_train)))
    X_test = list(map(lambda arr: np.array(arr), zip(*X_test)))

    # 9) Model (no day_type / holiday_type inputs)
    inputs = [
        tf.keras.Input(shape=(1,), name="item_input"),
        tf.keras.Input(shape=(10,), name="context_input"),
        tf.keras.Input(shape=(1,), name="sub_input"),
        tf.keras.Input(shape=(1,), name="category_input"),
        tf.keras.Input(shape=(1,), name="weekday_input"),
        tf.keras.Input(shape=(1,), name="meal_day_input"),
    ]

    emb = lambda size, dim: tf.keras.layers.Embedding(size + 1, dim)
    item_e = emb(len(le_map["menu_items"].classes_), 8)(inputs[0])
    context_e = emb(len(le_map["menu_items"].classes_), 8)(inputs[1])
    sub_e = emb(len(le_map["sub_category"].classes_), 4)(inputs[2])
    cat_e = emb(len(le_map["category"].classes_), 4)(inputs[3])
    wd_e = emb(len(le_map["weekday"].classes_), 2)(inputs[4])
    meal_e = emb(len(le_map["meal_day"].classes_), 2)(inputs[5])

    avg_ctx = tf.keras.layers.GlobalAveragePooling1D()(context_e)
    concat = tf.keras.layers.Concatenate()(
        [
            tf.keras.layers.Flatten()(item_e),
            avg_ctx,
            tf.keras.layers.Flatten()(sub_e),
            tf.keras.layers.Flatten()(cat_e),
            tf.keras.layers.Flatten()(wd_e),
            tf.keras.layers.Flatten()(meal_e),
        ]
    )
    x = tf.keras.layers.Dense(64, activation="relu")(concat)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

    preds = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    st.success(f"✅ Model trained. RMSE: {rmse:.4f} per pax")

    model.save("per_pax_tf_model.keras")

    # mark this model as v2 (no day_type/holiday_type features)
    with open("model_v2_no_day_holiday.txt", "w") as f:
        f.write("v2_no_day_holiday")


# --- Section 3: Prediction Function ---

def predict_quantity(target_item, menu_items, item_mg, sub_category, category, weekday, meal_day):
    model = tf.keras.models.load_model("per_pax_tf_model.keras")
    item_le = joblib.load("menu_items_encoder.pkl")
    sub_le = joblib.load("sub_category_encoder.pkl")
    cat_le = joblib.load("category_encoder.pkl")
    weekday_le = joblib.load("weekday_encoder.pkl")
    meal_day_le = joblib.load("meal_day_encoder.pkl")
    item_to_subcat = joblib.load("item_to_subcat.pkl")

    # Canonicalize/normalize all string inputs
    t_item = (target_item or "").strip()
    m_items = [(i or "").strip() for i in menu_items]
    s_sub = norm(sub_category)
    s_cat = canon_category(category)
    s_week = norm(weekday)

    known_items = set(item_le.classes_)
    pad_idx = len(item_le.classes_)

    # Case-insensitive exact resolve
    t_item = case_insensitive_item_lookup(t_item, known_items)

    # Fallback if unseen
    if t_item not in known_items:
        fallback_item = get_fallback_item_from_subcat(s_sub, item_le, item_to_subcat)
        if fallback_item:
            st.warning(f"⚠️ '{target_item}' not seen during training. Using fallback item '{fallback_item}'.")
            item_idx = item_le.transform([fallback_item])[0]
        else:
            st.error(f"❌ No fallback item found for sub-category '{sub_category}'. Skipping.")
            return None, None
    else:
        item_idx = item_le.transform([t_item])[0]

    # Context indices (case-insensitive)
    context_idxs = []
    for i in m_items:
        if i.lower() == t_item.lower():
            continue
        i_exact = case_insensitive_item_lookup(i, known_items)
        if i_exact in known_items:
            context_idxs.append(item_le.transform([i_exact])[0])
    padded_context = context_idxs[:10] + [pad_idx] * (10 - len(context_idxs))

    # Safe transforms for remaining categorical inputs
    sub_idx = safe_transform(sub_le, s_sub, default_idx=0)
    cat_idx = safe_transform(cat_le, s_cat, default_idx=0)
    weekday_idx = safe_transform(weekday_le, s_week, default_idx=0)
    meal_idx = safe_transform(meal_day_le, canon_mealday(meal_day), default_idx=0)

    per_pax_qty = model.predict(
        [
            np.array([item_idx]),
            np.array([padded_context]),
            np.array([sub_idx]),
            np.array([cat_idx]),
            np.array([weekday_idx]),
            np.array([meal_idx]),
        ],
        verbose=0,
    )[0][0]

    # Enforce per-category cap before rounding & totals
    per_pax_qty = enforce_pp_cap(per_pax_qty, s_cat)

    per_pax_qty_rounded = custom_round(per_pax_qty)
    total_qty = per_pax_qty_rounded * item_mg
    return per_pax_qty_rounded, total_qty


# --- Section 4: Streamlit UI Inputs ---

st.title("Per Pax Quantity Prediction")

# Check model files (include a version tag to force retrain once)
required_files = [
    "per_pax_tf_model.keras",
    "menu_items_encoder.pkl",
    "sub_category_encoder.pkl",
    "category_encoder.pkl",
    "weekday_encoder.pkl",
    "meal_day_encoder.pkl",
    "item_to_subcat.pkl",
    "item_to_cat.pkl",
    "cat_to_subs.pkl",
    "model_v2_no_day_holiday.txt",
]

if not all(os.path.exists(f) for f in required_files):
    st.warning("Some model files are missing or outdated. Training model now...")
    train_model()

# Load mappings
item_to_subcat = joblib.load("item_to_subcat.pkl")
item_to_cat = joblib.load("item_to_cat.pkl")
cat_to_subs = joblib.load("cat_to_subs.pkl")

# Case-insensitive lookup helper map
item_to_subcat_lc = {norm(k): v for k, v in item_to_subcat.items()}

# Date input
selected_date = st.date_input("Select today's date:")

# 🔥 Non-veg toggle right after date
is_nonveg_day = st.toggle("🍗 Is it a Non-Veg Day?", value=True)

day_name = selected_date.strftime("%A")
day_name_norm = norm(day_name)

month_name = selected_date.strftime("%B")

item_entries = []
menu_items = []

meal_day_type = "nonveg" if is_nonveg_day else "veg"

nonveg_item = None
nonveg_client_mg = 0
nonveg_category = None

if is_nonveg_day:
    # Non-Veg category selection
    nonveg_options = get_nonveg_categories_from_data()
    nonveg_category = st.selectbox("Select Non-Veg Category", options=nonveg_options)

    # Non-Veg Item
    st.subheader("Enter One Non-Veg Item")
    nonveg_item = st.text_input(f"Non-Veg Item Name for {nonveg_category}:", key="nonveg_item")

    if nonveg_item:
        key = norm(nonveg_item)
        if key in item_to_subcat_lc:
            subcat = item_to_subcat_lc[key]
            st.text(f"✅ Sub-category: {subcat}")
        else:
            subcat_options = cat_to_subs.get(canon_category(nonveg_category), [])
            subcat = st.selectbox(
                f"Select sub-category for '{nonveg_item}':", options=subcat_options, key="subcat_nonveg"
            )

        # Non veg Client MG
        nonveg_client_mg = st.number_input(
            f"Client MG for Non-Veg Item '{nonveg_item}':", min_value=1, step=1, value=300, key="nonveg_mg"
        )

        item_entries.append((nonveg_item, subcat, canon_category(nonveg_category), nonveg_client_mg))
        menu_items.append(nonveg_item)

# Client MG shared across veg items
client_mg = st.number_input("Enter shared Client MG for all items:", min_value=1, step=1, value=300)

# Menu items
st.subheader("Enter Menu Items by Category")

fixed_categories = [
    "Flavoured Rice",
    "Indian Bread",
    "White Rice",
    "Healty Rice",
    "Veg Dry",
    "Veg Gravy",
    "Dal",
    "Sambar/Rasam",
]

star_items = {"flavoured rice", "veg gravy"}
for cat in fixed_categories:
    label = f"Item name for {cat}:"
    if cat.strip().lower() in star_items:
        label = f"⭐ {label}"
    item = st.text_input(label, key=f"item_{cat}")
    if not item:
        continue
    menu_items.append(item)

    key = norm(item)
    canon_cat_key = canon_category(cat)
    if key in item_to_subcat_lc:
        subcat = item_to_subcat_lc[key]
        st.text(f"✅ Sub-category: {subcat}")
    else:
        subcat_options = cat_to_subs.get(canon_cat_key, [])
        subcat = st.selectbox(
            f"Select sub-category for '{item}' in {cat}:", options=subcat_options, key=f"subcat_{cat}"
        )

    item_entries.append((item, subcat, canon_cat_key, client_mg))

# --- Section 5: Prediction Output + Production Plans ---

if st.button("Predict"):
    st.markdown("### Prediction Results")
    st.markdown(
        f"**Date:** {selected_date} | **Day:** {day_name} | **Month:** {month_name}  \n"
        f"**Non-Veg Category:** {nonveg_category}"
    )
    st.markdown("---")

    if not item_entries:
        st.warning("Please add at least one menu item.")
    else:
        results = []
        vendor_mgs = []

        for item, subcat, cat, item_mg in item_entries:
            per_pax, total_qty = predict_quantity(
                target_item=item,
                menu_items=menu_items,
                item_mg=item_mg,
                sub_category=subcat,
                category=cat,
                weekday=day_name_norm,
                meal_day=meal_day_type,
            )

            if per_pax is None:
                continue

            is_nv_biryani = (
                is_nonveg_day
                and nonveg_category is not None
                and cat.strip().lower() == nonveg_category.strip().lower()
                and nonveg_category.strip().lower() == "non veg biryani"
            )

            og_prop = bumped_ceil_og_prop(per_pax, is_nonveg_biryani=is_nv_biryani)

            eps = 1e-9
            vendor_mg = total_qty / max(og_prop, eps)
            vendor_mgs.append(vendor_mg)

            results.append(
                {
                    "Category": cat,
                    "Item": item,
                    "Client PP": per_pax,
                    "Total Qty": total_qty,
                }
            )

        if results:
            df_res = pd.DataFrame(results)

            # Round Total Qty to 1 decimal
            df_res["Total Qty"] = df_res["Total Qty"].round(1)

            client_plan = df_res[["Category", "Item", "Client PP", "Total Qty"]]

            cleaned_client_plan = clean_format(client_plan.copy(), ["Client PP", "Total Qty"])
            st.markdown("#### 📋 Production Plan")
            if is_nonveg_day and nonveg_item:
                total_client_mg = client_mg
                veg_client_mg = max(total_client_mg - nonveg_client_mg, 0)
                st.write(f"**Veg Client MG: {veg_client_mg} | Non-Veg Client MG: {nonveg_client_mg}**")
            else:
                st.write(f"**Client MG: {client_mg}**")

            st.table(cleaned_client_plan)
            st.markdown("---")
