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

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- Section 2: Helpers & Model Logic ---

def custom_round(value: float) -> float:
    """Round down to nearest 0.005."""
    return math.floor(value / 0.005) * 0.005

def clean_format(df: pd.DataFrame, float_cols) -> pd.DataFrame:
    """Format float columns nicely without trailing zeros."""
    for col in float_cols:
        df[col] = df[col].map(
            lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if isinstance(x, float) else x
        )
    return df

def get_fallback_item_from_subcat(sub_category, item_le, item_to_subcat):
    """If a new item isn't in the encoder, fall back to another item from same sub-category."""
    for item, subcat in item_to_subcat.items():
        if subcat == sub_category and item in item_le.classes_:
            return item
    return None

def build_category_to_subcats(item_to_cat, item_to_subcat):
    """Build category -> sorted list of sub-categories from training data."""
    cat_to_subs = defaultdict(set)
    for item in item_to_cat:
        cat = item_to_cat[item]
        sub = item_to_subcat.get(item)
        if cat and sub:
            cat_to_subs[cat].add(sub)
    return {k: sorted(v) for k, v in cat_to_subs.items()}

def get_subcat_options_for_category(cat_name: str, cat_to_subs: dict) -> list:
    """
    Get sub-category options for a UI category name.
    Handles:
      - exact match
      - case-insensitive match
      - combined categories like 'Dal/Sambar'
    """
    # 1) Direct match
    direct = cat_to_subs.get(cat_name, [])
    if direct:
        return direct

    # 2) Combined categories like 'Dal/Sambar'
    parts = [
        p.strip()
        for p in cat_name.replace("\\", "/").split("/")
        if p.strip()
    ]
    if len(parts) > 1:
        merged = set()
        for p in parts:
            merged.update(cat_to_subs.get(p, []))
            for k, subs in cat_to_subs.items():
                if k.lower() == p.lower():
                    merged.update(subs)
        if merged:
            return sorted(merged)

    # 3) Case-insensitive fallback for whole name
    merged = set()
    for k, subs in cat_to_subs.items():
        if k.lower() == cat_name.lower():
            merged.update(subs)
    return sorted(merged)


# --- Section 2b: Training (no meal_day, no meal_type) ---

def train_model():
    st.info("Training per-pax quantity model…")
    df = pd.read_excel(
        r"C:\Users\Darshan.Pawar\Downloads\embedded code\Tessolve\Compass Tessolve\Tesslove_Wastage_Dataset.xlsx"
    )
    # Only clean column NAMES, not values
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Normalize key columns (no day_type / holiday_type / meal_day / meal_type)
    df["sub_category"] = df["sub_category"].str.strip().str.lower()

    # Context menu: list of items per date
    df["menu_items_list"] = df.groupby("date")["menu_items"].transform(list)

    # Weekday
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name()  # e.g. "Monday"

    # Save mappings
    item_to_subcat = df.set_index("menu_items")["sub_category"].to_dict()
    item_to_cat = df.set_index("menu_items")["category"].to_dict()
    joblib.dump(item_to_subcat, "item_to_subcat.pkl")
    joblib.dump(item_to_cat, "item_to_cat.pkl")
    joblib.dump(build_category_to_subcats(item_to_cat, item_to_subcat), "cat_to_subs.pkl")

    # Label encoders (weekday, menu_items, sub_category, category)
    le_map = {}
    for col in ["weekday", "menu_items", "sub_category", "category"]:
        le = LabelEncoder()
        df[f"{col}_idx"] = le.fit_transform(df[col])
        joblib.dump(le, f"{col}_encoder.pkl")
        le_map[col] = le

    # Context indices (sequence of other items on the same day)
    item_to_idx = {item: idx for idx, item in enumerate(le_map["menu_items"].classes_)}
    pad_idx = len(item_to_idx)

    def encode_context(ml):
        valid = [item_to_idx[i] for i in ml if i in item_to_idx]
        valid = valid[:10]
        return valid + [pad_idx] * (10 - len(valid))

    df["context_seq"] = df["menu_items_list"].map(encode_context)

    # Features & target
    X = [
        df["menu_items_idx"],
        np.array(df["context_seq"].tolist()),
        df["sub_category_idx"],
        df["category_idx"],
        df["weekday_idx"],
    ]
    y = df["ideal_pp"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        list(zip(*X)), y, test_size=0.2, random_state=42
    )
    X_train = list(map(lambda arr: np.array(arr), zip(*X_train)))
    X_test = list(map(lambda arr: np.array(arr), zip(*X_test)))

    # Model: 5 inputs (item, context, sub, category, weekday)
    inputs = [
        tf.keras.Input(shape=(1,), name="item_input"),
        tf.keras.Input(shape=(10,), name="context_input"),
        tf.keras.Input(shape=(1,), name="sub_input"),
        tf.keras.Input(shape=(1,), name="category_input"),
        tf.keras.Input(shape=(1,), name="weekday_input"),
    ]

    emb = lambda size, dim: tf.keras.layers.Embedding(size + 1, dim)

    item_e = emb(len(le_map["menu_items"].classes_), 8)(inputs[0])
    context_e = emb(len(le_map["menu_items"].classes_), 8)(inputs[1])
    sub_e = emb(len(le_map["sub_category"].classes_), 4)(inputs[2])
    cat_e = emb(len(le_map["category"].classes_), 4)(inputs[3])
    wd_e = emb(len(le_map["weekday"].classes_), 2)(inputs[4])

    avg_ctx = tf.keras.layers.GlobalAveragePooling1D()(context_e)
    concat = tf.keras.layers.Concatenate()(
        [
            tf.keras.layers.Flatten()(item_e),
            avg_ctx,
            tf.keras.layers.Flatten()(sub_e),
            tf.keras.layers.Flatten()(cat_e),
            tf.keras.layers.Flatten()(wd_e),
        ]
    )

    x = tf.keras.layers.Dense(64, activation="relu")(concat)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
    )

    preds = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    st.success(f"✅ Model trained. RMSE: {rmse:.4f} per pax")

    # Save new-arch model with v3 name
    model.save("per_pax_tf_model_v3.keras")


# --- Section 3: Prediction Function (no meal_day, no meal_type) ---

def predict_quantity(
    target_item,
    menu_items,
    item_mg,
    sub_category,
    category,
    weekday,
):
    """Predict per-pax and total qty for a given item."""
    model = tf.keras.models.load_model("per_pax_tf_model_v3.keras")
    item_le = joblib.load("menu_items_encoder.pkl")
    sub_le = joblib.load("sub_category_encoder.pkl")
    cat_le = joblib.load("category_encoder.pkl")
    weekday_le = joblib.load("weekday_encoder.pkl")
    item_to_subcat = joblib.load("item_to_subcat.pkl")

    known_items = set(item_le.classes_)
    pad_idx = len(item_le.classes_)

    # Handle unseen items by falling back to another item from same sub-category
    if target_item not in known_items:
        fallback_item = get_fallback_item_from_subcat(
            sub_category, item_le, item_to_subcat
        )
        if fallback_item:
            item_idx = item_le.transform([fallback_item])[0]
        else:
            # Completely unknown pattern; better to skip than give junk
            return None, None
    else:
        item_idx = item_le.transform([target_item])[0]

    # Context: other items on the same menu
    context_idxs = [
        item_le.transform([i])[0]
        for i in menu_items
        if i != target_item and i in known_items
    ]
    context_idxs = context_idxs[:10]
    padded_context = context_idxs + [pad_idx] * (10 - len(context_idxs))

    sub_idx = (
        sub_le.transform([sub_category])[0]
        if sub_category in sub_le.classes_
        else 0
    )
    cat_idx = (
        cat_le.transform([category])[0]
        if category in cat_le.classes_
        else 0
    )
    weekday_idx = weekday_le.transform([weekday])[0]

    per_pax_qty = model.predict(
        [
            np.array([item_idx]),
            np.array([padded_context]),
            np.array([sub_idx]),
            np.array([cat_idx]),
            np.array([weekday_idx]),
        ],
        verbose=0,
    )[0][0]

    per_pax_qty_rounded = custom_round(per_pax_qty)
    total_qty = per_pax_qty_rounded * item_mg
    return per_pax_qty_rounded, total_qty


# --- Section 4: Streamlit UI (Veg only, new categories) ---

st.title("Production Plan Prediction (Veg Only)")

# Check if model & encoders exist
required_files = [
    "per_pax_tf_model_v3.keras",
    "menu_items_encoder.pkl",
    "sub_category_encoder.pkl",
    "category_encoder.pkl",
    "weekday_encoder.pkl",
    "item_to_subcat.pkl",
    "item_to_cat.pkl",
    "cat_to_subs.pkl",
]

if not all(os.path.exists(f) for f in required_files):
    st.warning("Some model files are missing or outdated. Training model now...")
    train_model()

# Load mappings
item_to_subcat = joblib.load("item_to_subcat.pkl")
item_to_cat = joblib.load("item_to_cat.pkl")
cat_to_subs = joblib.load("cat_to_subs.pkl")

# Date input
selected_date = st.date_input("Select today's date:")
day_name = selected_date.strftime("%A")
month_name = selected_date.strftime("%B")

# Shared Client MG for all items
client_mg = st.number_input(
    "Enter shared Client MG for all items:",
    min_value=1,
    step=1,
    value=300,
)

item_entries = []
menu_items = []

st.subheader("Enter Menu Items by Category")

# New category list (Veg only)
fixed_categories = [
    "Healthy rice",
    "Indian Bread",
    "Flavoured Rice",
    "Veg gravy",
    "Veg Dry",
    "Dal/Sambar",
    "white rice",
    "Rasam",
]

# Highlight important categories if you want
star_items = {"flavoured rice", "veg gravy"}

for cat in fixed_categories:
    label = f"Item name for {cat}:"
    if cat.strip().lower() in star_items:
        label = f"⭐ {label}"
    item = st.text_input(label, key=f"item_{cat}")

    if not item:
        continue

    menu_items.append(item)

    # Sub-category from historical mapping if known, else UI selection/free text
    if item in item_to_subcat:
        subcat = item_to_subcat[item]
        st.text(f"✅ Sub-category: {subcat}")
    else:
        subcat_options = get_subcat_options_for_category(cat, cat_to_subs)
        if subcat_options:
            subcat = st.selectbox(
                f"Select sub-category for '{item}' in {cat}:",
                options=subcat_options,
                key=f"subcat_{cat}",
            )
        else:
            # Fallback: free text if no mapping exists for this category
            subcat = st.text_input(
                f"Enter sub-category for '{item}' in {cat}:",
                key=f"subcat_{cat}_free",
            )

    item_entries.append((item, subcat, cat, client_mg))


# --- Section 5: Prediction Output (single Client MG) ---

if st.button("Predict"):
    st.markdown("### Prediction Results")
    st.markdown(
        f"**Date:** {selected_date} | **Day:** {day_name} | **Month:** {month_name}"
    )
    st.markdown("---")

    if not item_entries:
        st.warning("Please add at least one menu item.")
    else:
        results = []

        for item, subcat, cat, item_mg in item_entries:
            per_pax, total_qty = predict_quantity(
                target_item=item,
                menu_items=menu_items,
                item_mg=item_mg,
                sub_category=subcat,
                category=cat,
                weekday=day_name,
            )

            if per_pax is None:
                continue

            results.append(
                {
                    "Category": cat,
                    "Item": item,
                    "Per Pax Qty": per_pax,
                    "Total Qty": total_qty,
                }
            )

        if results:
            df_res = pd.DataFrame(results)

            # Round Total Qty nicely
            df_res["Total Qty"] = df_res["Total Qty"].round(1)

            # Final production plan table for client
            client_plan = df_res[["Category", "Item", "Per Pax Qty", "Total Qty"]]
            cleaned_client_plan = clean_format(
                client_plan.copy(), ["Per Pax Qty", "Total Qty"]
            )

            st.markdown("#### 📋 Production Plan")
            st.write(f"**Client MG (shared): {client_mg}**")
            st.table(cleaned_client_plan)
            st.markdown("---")
        else:
            st.warning("Model could not generate predictions for the given items.")
