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
    # Tekion-friendly canonical labels
    CANON = {
        "indian bread": "indian bread", "indian breads": "indian bread",
        "veg dry": "veg dry",
        "veg curry": "veg curry", "gravy veg": "veg curry",
        "flavour rice": "flavoured rice", "flavoured rice": "flavoured rice",
        "white rice": "white rice", "steamed rice": "white rice",
        "dal": "dal", "sambar": "sambar", "rasam": "rasam", "salad": "salad",
    }
    return CANON.get(norm(s), norm(s))

def canon_daytype(s):
    CANON = {
        "regular": "regular",
        "previous day of holiday": "previous day of holiday",
        "next day of holiday": "next day of holiday",
        "holiday": "holiday",
    }
    return CANON.get(norm(s), norm(s))

def canon_holiday(s):
    CANON = {
        "n/a": "not applicable", "na": "not applicable", "not applicable": "not applicable",
        "non-important holiday": "non-important holiday",
        "important holiday": "important holiday",
        "compulsory holiday": "compulsory holiday",
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
    from collections import defaultdict
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
    df = pd.read_excel(r'C:\Users\Darshan.Pawar\Downloads\embedded code\Tekion\Tekion_dataset.xlsx')
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
    df = pd.read_excel(r'C:\Users\Darshan.Pawar\Downloads\embedded code\Tekion\Tekion_dataset.xlsx')

    # 1) Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2) Normalize key columns
    for c in ["menu_items","sub_category","category","day_type","holiday_type","meal_day","meal_type"]:
        if c in df.columns:
            if c == "category":
                df[c] = df[c].map(canon_category)
            elif c == "day_type":
                df[c] = df[c].map(canon_daytype)
            elif c == "holiday_type":
                df[c] = df[c].map(canon_holiday)
            elif c == "meal_day":
                df[c] = df[c].map(canon_mealday)   # ← add this
            else:
                df[c] = df[c].map(norm)


    # 3) Weekday normalized (only once)
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name().map(norm)

    # 4) Context list
    df["menu_items_list"] = df.groupby("date")["menu_items"].transform(list)


    item_to_subcat = df.set_index('menu_items')['sub_category'].to_dict()
    item_to_cat = df.set_index('menu_items')['category'].to_dict()
    joblib.dump(item_to_subcat, 'item_to_subcat.pkl')
    joblib.dump(item_to_cat, 'item_to_cat.pkl')
    joblib.dump(build_category_to_subcats(item_to_cat, item_to_subcat), 'cat_to_subs.pkl')

    le_map = {}
    for col in ['weekday', 'menu_items', 'sub_category', 'category', 'day_type', 'holiday_type', 'meal_day']:
        le = LabelEncoder()
        df[f"{col}_idx"] = le.fit_transform(df[col])
        joblib.dump(le, f"{col}_encoder.pkl")
        le_map[col] = le

    item_to_idx = {item: idx for idx, item in enumerate(le_map['menu_items'].classes_)}
    pad_idx = len(item_to_idx)
    df['context_seq'] = df['menu_items_list'].map(
        lambda ml: [item_to_idx[i] for i in ml if i in item_to_idx][:10] +
                   [pad_idx] * (10 - len([i for i in ml if i in item_to_idx][:10]))
    )

    X = [
        df['menu_items_idx'], 
        np.array(df['context_seq'].tolist()), 
        df['sub_category_idx'], df['category_idx'], 
        df['weekday_idx'], 
        df['day_type_idx'], df['holiday_type_idx'],
        df['meal_day_idx']
    ]
    y = df['ideal_pp']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(list(zip(*X)), y, test_size=0.2, random_state=42)
    X_train = list(map(lambda arr: np.array(arr), zip(*X_train)))
    X_test = list(map(lambda arr: np.array(arr), zip(*X_test)))

    inputs = [
        tf.keras.Input(shape=(1,), name="item_input"),
        tf.keras.Input(shape=(10,), name="context_input"),
        tf.keras.Input(shape=(1,), name="sub_input"),
        tf.keras.Input(shape=(1,), name="category_input"),
        tf.keras.Input(shape=(1,), name="weekday_input"),
        tf.keras.Input(shape=(1,), name="day_type_input"),
        tf.keras.Input(shape=(1,), name="holiday_type_input"),
        tf.keras.Input(shape=(1,), name="meal_day_input")
    ]

    emb = lambda size, dim: tf.keras.layers.Embedding(size + 1, dim)
    item_e = emb(len(le_map['menu_items'].classes_), 8)(inputs[0])
    context_e = emb(len(le_map['menu_items'].classes_), 8)(inputs[1])
    sub_e = emb(len(le_map['sub_category'].classes_), 4)(inputs[2])
    cat_e = emb(len(le_map['category'].classes_), 4)(inputs[3])
    wd_e = emb(len(le_map['weekday'].classes_), 2)(inputs[4])
    dt_e = emb(len(le_map['day_type'].classes_), 2)(inputs[5])
    hol_e = emb(len(le_map['holiday_type'].classes_), 2)(inputs[6])
    meal_e = emb(len(le_map['meal_day'].classes_), 2)(inputs[7])

    avg_ctx = tf.keras.layers.GlobalAveragePooling1D()(context_e)
    concat = tf.keras.layers.Concatenate()([
        tf.keras.layers.Flatten()(item_e), avg_ctx,
        tf.keras.layers.Flatten()(sub_e), tf.keras.layers.Flatten()(cat_e),
        tf.keras.layers.Flatten()(wd_e),
        tf.keras.layers.Flatten()(dt_e), tf.keras.layers.Flatten()(hol_e),
        tf.keras.layers.Flatten()(meal_e)
    ])
    x = tf.keras.layers.Dense(64, activation='relu')(concat)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

    preds = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    st.success(f"✅ Model trained. RMSE: {rmse:.4f} per pax")

    model.save("per_pax_tf_model.keras")


# --- Section 3: Prediction Function ---

def predict_quantity(target_item, menu_items, item_mg, sub_category, category, weekday,
                     day_type, holiday_type, meal_day):


    model = tf.keras.models.load_model('per_pax_tf_model.keras')
    item_le = joblib.load('menu_items_encoder.pkl')
    sub_le = joblib.load('sub_category_encoder.pkl')
    cat_le = joblib.load('category_encoder.pkl')
    weekday_le = joblib.load('weekday_encoder.pkl')
    day_type_le = joblib.load('day_type_encoder.pkl')
    holiday_type_le = joblib.load('holiday_type_encoder.pkl')
    meal_day_le = joblib.load('meal_day_encoder.pkl')
    item_to_subcat = joblib.load('item_to_subcat.pkl')

    # Canonicalize/normalize all string inputs
    t_item   = (target_item or "").strip()
    m_items  = [ (i or "").strip() for i in menu_items ]
    s_sub    = norm(sub_category)
    s_cat    = canon_category(category)
    s_week   = norm(weekday)              # use day_name_norm
    s_dt     = canon_daytype(day_type)    # use special_day_type_canon
    s_hol    = canon_holiday(holiday_type)# use holiday_type_canon

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
    sub_idx      = safe_transform(sub_le, s_sub, default_idx=0)
    cat_idx      = safe_transform(cat_le, s_cat, default_idx=0)
    weekday_idx  = safe_transform(weekday_le, s_week, default_idx=0)
    day_type_idx = safe_transform(day_type_le, s_dt, default_idx=0)
    holiday_idx  = safe_transform(holiday_type_le, s_hol, default_idx=0)
    meal_idx = safe_transform(meal_day_le, canon_mealday(meal_day), default_idx=0)


    per_pax_qty = model.predict([
        np.array([item_idx]),
        np.array([padded_context]),
        np.array([sub_idx]),
        np.array([cat_idx]),
        np.array([weekday_idx]),
        np.array([day_type_idx]),
        np.array([holiday_idx]),
        np.array([meal_idx]),
    ], verbose=0)[0][0]


    per_pax_qty_rounded = custom_round(per_pax_qty)
    total_qty = per_pax_qty_rounded * item_mg
    return per_pax_qty_rounded, total_qty


# --- Section 4: Streamlit UI Inputs ---

st.title("Per Pax Quantity & Vendor MG Prediction")

# Check model files
required_files = [
    'per_pax_tf_model.keras',
    'menu_items_encoder.pkl','sub_category_encoder.pkl','category_encoder.pkl',
    'weekday_encoder.pkl','day_type_encoder.pkl','holiday_type_encoder.pkl','meal_day_encoder.pkl',
    'item_to_subcat.pkl','item_to_cat.pkl','cat_to_subs.pkl'
]


if not all(os.path.exists(f) for f in required_files):
    st.warning("Some model files are missing. Training model now...")
    train_model()

# Load mappings
item_to_subcat = joblib.load('item_to_subcat.pkl')
item_to_cat    = joblib.load('item_to_cat.pkl')
cat_to_subs    = joblib.load('cat_to_subs.pkl')

# Case-insensitive lookup helper map
item_to_subcat_lc = {norm(k): v for k, v in item_to_subcat.items()}

# Date input
selected_date = st.date_input("Select today's date:")

day_name = selected_date.strftime('%A')
day_name_norm = norm(day_name)

month_name = selected_date.strftime('%B')

# Special day options
special_day_type = st.selectbox("Select Day Type", options=[
    "Regular", "Previous Day of Holiday", "Next Day of Holiday", "Holiday"
])
holiday_type = st.selectbox("Select Holiday Type", options=[
    "Not Applicable", "Non-Important Holiday", "Compulsory Holiday", "Important Holiday"
])

# Canonical versions (used for model + keys)
special_day_type_canon = canon_daytype(special_day_type)
holiday_type_canon = canon_holiday(holiday_type)


is_nonveg_day = st.toggle("🍗 Is it a Non-Veg Day?", value=True)

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
                f"Select sub-category for '{nonveg_item}':",
                options=subcat_options, key="subcat_nonveg"
            )


        #Non veg Client MG
        nonveg_client_mg = st.number_input(
            f"Client MG for Non-Veg Item '{nonveg_item}':", min_value=1, step=1, value=300, key="nonveg_mg"
        )


        item_entries.append((nonveg_item, subcat, canon_category(nonveg_category), nonveg_client_mg))
        menu_items.append(nonveg_item)

# Client MG shared across items
client_mg = st.number_input("Enter shared Client MG for all items:", min_value=1, step=1, value=300)



# Menu items
st.subheader("Enter Menu Items by Category")

fixed_categories = [
    "Flavour Rice", "Indian Bread", "White Rice",
    "Veg Dry", "Veg Curry", "Dal", "Sambar", "Rasam", "Salad"
]



star_items = {"flavour rice", "veg curry"}
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

    if canon_cat_key == "salad":
        subcat = "salad"
    elif key in item_to_subcat_lc:
        subcat = item_to_subcat_lc[key]
        st.text(f"✅ Sub-category: {subcat}")
    else:
        subcat_options = cat_to_subs.get(canon_cat_key, [])
        subcat = st.selectbox(
            f"Select sub-category for '{item}' in {cat}:",
            options=subcat_options, key=f"subcat_{cat}"
        )

    item_entries.append((item, subcat, canon_cat_key, client_mg))



# --- Section 5: Prediction Output + Production Plans ---

if st.button("Predict"):
    st.markdown("### Prediction Results")
    st.markdown(
        f"**Date:** {selected_date} | **Day:** {day_name} | **Month:** {month_name}  \n"
        f"**Day Type:** {special_day_type} | **Holiday Type:** {holiday_type} | "
        f"**Non‑Veg Category:** {nonveg_category}"
    )
    st.markdown("---")

    if not item_entries:
        st.warning("Please add at least one menu item.")
    else:
        results = []
        vendor_mgs = []

        for item, subcat, cat, item_mg in item_entries:
            if cat.strip().lower() == "salad":
                per_pax = 0.06
                total_qty = per_pax * item_mg
            else:
                per_pax, total_qty = predict_quantity(
                                                    target_item=item,
                                                    menu_items=menu_items,
                                                    sub_category=subcat,
                                                    category=cat,
                                                    weekday=day_name_norm,
                                                    day_type=special_day_type_canon,
                                                    holiday_type=holiday_type_canon,
                                                    meal_day=meal_day_type,
                                                    item_mg=item_mg
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

            results.append({
                "Category": cat,
                "Item": item,
                "Client PP": per_pax,
                "Vendor PP": og_prop,
                "Total Qty": total_qty,
                "Vendor MG": vendor_mg,
            })


        if results:
            df_res = pd.DataFrame(results)


            # Round Total Qty to 1 decimal
            df_res["Total Qty"] = df_res["Total Qty"].round(1)
            # Item Vendor MG rounding
            df_res["Vendor MG"] = df_res["Vendor MG"].round(0)


            cleaned_df_res = clean_format(df_res.copy(), ["Client PP", "Total Qty", "Vendor PP", "Vendor MG"])
            #st.markdown("#### 📊 Master Prediction Table")
            #st.dataframe(cleaned_df_res[["Category", "Item", "Client PP", "Total Qty", "Vendor PP", "Vendor MG"]])



            # Group categories
            nonveg_items = [row for row in results if same_cat(row["Category"], nonveg_category)]


            star_item_set = {"flavoured rice", "veg curry"}
            star_entries = [row for row in results if norm(row["Category"]) in star_item_set]
            remaining_entries = [row for row in results if row not in nonveg_items and row not in star_entries]

            # Compute group averages
            def safe_avg(entries):
                return sum(e["Vendor MG"] for e in entries) / len(entries) if entries else 0

            # ---- after computing averages ----
            avg_nonveg    = safe_avg(nonveg_items)
            avg_star      = safe_avg(star_entries)
            avg_remaining = safe_avg(remaining_entries)

            # raw MGs
            raw_veg_mg    = (avg_star + avg_remaining) / 2
            raw_nonveg_mg = avg_nonveg

            # two separate tier fns
            def tier_adjust_veg_mg(mg: float) -> float:
                if mg < 350:
                    return mg * 0.95           # –5%
                elif 351 <= mg <= 700:
                    return mg - 15             # –15 pax
                else:
                    return mg - 25             # –25 pax

            def tier_adjust_nonveg_mg(mg: float) -> float:
                if mg < 400:
                    return mg * 0.97           # –3%
                else:
                    return mg - 20             # –10 pax

            # --- apply & clamp with non-veg toggle in mind ---
            if is_nonveg_day:
                # On a non-veg day, leave the veg MG raw (no veg-tier adjustment)
                final_vendor_mg = raw_veg_mg
            else:
                # On a veg-only day, apply your usual veg-tier logic
                final_vendor_mg = max(tier_adjust_veg_mg(raw_veg_mg), 0)

            # Non-veg always uses its own tier adjustment
            final_nonveg_mg = max(tier_adjust_nonveg_mg(raw_nonveg_mg), 0)




            #st.success(f"✅ Final Vendor MG to be given: **{final_vendor_mg:.0f}**")

            # 2 production plan tables
            client_plan = df_res[["Category", "Item", "Client PP", "Total Qty"]]
            

            cleaned_client_plan = clean_format(client_plan.copy(), ["Client PP", "Total Qty"])
            st.markdown("#### 📋 Client Production Plan")
            if is_nonveg_day and nonveg_item:
                total_client_mg = client_mg
                veg_client_mg = max(total_client_mg - nonveg_client_mg, 0)
                st.write(f"**Veg Client MG: {veg_client_mg} | Non-Veg Client MG: {nonveg_client_mg}**")
            else:
                st.write(f"**Client MG: {client_mg}**")

            st.table(cleaned_client_plan)
            st.markdown("---")


            vendor_plan = df_res.copy()
            # ---- CHECKER: enforce vendor >= 82.5% of client; add any deficit to VEG only ----
            def ceil5(x: float) -> int:
                return int(math.ceil(x / 5.0)) * 5

            veg_mg_rounded = mg_rounder(final_vendor_mg)  # VEG side (all veg categories)
            nv_mg_rounded  = mg_rounder(final_nonveg_mg) if (is_nonveg_day and nonveg_items) else 0

            min_total    = 0.825 * client_mg
            vendor_total = veg_mg_rounded + nv_mg_rounded

            if vendor_total < min_total:
                deficit = min_total - vendor_total
                veg_mg_rounded = ceil5(veg_mg_rounded + deficit)  # add all deficit to VEG (only)

            rounded_vendor_mg = veg_mg_rounded  # use this for all VEG rows
            rounded_veg_vendor_mg = veg_mg_rounded - nv_mg_rounded
            vendor_plan["Ordered Qty"] = vendor_plan.apply(
                lambda row: max(
                    ((nv_mg_rounded if (is_nonveg_day and nonveg_items and same_cat(row["Category"], nonveg_category))
                    else rounded_vendor_mg) * row["Vendor PP"]),
                    row["Total Qty"]
                ),
                axis=1
            )



            vendor_plan["Ordered Qty"] = vendor_plan["Ordered Qty"].round(1)

            vendor_plan = vendor_plan[["Category", "Item", "Vendor PP", "Ordered Qty"]]

            



            cleaned_vendor_plan = clean_format(vendor_plan.copy(), ["Vendor PP", "Ordered Qty"])
            st.markdown("#### 🤝 Vendor Production Plan")
            if is_nonveg_day and nonveg_items:
                st.write(f"**Veg Vendor MG: {rounded_veg_vendor_mg} | Non-Veg Vendor MG: {nv_mg_rounded}**")
            else:
                st.write(f"**Vendor MG: {rounded_vendor_mg}**")



            st.table(cleaned_vendor_plan)
            st.markdown("---")



            # ----------  Aggressive Production Plan  ----------

            star_items = {"flavoured rice", "veg curry"}

            if nonveg_category:
                star_items.add(nonveg_category.strip().lower())

            def slab_bump(weight_kg: float) -> float:
                spiked_w = 1.10 * weight_kg
                spike_size = spiked_w - weight_kg

                if spike_size <= 2:
                    factor = 1.00
                elif spike_size <= 4:
                    factor = 0.35
                elif spike_size <= 6:
                    factor = 0.25
                elif spike_size <= 8:
                    factor = 0.15
                else:
                    factor = 0.10

                extra = factor * spike_size  # ✅ fixed to apply on spike_size
                return round(weight_kg + extra, 1)

            aggr_plan = vendor_plan.copy()

            def compute_aggressive_qty(row):
                category = row["Category"].strip().lower()
                base_qty = row["Ordered Qty"]
                if category in star_items:
                    return slab_bump(base_qty)
                else:
                    return base_qty

            aggr_plan["Ordered Qty"] = aggr_plan.apply(compute_aggressive_qty, axis=1)

            aggr_plan["Ordered Qty"] = aggr_plan["Ordered Qty"].round(1)


            # Step 1: Internal Aggressive MG (mean of bumped MGs)
            aggr_plan["Aggressive MG"] = aggr_plan["Ordered Qty"] / aggr_plan["Vendor PP"]


            # Group by category again
            if nonveg_category:
                nonveg_aggr = aggr_plan[
                    aggr_plan["Category"].notna() &
                    aggr_plan["Category"].str.strip().str.lower().eq(nonveg_category.strip().lower())
                ]
            else:
                nonveg_aggr = pd.DataFrame(columns=aggr_plan.columns)

            star_items = {"flavour rice", "veg curry"}
            star_aggr = aggr_plan[aggr_plan["Category"].str.lower().str.strip().isin(star_items)]
            remaining_aggr = aggr_plan[~aggr_plan.index.isin(nonveg_aggr.index) & ~aggr_plan.index.isin(star_aggr.index)]

            # Calculate safe averages
            def safe_avg_aggr(df):
                return df["Aggressive MG"].mean() if not df.empty else 0

            agg_avg_nonveg_aggr = safe_avg_aggr(nonveg_aggr)
            agg_avg_star_aggr = safe_avg_aggr(star_aggr)
            agg_avg_remaining_aggr = safe_avg_aggr(remaining_aggr)


            internal_aggressive_mg = (agg_avg_star_aggr + agg_avg_remaining_aggr) / 2



            

            # Step 2: Adjusted Aggressive MG
            #for non veg
            nonveg_diff = agg_avg_nonveg_aggr - avg_nonveg
            adjusted_nonveg_agg_mg = avg_nonveg - nonveg_diff
        
            difference = internal_aggressive_mg - final_vendor_mg
            adjusted_aggressive_mg = final_vendor_mg - difference

            adjusted_aggressive_mg = mg_rounder(adjusted_aggressive_mg)
            adjusted_nonveg_agg_mg    = mg_rounder(adjusted_nonveg_agg_mg)
            adjusted_veg_agg_mg = adjusted_aggressive_mg - adjusted_nonveg_agg_mg


            # Rebalance Vendor PP, using non‐veg aggressive MG where applicable
            def rebalance_vendor_pp(row):
                cat = row["Category"].strip().lower()
                # if it's non‐veg day _and_ this row is in your nonveg_category
                if is_nonveg_day and nonveg_category and cat == nonveg_category.strip().lower():
                    base_mg = adjusted_nonveg_agg_mg
                else:
                    base_mg = adjusted_aggressive_mg
                return aggressive_rounding(row["Ordered Qty"] / base_mg)

            aggr_plan["Vendor PP"] = aggr_plan.apply(rebalance_vendor_pp, axis=1)






            # Display
            cleaned_aggr_plan = clean_format(aggr_plan.copy(), ["Vendor PP", "Ordered Qty"])
            st.markdown("#### 🚀 Aggressive Vendor Production Plan")
            if is_nonveg_day and nonveg_items:
                st.write(f"**Veg Aggressive MG: {adjusted_veg_agg_mg} | Non-Veg Aggressive MG: {adjusted_nonveg_agg_mg}**")
            else:
                st.write(f"**Vendor MG: {adjusted_aggressive_mg}**")
            st.table(cleaned_aggr_plan[["Category", "Item", "Vendor PP", "Ordered Qty"]])
            st.markdown("---")






# --- Section 6: Special Day Logic (Non-Regular Days) ---

if special_day_type != "Regular" and st.button("Apply Special Day Logic"):
    st.markdown("### Special Day Adjustment (No Item-Level Prediction)")
    st.markdown(
        f"**Date:** {selected_date} | **Day:** {day_name} | **Month:** {month_name}  \n"
        f"**Day Type:** {special_day_type} | **Holiday Type:** {holiday_type}"
    )
    st.markdown("---")

    key = (
        special_day_type.strip().lower(),
        holiday_type.strip().lower(),
        day_name.strip().lower()
    )

    reduction_dict = {
        ("holiday", "important holiday", "monday"): 12,
        ("holiday", "important holiday", "tuesday"): 10,
        ("holiday", "important holiday", "wednesday"): 10,
        ("holiday", "important holiday", "thursday"): 10,
        ("holiday", "important holiday", "friday"): 12,
        ("holiday", "non-important holiday", "monday"): 13,
        ("holiday", "non-important holiday", "tuesday"): 10,
        ("holiday", "non-important holiday", "wednesday"): 10,
        ("holiday", "non-important holiday", "thursday"): 9,
        ("holiday", "non-important holiday", "friday"): 8,
        ("next day of holiday", "compulsory holiday", "monday"): 10,
        ("next day of holiday", "compulsory holiday", "tuesday"): 8,
        ("next day of holiday", "compulsory holiday", "wednesday"): 9,
        ("next day of holiday", "compulsory holiday", "thursday"): 10,
        ("next day of holiday", "compulsory holiday", "friday"): 11,
        ("next day of holiday", "important holiday", "monday"): 9,
        ("next day of holiday", "important holiday", "tuesday"): 8,
        ("next day of holiday", "important holiday", "wednesday"): 7,
        ("next day of holiday", "important holiday", "thursday"): 8,
        ("next day of holiday", "important holiday", "friday"): 9,
        ("next day of holiday", "non-important holiday", "monday"): 9,
        ("next day of holiday", "non-important holiday", "tuesday"): 9,
        ("next day of holiday", "non-important holiday", "wednesday"): 8,
        ("next day of holiday", "non-important holiday", "thursday"): 8,
        ("next day of holiday", "non-important holiday", "friday"): 9,
        ("previous day of holiday", "compulsory holiday", "monday"): 10,
        ("previous day of holiday", "compulsory holiday", "tuesday"): 9,
        ("previous day of holiday", "compulsory holiday", "wednesday"): 9,
        ("previous day of holiday", "compulsory holiday", "thursday"): 8,
        ("previous day of holiday", "compulsory holiday", "friday"): 10,
        ("previous day of holiday", "important holiday", "monday"): 10,
        ("previous day of holiday", "important holiday", "tuesday"): 6,
        ("previous day of holiday", "important holiday", "wednesday"): 6,
        ("previous day of holiday", "important holiday", "thursday"): 7,
        ("previous day of holiday", "important holiday", "friday"): 9,
        ("previous day of holiday", "non-important holiday", "monday"): 9,
        ("previous day of holiday", "non-important holiday", "tuesday"): 8,
        ("previous day of holiday", "non-important holiday", "wednesday"): 11,
        ("previous day of holiday", "non-important holiday", "thursday"): 10,
        ("previous day of holiday", "non-important holiday", "friday"): 10,
    }

    reduction = reduction_dict.get(key, 10)  # Default reduction if not found
    vendor_mg = client_mg * (100 - reduction) / 100.0

    st.write(f"**Reduction Percentage Applied:** {reduction}%")
    st.success(f"🎯 Adjusted Vendor MG: **{vendor_mg:.2f}**")
