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

def custom_round(value):
    return math.floor(value / 0.005) * 0.005

def bumped_ceil_og_prop(per_pax):
    bumped = per_pax + 0.01
    return math.ceil(bumped * 100) / 100.0


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

# --- Normalization helpers (keep all labels consistent) ---
def norm(s):
    return s.strip().lower() if isinstance(s, str) else s

CATEGORY_CANON = {
    "indian breads": "indian bread",
    "indian bread": "indian bread",
    "veg dry": "veg dry",
    "veg gravy": "veg curry",
    "veg curry": "veg curry",
    "flavoured rice": "flavoured rice",
    "flavour rice": "flavoured rice",
    "white rice": "white rice",
    "steamed rice": "white rice",
    "dal": "dal",
    "sambar": "sambar",
    "rasam": "rasam",
    "dal/sambar": "dal/sambar"
}

HOLIDAY_CANON = {
    "not applicable": "not applicable",
    "n/a": "not applicable",
    "na": "not applicable",
    "non-important holiday": "non-important holiday",
    "important holiday": "important holiday",
    "compulsory holiday": "compulsory holiday"
}

DAYTYPE_CANON = {
    "regular": "regular",
    "previous day of holiday": "previous day of holiday",
    "next day of holiday": "next day of holiday",
    "holiday": "holiday",
}

def canon_category(s): return CATEGORY_CANON.get(norm(s), norm(s))
def canon_holiday(s):  return HOLIDAY_CANON.get(norm(s), norm(s))
def canon_daytype(s):  return DAYTYPE_CANON.get(norm(s), norm(s))

# --- Fixed Client PP by Category (Odessia) ---
FIXED_PP_BY_CATEGORY = {
    # Veg buckets
    "dal/sambar": 0.066,        # combined bucket
    "flavoured rice": 0.102,
    "indian bread": 0.05,
    "rasam": 0.033,
    "veg dry": 0.075,
    "veg curry": 0.102,         # "Veg Gravy" -> "veg curry"
    "white rice": 0.088,

    # Non-veg buckets (from meal_type == "non veg")
    "non veg gravy": 0.122,
    "non veg biryani": 0.22,
    "non veg dry": 0.099,
}

DEFAULT_FIXED_PP = 0.10  # fallback if a category isn't in the map

def fixed_pp_for_category(cat: str) -> float:
    return FIXED_PP_BY_CATEGORY.get(canon_category(cat), DEFAULT_FIXED_PP)


def normalize_item_key(s):  # for dict keys
    return norm(s)

def case_insensitive_item_lookup(name, known_items):
    """Return the exact-cased known item if lowercased matches; else original name."""
    lower_to_exact = {k.lower(): k for k in known_items}
    return lower_to_exact.get(name.lower(), name)



def get_fallback_item_from_subcat(sub_category, item_le, item_to_subcat):
    for item, subcat in item_to_subcat.items():
        if subcat == sub_category and item in item_le.classes_:
            return item
    return None

def build_category_to_subcats(item_to_cat, item_to_subcat):
    from collections import defaultdict
    cat_to_subs = defaultdict(set)
    for item in item_to_cat:
        cat = item_to_cat[item]
        sub = item_to_subcat.get(item)
        if cat and sub:
            cat_to_subs[cat].add(sub)
    return {k: sorted(v) for k, v in cat_to_subs.items()}

@st.cache_data
def get_nonveg_categories_from_data():
    df = pd.read_excel(r'C:\Users\Darshan.Pawar\Downloads\embedded code\Odessia\Odessia Dataset.xlsx')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Safe non-veg filter (handles NaN) + canonicalize category names
    is_nonveg = df["meal_type"].fillna("").str.lower() == "non veg"
    df = df[is_nonveg].copy()
    df["category"] = df["category"].apply(canon_category)

    return sorted(df["category"].dropna().unique())


def train_model():
    st.info("Training per‑pax quantity model…")
    df = pd.read_excel(r'C:\Users\Darshan.Pawar\Downloads\embedded code\Odessia\Odessia Dataset.xlsx')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['category'] = df['category'].apply(canon_category)
    df['sub_category'] = df['sub_category'].apply(norm)

    # Ensure day/holiday columns exist, then canonicalize
    if 'day_type' not in df.columns:
        df['day_type'] = 'regular'
    else:
        df['day_type'] = df['day_type'].apply(canon_daytype)

    if 'holiday_type' not in df.columns:
        df['holiday_type'] = 'not applicable'
    else:
        df['holiday_type'] = df['holiday_type'].apply(canon_holiday)

    # Always build these (not inside the else)
    df['menu_items_list'] = df.groupby('date')['menu_items'].transform(list)
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.day_name().str.lower()   # <- lowercase to match prediction

    

    item_to_subcat = df.set_index('menu_items')['sub_category'].to_dict()
    item_to_cat = df.set_index('menu_items')['category'].to_dict()
    joblib.dump(item_to_subcat, 'item_to_subcat.pkl')
    joblib.dump(item_to_cat, 'item_to_cat.pkl')
    joblib.dump(build_category_to_subcats(item_to_cat, item_to_subcat), 'cat_to_subs.pkl')

    le_map = {}
    for col in ['weekday', 'menu_items', 'sub_category', 'category', 'day_type', 'holiday_type']:
        le = LabelEncoder()
        df[f"{col}_idx"] = le.fit_transform(df[col])
        joblib.dump(le, f"{col}_encoder.pkl")
        le_map[col] = le

    item_to_idx = {item: idx for idx, item in enumerate(le_map['menu_items'].classes_)}
    pad_idx = len(item_to_idx)
    df['context_list'] = df.apply(lambda r: [i for i in r['menu_items_list'] if i != r['menu_items']], axis=1)
    df['context_seq'] = df['context_list'].map(
        lambda ml: [item_to_idx[i] for i in ml if i in item_to_idx][:10] +
                [pad_idx] * (10 - len([i for i in ml if i in item_to_idx][:10]))
    )

    X = [
        df['menu_items_idx'], 
        np.array(df['context_seq'].tolist()), 
        df['sub_category_idx'], df['category_idx'], 
        df['weekday_idx'], 
        df['day_type_idx'], df['holiday_type_idx']
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
        tf.keras.Input(shape=(1,), name="holiday_type_input")
       
    ]

    emb = lambda size, dim: tf.keras.layers.Embedding(size + 1, dim)
    item_e = emb(len(le_map['menu_items'].classes_), 8)(inputs[0])
    context_e = emb(len(le_map['menu_items'].classes_), 8)(inputs[1])
    sub_e = emb(len(le_map['sub_category'].classes_), 4)(inputs[2])
    cat_e = emb(len(le_map['category'].classes_), 4)(inputs[3])
    wd_e = emb(len(le_map['weekday'].classes_), 2)(inputs[4])
    dt_e = emb(len(le_map['day_type'].classes_), 2)(inputs[5])
    hol_e = emb(len(le_map['holiday_type'].classes_), 2)(inputs[6])
    

    avg_ctx = tf.keras.layers.GlobalAveragePooling1D()(context_e)
    concat = tf.keras.layers.Concatenate()([
        tf.keras.layers.Flatten()(item_e), avg_ctx,
        tf.keras.layers.Flatten()(sub_e), tf.keras.layers.Flatten()(cat_e),
        tf.keras.layers.Flatten()(wd_e),
        tf.keras.layers.Flatten()(dt_e), tf.keras.layers.Flatten()(hol_e)
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

def predict_quantity(target_item, menu_items, client_mg, sub_category, category, weekday,
                     day_type="Regular", holiday_type="n/a"):
    
    # Canonicalize all string inputs
    target_item  = target_item.strip()
    menu_items   = [m.strip() for m in menu_items]
    sub_category = norm(sub_category)
    category     = canon_category(category)
    weekday      = norm(weekday)
    day_type     = canon_daytype(day_type)
    holiday_type = canon_holiday(holiday_type)


    model = tf.keras.models.load_model('per_pax_tf_model.keras')
    item_le = joblib.load('menu_items_encoder.pkl')
    sub_le = joblib.load('sub_category_encoder.pkl')
    cat_le = joblib.load('category_encoder.pkl')
    weekday_le = joblib.load('weekday_encoder.pkl')
    day_type_le = joblib.load('day_type_encoder.pkl')
    holiday_type_le = joblib.load('holiday_type_encoder.pkl')
    item_to_subcat = joblib.load('item_to_subcat.pkl')

    known_items = set(item_le.classes_)
    pad_idx = len(item_le.classes_)

    # Handle unseen item using sub-category fallback
    target_item_exact = case_insensitive_item_lookup(target_item, known_items)

    if target_item_exact not in known_items:
        fallback_item = get_fallback_item_from_subcat(sub_category, item_le, item_to_subcat)
        if fallback_item:
            st.warning(f"⚠️ '{target_item}' not seen during training. Using fallback item '{fallback_item}'.")
            item_idx = item_le.transform([fallback_item])[0]
        else:
            st.error(f"❌ No fallback item found for sub-category '{sub_category}'. Skipping.")
            return None, None
    else:
        item_idx = item_le.transform([target_item_exact])[0]

    context_idxs = []
    for i in menu_items:
        if i.strip().lower() == target_item.strip().lower():
            continue
        i_exact = case_insensitive_item_lookup(i, known_items)
        if i_exact in known_items:
            context_idxs.append(item_le.transform([i_exact])[0])


    sub_idx = sub_le.transform([sub_category])[0] if sub_category in sub_le.classes_ else 0
    cat_idx = cat_le.transform([category])[0] if category in cat_le.classes_ else 0
    weekday_idx = weekday_le.transform([weekday])[0]
    day_type_idx = day_type_le.transform([day_type])[0]
    holiday_type_idx = holiday_type_le.transform([holiday_type])[0]
    padded_context = context_idxs[:10] + [pad_idx] * (10 - len(context_idxs))


    # Predict per pax quantity
    per_pax_qty = model.predict([
        np.array([item_idx]),
        np.array([padded_context]),
        np.array([sub_idx]),
        np.array([cat_idx]),
        np.array([weekday_idx]),
        np.array([day_type_idx]),
        np.array([holiday_type_idx])
    ], verbose=0)[0][0]

    per_pax_qty_rounded = custom_round(per_pax_qty)
    total_qty = per_pax_qty_rounded * client_mg
    return per_pax_qty_rounded, total_qty



# --- Section 4: Streamlit UI Inputs ---

st.title("Per Pax Quantity & Vendor MG Prediction")

# Check model files
required_files = [
    'per_pax_tf_model.keras', 'menu_items_encoder.pkl', 'sub_category_encoder.pkl',
    'category_encoder.pkl', 'weekday_encoder.pkl',
    'day_type_encoder.pkl', 'holiday_type_encoder.pkl',
    'item_to_subcat.pkl', 'item_to_cat.pkl', 'cat_to_subs.pkl'
]

if not all(os.path.exists(f) for f in required_files):
    st.warning("Some model files are missing. Training model now...")
    train_model()

# Case-insensitive maps for items
item_to_subcat = joblib.load('item_to_subcat.pkl')
item_to_cat    = joblib.load('item_to_cat.pkl')
cat_to_subs    = joblib.load('cat_to_subs.pkl')

item_to_subcat_norm = {normalize_item_key(k): v for k, v in item_to_subcat.items()}
item_to_cat_norm    = {normalize_item_key(k): v for k, v in item_to_cat.items()}


# Date input
selected_date = st.date_input("Select today's date:")
day_name = selected_date.strftime('%A')
month_name = selected_date.strftime('%B')

# Special day options
special_day_type = st.selectbox("Select Day Type", options=[
    "Regular", "Previous Day of Holiday", "Next Day of Holiday", "Holiday"
])
holiday_type = st.selectbox("Select Holiday Type", options=[
    "Not Applicable", "Non-Important Holiday", "Compulsory Holiday", "Important Holiday"
])

# Non-Veg category selection from training data
nonveg_options = get_nonveg_categories_from_data()
nonveg_category = st.selectbox("Select Non-Veg Category", options=nonveg_options)

# Client MG shared across items
client_mg = st.number_input("Enter shared Client MG for all items:", min_value=1, step=1, value=300)

# Menu items
st.subheader("Enter Menu Items by Category")

fixed_categories = [
    "Dal/Sambar",
    "Flavoured rice",
    "Indian Bread",
    "Rasam",
    "Veg dry",
    "Veg Gravy",
    "White Rice"
]



item_entries = []
menu_items = []

star_items = {"flavoured rice", "veg gravy"}
for cat in fixed_categories:
    label = f"Item name for {cat}:"
    if cat.strip().lower() in star_items:
        label = f"⭐ {label}"
    item = st.text_input(label, key=f"item_{cat}")

    if not item:
        continue
    menu_items.append(item)

    user_item_key = normalize_item_key(item)
    if user_item_key in item_to_subcat_norm:
        subcat = item_to_subcat_norm[user_item_key]
        st.text(f"✅ Sub-category: {subcat}")
    else:
        subcat_options = cat_to_subs.get(canon_category(cat), [])
        subcat = st.selectbox(
            f"Select sub-category for '{item}' in {cat}:",
            options=subcat_options, key=f"subcat_{cat}"
        )
    item_entries.append((item, subcat, canon_category(cat)))



# Non-Veg Item
st.subheader("Enter One Non-Veg Item")
nonveg_item = st.text_input(f"Non-Veg Item Name for {nonveg_category}:", key="nonveg_item")
if nonveg_item:
    nv_key = normalize_item_key(nonveg_item)
    if nv_key in item_to_subcat_norm:
        subcat = item_to_subcat_norm[nv_key]
        st.text(f"✅ Sub-category: {subcat}")
    else:
        subcat_options = cat_to_subs.get(canon_category(nonveg_category), [])
        subcat = st.selectbox(
            f"Select sub-category for '{nonveg_item}':",
            options=subcat_options, key="subcat_nonveg"
        )
    item_entries.append((nonveg_item, subcat, canon_category(nonveg_category)))
    menu_items.append(nonveg_item)

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

        for item, subcat, cat in item_entries:
            per_pax, total_qty = predict_quantity(
                target_item=item,
                menu_items=menu_items,
                client_mg=client_mg,
                sub_category=subcat,
                category=cat,
                weekday=day_name,
                day_type=special_day_type,
                holiday_type=holiday_type,
            )
            if per_pax is None:
                continue

            og_prop = bumped_ceil_og_prop(per_pax)
            vendor_mg = total_qty / og_prop if og_prop else 0
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
            nv_key = canon_category(nonveg_category)
            nonveg_items = [r for r in results if canon_category(r["Category"]) == nv_key]
            star_items = {canon_category(s) for s in ["flavoured rice", "veg gravy"]}
            star_entries = [r for r in results if canon_category(r["Category"]) in star_items]

            remaining_entries = [row for row in results if row not in nonveg_items and row not in star_entries]

            # Compute group averages
            def safe_avg(entries):
                return sum(e["Vendor MG"] for e in entries) / len(entries) if entries else 0

            avg_nonveg = safe_avg(nonveg_items)
            avg_star = safe_avg(star_entries)
            avg_remaining = safe_avg(remaining_entries)

            # Final vendor MG as average of the 3 groups
            final_vendor_mg = (avg_nonveg + avg_star + avg_remaining) / 3

            if final_vendor_mg < 120:
                # reduce by 6%
                final_vendor_mg = final_vendor_mg * 0.94
            elif 121 <= final_vendor_mg <= 230:
                final_vendor_mg = final_vendor_mg - 5
            elif 231 <= final_vendor_mg <= 400:
                final_vendor_mg = final_vendor_mg - 10
            else:
                final_vendor_mg = final_vendor_mg - 15


            #st.success(f"✅ Final Vendor MG to be given: **{final_vendor_mg:.0f}**")

            # --- Client Production Plan (fixed PP per category) ---
            client_plan = df_res[["Category", "Item"]].copy()
            client_plan["Client PP"] = client_plan["Category"].map(fixed_pp_for_category)

            # Keep Total Qty numeric, computed from the unrounded PP
            client_plan["Total Qty"] = client_plan["Client PP"] * client_mg

            # Display: force Client PP to exactly 3 decimals; only format Total Qty
            client_plan_display = client_plan.copy()
            client_plan_display["Client PP"] = client_plan_display["Client PP"].map(lambda x: f"{x:.3f}")
            cleaned_client_plan = clean_format(client_plan_display.copy(), ["Total Qty"])

            st.markdown("#### 📋 Client Production Plan (Fixed PP)")
            st.write(f"**Client MG: {client_mg}**")
            st.table(cleaned_client_plan)
            st.markdown("---")


            # --- Vendor Production Plan ---
            vendor_plan = df_res.copy()
            rounded_vendor_mg = mg_rounder(final_vendor_mg)

            # Internal quantity used for downstream logic (kept as before)
            vendor_plan["Ordered Qty Internal"] = vendor_plan.apply(
                lambda row: max(row["Vendor PP"] * rounded_vendor_mg, row["Total Qty"]),
                axis=1
            ).round(1)

            # Display quantity: strictly Vendor PP * Vendor MG (as requested)
            vendor_plan["Ordered Qty"] = vendor_plan["Vendor PP"] * rounded_vendor_mg

            # What we show in the table
            vendor_plan_display = vendor_plan[["Category", "Item", "Vendor PP", "Ordered Qty"]].copy()
            cleaned_vendor_plan = clean_format(vendor_plan_display.copy(), ["Vendor PP", "Ordered Qty"])

            st.markdown("#### 🤝 Vendor Production Plan")
            st.write(f"**Vendor MG: {rounded_vendor_mg}**")
            st.table(cleaned_vendor_plan)
            st.markdown("---")




            # ----------  Aggressive Production Plan  ----------

            star_items = {s.strip().lower() for s in [
                "flavoured rice",
                "veg gravy"
            ]}
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
                base_qty = row["Ordered Qty Internal"]
                if category in star_items:
                    return slab_bump(base_qty)
                else:
                    return base_qty

            aggr_plan["Ordered Qty"] = aggr_plan.apply(compute_aggressive_qty, axis=1)

            aggr_plan["Ordered Qty"] = aggr_plan["Ordered Qty"].round(1)


            # Step 1: Internal Aggressive MG (mean of bumped MGs)
            aggr_plan["Aggressive MG"] = aggr_plan["Ordered Qty"] / aggr_plan["Vendor PP"]


            # Group by category again
            nonveg_aggr = aggr_plan[aggr_plan["Category"].str.lower().str.strip() == nonveg_category.strip().lower()]
            star_items = {"flavoured rice", "veg gravy"}
            star_aggr = aggr_plan[aggr_plan["Category"].str.lower().str.strip().isin(star_items)]
            remaining_aggr = aggr_plan[~aggr_plan.index.isin(nonveg_aggr.index) & ~aggr_plan.index.isin(star_aggr.index)]

            # Calculate safe averages
            def safe_avg_aggr(df):
                return df["Aggressive MG"].mean() if not df.empty else 0

            avg_nonveg_aggr = safe_avg_aggr(nonveg_aggr)
            avg_star_aggr = safe_avg_aggr(star_aggr)
            avg_remaining_aggr = safe_avg_aggr(remaining_aggr)

            # Compute group-averaged internal aggressive MG
            internal_aggressive_mg = (avg_nonveg_aggr + avg_star_aggr + avg_remaining_aggr) / 3


            

            # Step 2: Adjusted Aggressive MG
            difference = internal_aggressive_mg - final_vendor_mg
            adjusted_aggressive_mg = final_vendor_mg - difference

            adjusted_aggressive_mg = mg_rounder(adjusted_aggressive_mg)


            # Rebalance Vendor PP to match adjusted aggressive MG
            aggr_plan["Vendor PP"] = aggr_plan["Ordered Qty"] / adjusted_aggressive_mg
            aggr_plan["Vendor PP"] = aggr_plan["Vendor PP"].apply(aggressive_rounding)





            # Display
            cleaned_aggr_plan = clean_format(aggr_plan.copy(), ["Vendor PP", "Ordered Qty"])
            st.markdown("#### 🚀 Aggressive Vendor Production Plan")
            st.write(f"**Adjusted Aggressive MG: {adjusted_aggressive_mg}**")
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
        canon_daytype(special_day_type),
        canon_holiday(holiday_type),
        norm(day_name)
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
