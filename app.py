"""
UI — Streamlit app with client dropdown. Dynamic UI per client capabilities.
"""
import os,warnings,logging
import streamlit as st, pandas as pd
from client_database import CLIENT_LIST, name_to_key, get_info
from client_logic import get_logic
from ml_core import (artifacts_exist,clear_cache,load_map,norm,fmt_cols,cats_match,
                     get_nv_cats,predict,train_model,floor005,load_enc)
from planner import (build_row,client_plan,fixed_pp_client_plan,vendor_plan,
                     aggressive_plan,special_day_mg,gavg,classify,mg5,r005)

warnings.filterwarnings("ignore",category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ═══════════════ CLIENT SELECTOR ═══════════════
st.title("Per Pax Quantity & Production Plan Prediction")
sel=st.selectbox("Select Client",CLIENT_LIST,key="client_sel")
CK=name_to_key(sel); INFO=get_info(CK); L=get_logic(CK)
st.caption(f"Client: **{sel}** | Mode: **{'Embedding' if L.has_embeddings else 'Multiplier-only'}**")

# ═══════════════ TOASTTAB (multiplier-only, separate UI) ═══════════════
if not L.has_embeddings:
    from datetime import date as _d, datetime
    try:
        from zoneinfo import ZoneInfo; _today=datetime.now(ZoneInfo("Asia/Kolkata")).date()
    except: _today=_d.today()
    c1,c2=st.columns(2)
    with c1: sd=st.date_input("Date",value=_today)
    with c2: cmg=st.number_input("Client MG",min_value=1,step=1,value=L.default_mg)
    r5=st.checkbox("Round to nearest 5",value=True)
    adj=L.toasttab_adjust(cmg)
    vmg=mg5(adj) if r5 else int(adj)
    st.divider()
    st.subheader(f"Vendor MG for {sd:%A, %d %b %Y}")
    a,b,c=st.columns(3)
    with a: st.metric("Client MG",f"{cmg}")
    with b: st.metric("Adjusted",f"{adj:.0f}")
    with c: st.metric("Vendor MG",f"{vmg}")
    st.stop()

# ═══════════════ ENSURE ARTIFACTS ═══════════════
def _ensure():
    if artifacts_exist(CK,L.encoder_columns): return
    ds=INFO["dataset"]
    if not ds: st.error(f"No dataset for {sel}."); st.stop()
    st.warning(f"Training model for {sel}...")
    _,rmse=train_model(CK,ds,L)
    clear_cache(CK)
    st.success(f"Model trained for {sel}. RMSE: {rmse:.4f}")
_ensure()

# ═══════════════ LOAD MAPPINGS ═══════════════
i2s=load_map(CK,"item_to_subcat"); i2c=load_map(CK,"item_to_cat"); c2s=load_map(CK,"cat_to_subs")
i2s_lc={norm(k):v for k,v in i2s.items()}

# ═══════════════ INPUTS ═══════════════
sel_date=st.date_input("Select today's date:")
day_name=sel_date.strftime("%A"); day_norm=norm(day_name); month=sel_date.strftime("%B")

# Day type / holiday (if client supports special day)
if L.has_special_day:
    sdt=st.selectbox("Day Type",["Regular","Previous Day of Holiday","Next Day of Holiday","Holiday"])
    ht=st.selectbox("Holiday Type",["Not Applicable","Non-Important Holiday","Compulsory Holiday","Important Holiday"])
else:
    sdt="Regular"; ht="Not Applicable"
cdt=L.canonicalize_day_type(sdt); cht=L.canonicalize_holiday_type(ht)

# Nonveg toggle
is_nv=False; meal_day="veg"; nv_item=None; nv_mg=0; nv_cat=None
nv_items_list=[]  # for Stripe's 2-nonveg

if L.has_nonveg_toggle:
    is_nv=st.toggle("🍗 Non-Veg Day?",value=True)
    meal_day="nonveg" if is_nv else "veg"

entries=[]; menu=[]

# Nonveg item(s) — shown if toggle on OR if client always has nonveg
show_nv = (L.has_nonveg_toggle and is_nv) or (not L.has_nonveg_toggle and L.nonveg_item_count>0 and L.has_vendor_plan)
if show_nv:
    nv_opts=get_nv_cats(CK,INFO["dataset"],L) if INFO["dataset"] else []
    if nv_opts:
        for ni in range(L.nonveg_item_count):
            nvc=st.selectbox(f"Non-Veg Category{f' #{ni+1}' if L.nonveg_item_count>1 else ''}",nv_opts,key=f"nvc_{ni}")
            nvi=st.text_input(f"Non-Veg Item for {nvc}:",key=f"nvi_{ni}")
            if nvi:
                k=norm(nvi)
                if k in i2s_lc: sc=i2s_lc[k]; st.text(f"✅ Sub-category: {sc}")
                else:
                    opts=c2s.get(L.canonicalize_category(nvc),[])
                    sc=st.selectbox(f"Sub-cat for '{nvi}':",opts,key=f"nvsc_{ni}") if opts else st.text_input(f"Sub-cat for '{nvi}':",key=f"nvsc_{ni}_t")
                if L.has_nonveg_toggle:
                    nv_mg=st.number_input(f"Client MG for '{nvi}':",min_value=1,step=1,value=L.default_mg,key=f"nvmg_{ni}")
                    entries.append((nvi,sc,L.canonicalize_category(nvc),nv_mg)); nv_item=nvi
                else:
                    entries.append((nvi,sc,L.canonicalize_category(nvc),0))  # MG filled later
                menu.append(nvi)
                nv_items_list.append(nvc)
            if ni==0: nv_cat=nvc  # primary nonveg category

cmg=st.number_input("Shared Client MG:",min_value=1,step=1,value=L.default_mg)
# patch nonveg entries that used 0 MG
entries=[(it,sc,ca,cmg if mg==0 else mg) for it,sc,ca,mg in entries]

st.subheader("Enter Menu Items by Category")
star_ui={"flavour rice","flavoured rice","veg curry","veg gravy"}
north_pp=None  # for Rippling

for cat in L.fixed_categories:
    lbl=f"Item name for {cat}:"
    if cat.strip().lower() in star_ui: lbl=f"⭐ {lbl}"
    item=st.text_input(lbl,key=f"item_{cat}")
    if not item: continue
    menu.append(item)
    k=norm(item)
    icat=L.category_display_map.get(cat,cat)  # map display→internal
    cc=L.canonicalize_category(icat)
    if cc=="salad": sc="salad"
    elif k in i2s_lc: sc=i2s_lc[k]; st.text(f"✅ Sub-category: {sc}")
    else:
        opts=c2s.get(cc,[]); sc=st.selectbox(f"Sub-cat for '{item}' ({cat}):",opts,key=f"sc_{cat}") if opts else st.text_input(f"Sub-cat for '{item}':",key=f"sc_{cat}_t")
    entries.append((item,sc,cc,cmg,cat))  # 5-tuple with display_cat

# ═══════════════ PREDICT ═══════════════
if st.button("Predict"):
    st.markdown(f"### Prediction Results — {sel}")
    st.markdown(f"**Date:** {sel_date} | **Day:** {day_name} | **Month:** {month}")
    if L.has_special_day: st.markdown(f"**Day Type:** {sdt} | **Holiday Type:** {ht}")
    st.markdown("---")

    if not entries: st.warning("Add at least one menu item."); st.stop()

    results=[]
    for e in entries:
        if len(e)==5: item,sc,cat,img,dcat=e
        else: item,sc,cat,img=e; dcat=cat

        # Rippling: South Veg dry copies North's PP
        if dcat=="South Veg dry" and north_pp is not None:
            pp,tq=north_pp
        elif norm(cat)=="salad":
            pp=L.salad_per_pax; tq=pp*img
        else:
            pr=predict(CK,item,menu,img,sc,cat,day_norm,L,day_type=cdt,holiday_type=cht,meal_day=meal_day)
            if not pr.ok: st.error(f"❌ {pr.error} Skipping '{item}'."); continue
            if pr.fallback: st.warning(f"⚠️ '{item}' unseen, fallback '{pr.fallback_item}'.")
            pp=pr.per_pax; tq=pr.total_qty

        if dcat=="North Veg dry": north_pp=(pp,tq)

        row=build_row(pp,tq,L.canonicalize_category(dcat) if dcat!=cat else cat,item,is_nv,nv_cat,L)
        # keep display category for Rippling
        if dcat in ("North Veg dry","South Veg dry"): row["Category"]=dcat
        results.append(row)

    if not results: st.stop()

    df=pd.DataFrame(results); df["Total Qty"]=df["Total Qty"].round(1); df["Vendor MG"]=df["Vendor MG"].round(0)

    # ── CLIENT PLAN ──
    if L.fixed_pp_map:
        cp=fixed_pp_client_plan(df,L.fixed_pp_map,cmg)
    else:
        cp=client_plan(df)
    st.markdown("#### 📋 Client Production Plan")
    if L.has_nonveg_toggle and is_nv and nv_item:
        st.write(f"**Veg MG: {max(cmg-nv_mg,0)} | Non-Veg MG: {nv_mg}**")
    else:
        st.write(f"**Client MG: {cmg}**")
    st.table(fmt_cols(cp,["Client PP","Total Qty"]))
    st.markdown("---")

    # ── VENDOR PLAN ──
    if L.has_vendor_plan:
        vp,vmg,nvmg=vendor_plan(df,results,cmg,is_nv,nv_cat,L,weekday=day_norm)
        st.markdown("#### 🤝 Vendor Production Plan")
        if L.has_separate_nonveg_mg and nvmg>0:
            st.write(f"**Veg Vendor MG: {vmg-nvmg} | Non-Veg Vendor MG: {nvmg}**")
        else:
            label="Vendor MG (day-based)" if L.vendor_mg_method=="day_based" else "Vendor MG"
            st.write(f"**{label}: {vmg}**")
        st.table(fmt_cols(vp,["Vendor PP","Ordered Qty"]))
        st.markdown("---")

        # ── AGGRESSIVE PLAN ──
        if L.has_aggressive_plan:
            if L.vendor_mg_method=="tekion_2group":
                fv,_=classify(results,nv_cat,L.star_categories); fv_mg=_tekion_fv(results,is_nv,nv_cat,L)
                avg_nv=gavg([r for r in results if cats_match(r["Category"],nv_cat)])
                ag,at,an,av=aggressive_plan(vp,results,fv_mg,avg_nv,is_nv,nv_cat,L,method_groups=2)
            elif L.vendor_mg_method in ("3group","day_based"):
                ag,at,an,av=aggressive_plan(vp,results,vmg,0,is_nv,nv_cat,L,method_groups=3)
            else:
                ag,at,an,av=aggressive_plan(vp,results,vmg,0,is_nv,nv_cat,L,method_groups=3)

            st.markdown("#### 🚀 Aggressive Vendor Production Plan")
            if L.has_separate_nonveg_mg and an>0:
                st.write(f"**Veg Aggressive MG: {av} | Non-Veg Aggressive MG: {an}**")
            else:
                st.write(f"**Adjusted Aggressive MG: {at}**")
            st.table(fmt_cols(ag,["Vendor PP","Ordered Qty"]))
            st.markdown("---")

    # ── SPECIAL DAY ──
    if L.has_special_day and sdt!="Regular" and st.button("Apply Special Day Logic"):
        vmg_sd,pct=special_day_mg(cmg,sdt,ht,day_name,L)
        st.write(f"**Reduction: {pct}%**")
        st.success(f"🎯 Adjusted Vendor MG: **{vmg_sd:.2f}**")


# helper for tekion aggressive plan
def _tekion_fv(results,is_nv,nv_cat,L):
    nv,st2,rest=classify(results,nv_cat,L.star_categories)
    raw=(gavg(st2)+gavg(rest))/2
    return raw if is_nv else max(L.adjust_vendor_mg(raw),0)
