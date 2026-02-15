"""
Planner — production plan builders supporting all client vendor MG methods.

Supports: tekion_2group, 3group, day_based, toasttab_formula, and None.
All functions take a logic object so business rules are pluggable.
"""
import math, pandas as pd
from ml_core import norm, cats_match

def mg5(x): return int(round(x/5.0)*5)
def ceil5(x): return int(math.ceil(x/5.0)*5)
def r005(v): return round(v*200)/200.0
def vpp(pp,bump): return math.ceil((pp+bump)*100)/100.0
def gavg(entries,key="Vendor MG"): return sum(e[key] for e in entries)/len(entries) if entries else 0

def classify(results, nv_cat, stars):
    nv=[r for r in results if cats_match(r["Category"],nv_cat)]
    st=[r for r in results if norm(r["Category"]) in stars]
    rest=[r for r in results if r not in nv and r not in st]
    return nv,st,rest

# ── build one result row ──
def build_row(pp,tq,cat,item,is_nv_day,nv_cat,logic):
    is_biry=(is_nv_day and nv_cat and norm(cat)==norm(nv_cat) and norm(nv_cat)=="non veg biryani")
    bump=logic.biryani_bump if is_biry else logic.default_bump
    vp=vpp(pp,bump); vmg=tq/max(vp,1e-9)
    return {"Category":cat,"Item":item,"Client PP":pp,"Vendor PP":vp,"Total Qty":tq,"Vendor MG":vmg}

# ── client plan ──
def client_plan(df):
    return df[["Category","Item","Client PP","Total Qty"]].copy()

# ── fixed-PP client plan (Odessia) ──
def fixed_pp_client_plan(df, fpp_map, mg):
    p=df[["Category","Item"]].copy()
    p["Client PP"]=p["Category"].map(lambda c: fpp_map.get(norm(c),0.10))
    p["Total Qty"]=(p["Client PP"]*mg).round(1)
    return p

# ══════════════ VENDOR MG COMPUTATION ══════════════

def _tekion_vendor_mgs(results, is_nv, nv_cat, logic):
    """Tekion: (star+remaining)/2 for veg, separate nonveg track, tier adjust, floor check."""
    nv,st,rest=classify(results,nv_cat,logic.star_categories)
    raw_veg=(gavg(st)+gavg(rest))/2; raw_nv=gavg(nv)
    fv=raw_veg if is_nv else max(logic.adjust_vendor_mg(raw_veg),0)
    fnv=max(logic.adjust_nonveg_vendor_mg(raw_nv),0)
    return fv,fnv

def _3group_vendor_mg(results, nv_cat, logic):
    """Odessia/Rippling: (nonveg+star+remaining)/3, single tier adjust."""
    nv,st,rest=classify(results,nv_cat,logic.star_categories)
    raw=(gavg(nv)+gavg(st)+gavg(rest))/3
    return max(logic.adjust_vendor_mg(raw),0)

def _day_vendor_mg(client_mg, weekday, logic):
    """Stripe: day-based percentage reduction."""
    pct=logic.day_reductions.get(norm(weekday),0.22)
    return client_mg*(1-pct)

# ── vendor plan builder ──
def vendor_plan(df, results, cmg, is_nv, nv_cat, logic, weekday=None):
    method=logic.vendor_mg_method
    if method=="tekion_2group":
        fv,fnv=_tekion_vendor_mgs(results,is_nv,nv_cat,logic)
        nv_rows,_,_=classify(results,nv_cat,logic.star_categories)
        vmg=mg5(fv); nvmg=mg5(fnv) if (is_nv and nv_rows) else 0
        mn=logic.vendor_floor_ratio*cmg
        if (vmg+nvmg)<mn: vmg=ceil5(vmg+(mn-vmg-nvmg))
        vp=df.copy()
        vp["Ordered Qty"]=vp.apply(lambda r:max((nvmg if(is_nv and nv_rows and cats_match(r["Category"],nv_cat)) else vmg)*r["Vendor PP"],r["Total Qty"]),axis=1).round(1)
        vp=vp[["Category","Item","Vendor PP","Ordered Qty"]]
        return vp,vmg,nvmg
    elif method=="3group":
        fmg=_3group_vendor_mg(results,nv_cat,logic)
        vmg=mg5(fmg)
        vp=df.copy()
        vp["Ordered Qty"]=vp.apply(lambda r:max(r["Vendor PP"]*vmg,r["Total Qty"]),axis=1).round(1)
        vp=vp[["Category","Item","Vendor PP","Ordered Qty"]]
        return vp,vmg,0
    elif method=="day_based":
        raw=_day_vendor_mg(cmg,weekday,logic); vmg=mg5(raw)
        # For day_based, Vendor PP = total_qty / vendor_mg
        vp=df.copy()
        vp["Vendor PP"]=vp["Total Qty"].apply(lambda tq: r005(tq/vmg) if vmg>0 else 0)
        vp["Ordered Qty"]=(vp["Vendor PP"]*vmg).round(1)
        vp["Ordered Qty"]=vp.apply(lambda r:max(r["Ordered Qty"],r["Total Qty"]),axis=1)
        vp=vp[["Category","Item","Vendor PP","Ordered Qty"]]
        return vp,vmg,0
    return df,0,0

# ── aggressive plan builder ──
def aggressive_plan(vplan, results, final_vmg, avg_nv, is_nv, nv_cat, logic, method_groups=2):
    eff=set(logic.star_categories)
    if nv_cat: eff.add(norm(nv_cat))
    ag=vplan.copy()
    ag["Ordered Qty"]=ag.apply(lambda r:logic.aggressive_bump(r["Ordered Qty"]) if norm(r["Category"]) in eff else r["Ordered Qty"],axis=1).round(1)
    ag["AMG"]=ag["Ordered Qty"]/ag["Vendor PP"]

    if nv_cat:
        nva=ag[ag["Category"].notna()&ag["Category"].str.strip().str.lower().eq(norm(nv_cat))]
    else:
        nva=pd.DataFrame(columns=ag.columns)
    sta=ag[ag["Category"].str.lower().str.strip().isin(logic.star_categories)]
    rsa=ag[~ag.index.isin(nva.index)&~ag.index.isin(sta.index)]
    def _m(d): return d["AMG"].mean() if not d.empty else 0

    if method_groups==2:
        iam=(_m(sta)+_m(rsa))/2
        nv_diff=_m(nva)-avg_nv; adj_nv=avg_nv-nv_diff
        veg_diff=iam-final_vmg; adj_t=final_vmg-veg_diff
    else:
        iam=(_m(nva)+_m(sta)+_m(rsa))/3
        diff=iam-final_vmg; adj_t=final_vmg-diff; adj_nv=0

    adj_t=mg5(adj_t); adj_nv=mg5(adj_nv); adj_v=adj_t-adj_nv

    def _rb(r):
        c=norm(r["Category"])
        if is_nv and nv_cat and c==norm(nv_cat): base=adj_nv
        else: base=adj_t
        return r005(r["Ordered Qty"]/base) if base else 0
    ag["Vendor PP"]=ag.apply(_rb,axis=1)
    ag=ag[["Category","Item","Vendor PP","Ordered Qty"]]
    return ag,adj_t,adj_nv,adj_v

# ── special day ──
def special_day_mg(cmg,dt,ht,wd,logic):
    p=logic.get_reduction_pct(dt,ht,wd)
    return cmg*(100-p)/100.0, p
