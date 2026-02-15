"""
Client Logic — every client's unique configuration in one scalable file.

ARCHITECTURE:
  BaseLogic          — abstract contract
  StandardLogic      — default (Tekion-like) with all features
  Per-client classes  — override only what differs
  LOGIC_REGISTRY      — maps client key → logic class

TO ADD A CLIENT WITH SAME LOGIC:  add 1 line to LOGIC_REGISTRY
TO ADD A CLIENT WITH DIFFERENT LOGIC:  subclass, override, add to LOGIC_REGISTRY
"""
import math
from abc import ABC, abstractmethod
from ml_core import norm

# ═══════════════════════════════════════════════════════════════════════
# Base contract
# ═══════════════════════════════════════════════════════════════════════

class BaseLogic(ABC):
    @property
    @abstractmethod
    def encoder_columns(self): ...        # which columns the model trains on
    @property
    @abstractmethod
    def fixed_categories(self): ...       # UI category list
    @property
    @abstractmethod
    def star_categories(self): ...        # priority categories
    @property
    def has_nonveg_toggle(self): return True
    @property
    def nonveg_item_count(self): return 1      # how many nonveg items to enter
    @property
    def has_vendor_plan(self): return True
    @property
    def has_aggressive_plan(self): return True
    @property
    def has_special_day(self): return True
    @property
    def has_embeddings(self): return True
    @property
    def salad_per_pax(self): return 0.06
    @property
    def default_mg(self): return 300
    @property
    def default_bump(self): return 0.01
    @property
    def biryani_bump(self): return 0.035
    @property
    def vendor_floor_ratio(self): return 0.825
    # vendor MG method: "tekion_2group", "3group", "day_based", None
    @property
    def vendor_mg_method(self): return "tekion_2group"
    @property
    def has_separate_nonveg_mg(self): return True  # Tekion splits veg/nv MG
    @property
    def fixed_pp_map(self): return None   # if set, client plan uses these fixed PP values
    @property
    def category_display_map(self): return {}  # e.g. {"North Veg dry":"Veg dry"}

    # ── canon maps (override per client) ──
    @property
    def category_map(self): return {}
    @property
    def day_type_map(self): return {"regular":"regular","previous day of holiday":"previous day of holiday","next day of holiday":"next day of holiday","holiday":"holiday"}
    @property
    def holiday_type_map(self): return {"n/a":"not applicable","na":"not applicable","not applicable":"not applicable","non-important holiday":"non-important holiday","important holiday":"important holiday","compulsory holiday":"compulsory holiday"}
    @property
    def meal_day_map(self): return {"veg":"veg","nonveg":"nonveg","non-veg":"nonveg","non veg":"nonveg"}

    def canonicalize_category(self,s): n=norm(s); return self.category_map.get(n,n)
    def canonicalize_day_type(self,s): n=norm(s); return self.day_type_map.get(n,n)
    def canonicalize_holiday_type(self,s): n=norm(s); return self.holiday_type_map.get(n,n)
    def canonicalize_meal_day(self,s): n=norm(s); return self.meal_day_map.get(n,n)

    # ── tier / vendor MG adjustment ──
    def adjust_vendor_mg(self, mg): return mg
    def adjust_nonveg_vendor_mg(self, mg): return mg

    # ── aggressive bump ──
    def aggressive_bump(self, w):
        sp=0.10*w
        if sp<=2:f=1.0
        elif sp<=4:f=0.35
        elif sp<=6:f=0.25
        elif sp<=8:f=0.15
        else:f=0.10
        return round(w+f*sp,1)

    # ── special day reduction ──
    _REDUCTION = {
        ("holiday","important holiday","monday"):12,("holiday","important holiday","tuesday"):10,("holiday","important holiday","wednesday"):10,("holiday","important holiday","thursday"):10,("holiday","important holiday","friday"):12,
        ("holiday","non-important holiday","monday"):13,("holiday","non-important holiday","tuesday"):10,("holiday","non-important holiday","wednesday"):10,("holiday","non-important holiday","thursday"):9,("holiday","non-important holiday","friday"):8,
        ("next day of holiday","compulsory holiday","monday"):10,("next day of holiday","compulsory holiday","tuesday"):8,("next day of holiday","compulsory holiday","wednesday"):9,("next day of holiday","compulsory holiday","thursday"):10,("next day of holiday","compulsory holiday","friday"):11,
        ("next day of holiday","important holiday","monday"):9,("next day of holiday","important holiday","tuesday"):8,("next day of holiday","important holiday","wednesday"):7,("next day of holiday","important holiday","thursday"):8,("next day of holiday","important holiday","friday"):9,
        ("next day of holiday","non-important holiday","monday"):9,("next day of holiday","non-important holiday","tuesday"):9,("next day of holiday","non-important holiday","wednesday"):8,("next day of holiday","non-important holiday","thursday"):8,("next day of holiday","non-important holiday","friday"):9,
        ("previous day of holiday","compulsory holiday","monday"):10,("previous day of holiday","compulsory holiday","tuesday"):9,("previous day of holiday","compulsory holiday","wednesday"):9,("previous day of holiday","compulsory holiday","thursday"):8,("previous day of holiday","compulsory holiday","friday"):10,
        ("previous day of holiday","important holiday","monday"):10,("previous day of holiday","important holiday","tuesday"):6,("previous day of holiday","important holiday","wednesday"):6,("previous day of holiday","important holiday","thursday"):7,("previous day of holiday","important holiday","friday"):9,
        ("previous day of holiday","non-important holiday","monday"):9,("previous day of holiday","non-important holiday","tuesday"):8,("previous day of holiday","non-important holiday","wednesday"):11,("previous day of holiday","non-important holiday","thursday"):10,("previous day of holiday","non-important holiday","friday"):10,
    }
    def get_reduction_pct(self,dt,ht,wd):
        return self._REDUCTION.get((dt.strip().lower(),ht.strip().lower(),wd.strip().lower()),10)

    # ── multiplier-only (Toasttab) ──
    def get_multiplier_pp(self, cat): raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# TEKION
# ═══════════════════════════════════════════════════════════════════════
class TekionLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","day_type","holiday_type","meal_day"]
    @property
    def fixed_categories(self): return ["Flavour Rice","Indian Bread","White Rice","Veg Dry","Veg Curry","Dal","Sambar","Rasam","Salad"]
    @property
    def star_categories(self): return {"flavoured rice","veg curry"}
    @property
    def category_map(self): return {"indian bread":"indian bread","indian breads":"indian bread","veg dry":"veg dry","veg curry":"veg curry","gravy veg":"veg curry","flavour rice":"flavoured rice","flavoured rice":"flavoured rice","white rice":"white rice","steamed rice":"white rice","dal":"dal","sambar":"sambar","rasam":"rasam","salad":"salad","non veg biryani":"non veg biryani","non veg curry":"non veg curry"}
    def adjust_vendor_mg(self,mg):
        if mg<350: return mg*0.95
        elif 350<=mg<=700: return mg-15
        else: return mg-25
    def adjust_nonveg_vendor_mg(self,mg):
        if mg<400: return mg*0.97
        else: return mg-20

# ═══════════════════════════════════════════════════════════════════════
# CLARIO — 5-feature model, client plan only, has PP cap
# ═══════════════════════════════════════════════════════════════════════
class ClarioLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","meal_day"]
    @property
    def fixed_categories(self): return ["Flavoured Rice","Indian Bread","White Rice","Healty Rice","Veg Dry","Veg Gravy","Dal","Sambar/Rasam"]
    @property
    def star_categories(self): return {"flavoured rice","veg gravy"}
    @property
    def category_map(self): return {"indian bread":"indian bread","indian breads":"indian bread","veg dry":"veg dry","veg curry":"veg gravy","gravy veg":"veg gravy","veg gravy":"veg gravy","healty rice":"healty rice","flavour rice":"flavoured rice","flavoured rice":"flavoured rice","white rice":"white rice","steamed rice":"white rice","dal":"dal","sambar":"sambar/rasam","sambar/rasam":"sambar/rasam","rasam":"sambar/rasam"}
    @property
    def has_vendor_plan(self): return False
    @property
    def has_aggressive_plan(self): return False
    @property
    def has_special_day(self): return False
    @property
    def vendor_mg_method(self): return None

# ═══════════════════════════════════════════════════════════════════════
# ODESSIA — 6-feature, 3-group vendor MG, fixed PP client plan
# ═══════════════════════════════════════════════════════════════════════
class OdessiaLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","day_type","holiday_type"]
    @property
    def fixed_categories(self): return ["Dal/Sambar","Flavoured rice","Indian Bread","Rasam","Veg dry","Veg Gravy","White Rice"]
    @property
    def star_categories(self): return {"flavoured rice","veg curry"}
    @property
    def category_map(self): return {"indian breads":"indian bread","indian bread":"indian bread","veg dry":"veg dry","veg gravy":"veg curry","veg curry":"veg curry","flavoured rice":"flavoured rice","flavour rice":"flavoured rice","white rice":"white rice","steamed rice":"white rice","dal":"dal","sambar":"sambar","rasam":"rasam","dal/sambar":"dal/sambar"}
    @property
    def vendor_mg_method(self): return "3group"
    @property
    def has_separate_nonveg_mg(self): return False
    @property
    def has_nonveg_toggle(self): return False  # always shows nonveg section
    @property
    def vendor_floor_ratio(self): return 0.0   # no floor check
    @property
    def fixed_pp_map(self):
        return {"dal/sambar":0.066,"flavoured rice":0.102,"indian bread":0.05,"rasam":0.033,"veg dry":0.075,"veg curry":0.102,"white rice":0.088,"non veg gravy":0.122,"non veg biryani":0.22,"non veg dry":0.099}

    def adjust_vendor_mg(self,mg):
        if mg<120: return mg*0.94
        elif 121<=mg<=230: return mg-5
        elif 231<=mg<=400: return mg-10
        else: return mg-15

# ═══════════════════════════════════════════════════════════════════════
# RIPPLING — 6-feature, 3-group, North/South veg dry
# ═══════════════════════════════════════════════════════════════════════
class RipplingLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","day_type","holiday_type"]
    @property
    def fixed_categories(self): return ["Flavoured rice","White Rice","North Veg dry","South Veg dry","Veg gravy","Dal","Sambar/Rasam"]
    @property
    def star_categories(self): return {"flavoured rice","veg gravy"}
    @property
    def category_display_map(self): return {"North Veg dry":"Veg dry","South Veg dry":"Veg dry"}
    @property
    def vendor_mg_method(self): return "3group"
    @property
    def has_separate_nonveg_mg(self): return False
    @property
    def has_nonveg_toggle(self): return False
    @property
    def vendor_floor_ratio(self): return 0.0

    def adjust_vendor_mg(self,mg):
        if mg<120: return mg*0.94
        else: return mg-10

# ═══════════════════════════════════════════════════════════════════════
# STRIPE — 6-feature, day-based vendor MG, 2 nonveg items
# ═══════════════════════════════════════════════════════════════════════
class StripeLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category","day_type","holiday_type"]
    @property
    def fixed_categories(self): return ["indian bread","veg dry","veg curry","flavoured rice","white rice","dal","sambar","rasam"]
    @property
    def star_categories(self): return {"flavoured rice","veg curry"}
    @property
    def category_map(self): return {"indian breads":"indian bread","indian bread":"indian bread","veg dry":"veg dry","gravy veg":"veg curry","veg curry":"veg curry","flavoured rice":"flavoured rice","steamed rice":"white rice","white rice":"white rice","dal":"dal","sambar":"sambar","rasam":"rasam"}
    @property
    def vendor_mg_method(self): return "day_based"
    @property
    def nonveg_item_count(self): return 2
    @property
    def has_nonveg_toggle(self): return False
    @property
    def has_separate_nonveg_mg(self): return False
    @property
    def vendor_floor_ratio(self): return 0.0
    @property
    def day_reductions(self): return {"monday":0.23,"tuesday":0.21,"wednesday":0.21,"thursday":0.22,"friday":0.23}

# ═══════════════════════════════════════════════════════════════════════
# TESSOLVE — 4-feature model, veg only, client plan only
# ═══════════════════════════════════════════════════════════════════════
class TessolveLogic(BaseLogic):
    @property
    def encoder_columns(self): return ["weekday","menu_items","sub_category","category"]
    @property
    def fixed_categories(self): return ["Healthy rice","Indian Bread","Flavoured Rice","Veg gravy","Veg Dry","Dal/Sambar","white rice","Rasam"]
    @property
    def star_categories(self): return {"flavoured rice","veg gravy"}
    @property
    def has_nonveg_toggle(self): return False
    @property
    def has_vendor_plan(self): return False
    @property
    def has_aggressive_plan(self): return False
    @property
    def has_special_day(self): return False
    @property
    def vendor_mg_method(self): return None

# ═══════════════════════════════════════════════════════════════════════
# TOASTTAB — multiplier only, no ML
# ═══════════════════════════════════════════════════════════════════════
class ToasttabLogic(BaseLogic):
    @property
    def has_embeddings(self): return False
    @property
    def encoder_columns(self): return []
    @property
    def fixed_categories(self): return []
    @property
    def star_categories(self): return set()
    @property
    def has_nonveg_toggle(self): return False
    @property
    def has_vendor_plan(self): return False
    @property
    def has_aggressive_plan(self): return False
    @property
    def has_special_day(self): return False
    @property
    def vendor_mg_method(self): return "toasttab_formula"
    @property
    def default_mg(self): return 100

    def toasttab_adjust(self, c):
        if c>=135: return c-15
        elif c>=95: return c-10
        elif c<55: return c
        else: return c-5

# ═══════════════════════════════════════════════════════════════════════
# REGISTRY — map client keys to logic classes
# Same logic? Just point to the same class.
# ═══════════════════════════════════════════════════════════════════════
LOGIC_REGISTRY = {
    "tekion":    TekionLogic,
    "clario":    ClarioLogic,
    "odessia":   OdessiaLogic,
    "rippling":  RipplingLogic,
    "stripe":    StripeLogic,
    "tessolve":  TessolveLogic,
    "toasttab":  ToasttabLogic,
}

def get_logic(ck):
    cls=LOGIC_REGISTRY.get(ck.strip().lower())
    if not cls: raise ValueError(f"No logic for '{ck}'. Available: {sorted(LOGIC_REGISTRY)}")
    return cls()
