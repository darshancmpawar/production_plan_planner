# main.py
# Vendor MG calculator using only the adjustment rule:
# IF(C >= 95, C - 10, IF(C < 55, C, C - 5))

from datetime import date as _date, datetime
import streamlit as st

# Try to default "today" to Asia/Kolkata; fall back to local if not available
try:
    from zoneinfo import ZoneInfo  
    _today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
except Exception:
    _today_ist = _date.today()

st.set_page_config(page_title="Vendor MG Calculator", layout="centered")

# ---------- Helpers ----------
def mg_rounder(x: float) -> int:
    """Round to nearest multiple of 5."""
    return int(round(float(x) / 5.0)) * 5

def adjust_client_mg(c: float) -> float:
    """Apply adjustment: IF(C>=95, C-10, IF(C<55, C, C-5))."""
    if c >= 135:
        return c - 15
    elif c >= 95:
        return c - 10
    elif c < 55:
        return c
    else:
        return c - 5

# ---------- UI ----------
st.title("Vendor MG Calculator")
st.caption("Inputs: **Date** + **Client MG** → Output: **Vendor MG** using historical pattern adjustment rule.")

col_a, col_b = st.columns(2)
with col_a:
    sel_date = st.date_input("Select date", value=_today_ist, format="DD/MM/YYYY")
with col_b:
    client_mg = st.number_input("Client MG", min_value=1, step=1, value=100)

round_to_5 = st.checkbox("Round Vendor MG to nearest 5", value=True)

# Compute
adjusted = adjust_client_mg(client_mg)
vendor_mg = mg_rounder(adjusted) if round_to_5 else int(adjusted)

# ---------- Output ----------
st.divider()
st.subheader(f"Vendor MG for {sel_date:%A, %d %b %Y}")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Client MG", f"{client_mg}")
with c2:
    st.metric("Adjusted (pre-round)", f"{adjusted:.0f}")
with c3:
    st.metric("Vendor MG", f"{vendor_mg}")

# Working / explanation
#st.write("#### Working")
#st.write(
 #   f"- **Rule**: IF(C ≥ 95, C − 10; IF(C < 55, C; C − 5))"
#)
#st.write(
#    f"- **Adjusted**: {adjusted:.0f}"
#    + (" → **rounded to nearest 5**" if round_to_5 else " (no rounding)")
#    + f" → **Vendor MG = {vendor_mg}**"
#)

# Optional CSV download
csv = "date,client_mg,adjusted,vendor_mg,rounded_to_5\n"
csv += f"{sel_date},{client_mg},{int(adjusted)},{vendor_mg},{round_to_5}\n"
#st.download_button("Download result (CSV)", data=csv, file_name="vendor_mg_result.csv", mime="text/csv")
