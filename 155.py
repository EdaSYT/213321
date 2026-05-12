import streamlit as st
import pandas as pd

# Kütüphane kontrolü
try:
    from gurobipy import Model, GRB, quicksum
    GUROBI_AVAILABLE = True
except ModuleNotFoundError:
    GUROBI_AVAILABLE = False

# Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Hattı Dengeleme Sistemi")
st.title("🏭 Montaj Hattı Dengeleme & Operatör Atama Sistemi")

if not GUROBI_AVAILABLE:
    st.error("❌ 'gurobipy' kütüphanesi bulunamadı! Lütfen terminale 'pip install gurobipy' yazarak yükleyin veya requirements.txt dosyanıza ekleyin.")
    st.stop()

# =========================================================
# VERİ TANIMLARI
# =========================================================
I = range(1, 64)
J = range(1, 37)
W = range(1, 37)

t = {
    1: 2.43,  2: 9.79,  3: 2.12,  4: 9.92,  5: 4.66,  6: 11.58, 7: 1.01,  8: 1.44,  9: 9.66, 10: 10.30, 
    11: 0.49, 12: 7.13, 13: 7.18, 14: 2.44, 15: 3.58, 16: 4.90, 17: 3.21, 18: 7.78, 19: 11.27, 20: 11.35, 
    21: 0.80, 22: 3.31, 23: 9.83, 24: 0.80, 25: 4.61, 26: 5.20, 27: 11.89, 28: 6.30, 29: 13.32, 30: 0.98,
    31: 14.20, 32: 6.13, 33: 0.98, 34: 14.49, 35: 3.14, 36: 12.12, 37: 1.07, 38: 5.14, 39: 5.63, 40: 0.57, 
    41: 10.13, 42: 0.90, 43: 1.39, 44: 1.43, 45: 0.51, 46: 10.74, 47: 5.65, 48: 7.38, 49: 1.71, 50: 15.09, 
    51: 7.31, 52: 6.93, 53: 10.72, 54: 1.31, 55: 6.45, 56: 2.39, 57: 0.89, 58: 11.06, 59: 8.02, 60: 6.48,
    61: 3.13, 62: 0.53, 63: 7.74
}

P = [(i, i + 1) for i in range(1, 63)]
d_dist = {j: {k: 2 * abs(j - k) for k in range(1, 38)} for j in range(1, 38)}
BIG_M = sum(t.values())

# =========================================================
# YAN MENÜ (SIDEBAR)
# =========================================================
with st.sidebar:
    st.header("⚙️ Ayarlar")
    L = st.number_input("Maksimum Yürüme Mesafesi (L)", value=4)
    D_target = st.number_input("Hedef Üretim Miktarı (D)", value=32)
    T_shift = st.number_input("Vardiya Süresi (T - dk)", value=510)
    U_MAX = st.slider("Maks. Operatör Doluluğu (U_MAX)", 0.1, 1.0, 1.0, step=0.05)
    st.markdown("---")
    target_workers = st.slider("Detaylı Rapor İçin Operatör Seç", 1, 36, 29)

# =========================================================
# ÇÖZÜCÜ FONKSİYON
# =========================================================
@st.cache_data(show_spinner=False)
def solve_gurobi_model(exact_workers, L_val, D_val, T_val, U_limit):
    m = Model("line_balancing")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 5)

    x = m.addVars(I, J, vtype=GRB.BINARY, name="x")
    y = m.addVars(W, J, vtype=GRB.BINARY, name="y")
    z = m.addVars(W, vtype=GRB.BINARY, name="z")
    l = m.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="l")
    q = m.addVars(W, J, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    C = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C")

    # Temel Kısıtlar
    for i in I: m.addConstr(quicksum(x[i, j] for j in J) == 1)
    for i, h in P: m.addConstr(quicksum(j * x[i, j] for j in J) <= quicksum(j * x[h, j] for j in J))
    for j in J: m.addConstr(l[j] == quicksum(t[i] * x[i, j] for i in I))
    for j in J: m.addConstr(quicksum(y[w, j] for w in W) == 1)
    for w in W:
        for j in J: m.addConstr(y[w, j] <= z[w])
    
    # Doğrusallaştırma ve Çevrim Süresi
    for w in W:
        for j in J:
            m.addConstr(q[w, j] <= l[j])
            m.addConstr(q[w, j] <= BIG_M * y[w, j])
            m.addConstr(q[w, j] >= l[j] - BIG_M * (1 - y[w, j]))
        m.addConstr(quicksum(q[w, j] for j in J) <= C)
    
    for j in J: m.addConstr(l[j] <= C)
    
    # Doluluk ve Mesafe
    for w in W: m.addConstr((D_val / T_val) * quicksum(q[w, j] for j in J) <= U_limit)
    for w in W:
        for j in J:
            for k in J:
                if j < k and d_dist[j][k] > L_val: m.addConstr(y[w, j] + y[w, k] <= 1)

    m.addConstr(quicksum(z[w] for w in W) == exact_workers)
    m.setObjective(C, GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL or m.status == GRB.FEASIBLE:
        return {
            "C": C.X,
            "ops": {j: [i for i in I if x[i, j].X > 0.5] for j in J},
            "w_st": {w: [j for j in J if y[w, j].X > 0.5] for w in W},
            "l_val": {j: l[j].X for j in J},
            "util": {w: 100 * (D_val / T_val) * sum(q[w, j].X for j in J) for w in W}
        }
    return None

# =========================================================
# ARAYÜZ ÇIKTILARI
# =========================================================
if st.button("🚀 Senaryoları Hesapla"):
    summary = []
    all_res = {}
    
    for n in range(1, 37):
        res = solve_gurobi_model(n, L, D_target, T_shift, U_MAX)
        if res:
            cap = T_shift / res['C']
            summary.append([n, f"{res['C']:.2f} dk", f"{cap:.2f} Adet", "✅" if cap >= D_target - 0.01 else "⚠️"])
            all_res[n] = res
        else:
            summary.append([n, "-", "-", "❌"])

    st.table(pd.DataFrame(summary, columns=["İşçi", "Çevrim S.", "Kapasite", "Durum"]))

    if target_workers in all_res:
        st.divider()
        res_t = all_res[target_workers]
        c1, c2 = st.columns(2)
        with c1:
            st.write("📋 **İstasyon Detayları**")
            st.table(pd.DataFrame([[j, str(res_t['ops'][j]), f"{res_t['l_val'][j]:.2f}"] for j in J if res_t['l_val'][j]>0], columns=["İst", "Op", "Yük"]))
        with c2:
            st.write("👷 **Operatör Dolulukları**")
            st.table(pd.DataFrame([[w, str(res_t['w_st'][w]), f"%{res_t['util'][w]:.2f}"] for w in W if res_t['w_st'][w]], columns=["Op", "İstasyonlar", "Doluluk"]))
