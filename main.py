import numpy as np
from scipy.linalg import solve
from components.turbine import expand
from components.pump import compress
from properties.steam import get_state
from components.heater import get_fwh_coefficients
from iapws import IAPWS97

# Computes exergy array 
def compute_exergy(h, s, h0, s0, T0):
    h = np.array(h)
    s = np.array(s)
    return (h - h0) - (T0 * (s - s0))

# 1.1 Dead state
T0 = 298.15  
P0 = 0.101325  

dead_state = IAPWS97(T=T0, P=P0)

h0 = dead_state.h
s0 = dead_state.s


# 1. Parameters
P1 = 6; T1 = 563.15; m1 = 1; P2 = 1.8; eff_T = 0.88
T3 = 535.15; P4 = 0.008; eff_P = 0.85; TTD = 5; 
P12 = 0.2; P11 = 0.8; P18 = 3; P_prim = 15.5; T_prim_in = 594.26
T_prim_out = 560.93; m_prim = 1 

# 2. Calculate state

st1 = get_state("State 1", P=P1, T=T1) # Boiler outlet

st18 = expand(st1, P18, eff_T, name="State 18") # High pressure turbine extraction 3

st2 = expand(st1, P2, eff_T, name="State 2") # High pressure turbine outlet

st3 = get_state("state 3", P=P2, T=T3) # Reheater outlet

st11 = expand(st3, P11, eff_T, name="State 11") # Low pressure turbine extraction 2

st12 = expand(st3, P12, eff_T, name="State 12") # Low pressure turbine extraction 1

st4 = expand(st3, P4, eff_T, name="State 4") # Low pressure turbine outlet

st5 = get_state("state 5", P=P4, x=0) # Condenser outlet

st6 = compress(st5, P1, eff_P, name="State 6") # Pump outlet

st13 = get_state("state 13", P=P12, x=0) # Drain FWH 1 

st15 = compress(st13, P1, eff_P, name="State 15") # Drain FWH 1 compressed 

st7 = get_state("state 7", P=P1, T=st13.T-TTD) # FWH 1 outlet

st16 = get_state("state 16", P=P11, x=0) # Drain FWH 2 

st17 = compress(st16, P1, eff_P, name="State 17") # Drain FWH 2 compressed

st8 = get_state("state 8", P=st7.P, T=st16.T-TTD) # FWH 2 outlet

st20 = get_state("state 20", P=P18, x=0) # Drain FWH 3

st21 = compress(st20, P1, eff_P, name="State 21") # Drain FWH 3 compressed

st9 = get_state("state 9", P=P1, T=st20.T-TTD) # FWH3 outlet

# 3. Build and solve matrix
h1_hot, h1_cold = get_fwh_coefficients(st12.h, st13.h, st6.h, st7.h) # FWH 1 (Low Pressure)
h2_hot, h2_cold = get_fwh_coefficients(st11.h, st16.h, st7.h, st8.h) # FWH 2 (Intermediate Pressure)
h15_drop = st15.h - st16.h # drain from Heater 1

h3_hot, h3_cold = get_fwh_coefficients(st18.h, st20.h, st8.h, st9.h) # FWH 3 (High Pressure)
h17_drop = st17.h - st20.h # drain from Heater 2

row1 = [0, 0, h1_hot, h1_cold] # Row 1 (FWH 1): m12*h1_hot + m4*h1_cold = 0
row2 = [0, h2_hot, h15_drop, h2_cold] # Row 2 (FWH 2): m11*h2_hot + m12*h15_drop + m4*h2_cold = 0
row3 = [h3_hot, h17_drop, h17_drop, h3_cold] # Row 3 (FWH 3): m18*h3_hot + (m11+m12)*h17_drop + m4*h3_cold = 0
row4 = [1, 1, 1, 1] # Row 4 (Mass Balance): Sum = 1


A = np.array([row1, row2, row3, row4])
B = np.array([0, 0, 0, 1])

masses = solve(A, B) 
m18, m11, m12, m4 = masses

# 4.Calculate remaining state

h14 = (m11*st11.h + m12*st15.h)/(m11 + m12) # Mix of compressed drain FWH 1 and extraction 2
st14 = get_state("state 14", P=P11, h=h14) 

h19 = (m18*st18.h + (m11 + m12)*st17.h)/(m18 + m11 + m12) # Mix of compressed drain FWH 2 and extraction 3
st19 = get_state("state 19", P=P18, h=h19)

h10 = m4*st9.h + (m18 + m11 + m12)*st21.h # Mix of compressed drain FWH 3 and main flow
st10 = get_state("state 10", P=P1, h=h10)

# Primary Side States
st_prim_in = IAPWS97(P=P_prim, T=T_prim_in)
st_prim_out = IAPWS97(P=P_prim, T=T_prim_out)

# 5. Overall First law performance
Q_out = (m4*(st4.h - st5.h))
Q_in = (m1*(st1.h - st10.h)) + ((m1 - m18)*(st3.h - st2.h))
W_HT = (m1*(st1.h - st18.h)) + ((m1 - m18)*(st18.h - st2.h))
W_LT = (m4*(st3.h - st4.h)) + (m11*(st3.h - st11.h)) + (m12*(st3.h - st12.h))
W_P1 = (m4*(st6.h - st5.h))
W_P2 = (m12*(st15.h - st13.h))
W_P3 = (m11*(st17.h - st16.h))
W_P4 = (m18*(st21.h - st20.h))
W_P = W_P1 + W_P2 + W_P3 + W_P4

W_net = W_HT + W_LT - W_P
eff = W_net / Q_in

print(f"Efficiency: {eff * 100} %")
# print("Work :", W_net, "kW")
print((Q_in-Q_out-W_net)/ Q_in)


# 6. Exergy analysis
states_list = [st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13, st14, st15, st16, st17, st18, st19, st20, st21]

h = np.array([st.h for st in states_list])
s = np.array([st.s for st in states_list])

e = compute_exergy(h, s, h0, s0, T0)

for i, st in enumerate(states_list):
    st.e = e[i]
    # print(f'state {i+1} : {st.e}')

# Calculates Exergy for Primary states
e_prim = compute_exergy(
    [st_prim_in.h, st_prim_out.h], 
    [st_prim_in.s, st_prim_out.s], 
    h0, s0, T0
)
e_prim_in = e_prim[0]
e_prim_out = e_prim[1]

# 7. Component-wise Exergy balance  
ed_ht = (m1 * st1.e) - (m18 * st18.e + (m1 - m18) * st2.e) - W_HT
ed_lt = ((m4 + m11 + m12) * st3.e) - (m11 * st11.e + m12 * st12.e + m4 * st4.e) - W_LT
ed_c = m4 * (st4.e - st5.e)
ed_P1 = W_P1 - m4 * (st6.e - st5.e)
ed_fwh1 = (m12 * st12.e + m4 * st6.e) - (m12 * st13.e + m4 * st7.e)
ed_P2 = W_P2 - m12 * (st15.e - st13.e)
ed_fwh2 = (m11 * st11.e + m4 * st7.e) - (m11 * st16.e + m4 * st8.e)
ed_P3 = W_P3 - m11 * (st17.e - st16.e)
ed_fwh3 = (m18 * st18.e + m4 * st8.e) - (m18 * st20.e + m4 * st9.e)
ed_P4 = W_P4 - m18 * (st21.e - st20.e)

# Calculate required primary mass flow to satisfy the energy balance
Q_primary_specific = st_prim_in.h - st_prim_out.h
m_prim = Q_in / Q_primary_specific

Ex_supplied = m_prim * (e_prim_in - e_prim_out) # Exergy Rate Supplied (Primary side)
Ex_recovered = m1 * (st1.e - st10.e) + (m1 - m18) * (st3.e - st2.e) # Exergy Rate Recovered (Secondary side)
ed_sg = Ex_supplied - Ex_recovered # Exergy Destruction in the Steam Generator

# print(f"Exergy Destruction HP Turbine: {ed_ht} kW")
# print(f"Exergy Destruction LP Turbine: {ed_lt} kW")
# print(f"Exergy Destruction Condenser: {ed_c} kW")
# print(f"Exergy Destruction Pump 1: {ed_P1} kW")
# print(f"Exergy Destruction FWH 1: {ed_fwh1} kW")
# print(f"Exergy Destruction Pump 2: {ed_P2} kW")
# print(f"Exergy Destruction FWH 2: {ed_fwh2} kW")
# print(f"Exergy Destruction Pump 3: {ed_P3} kW")
# print(f"Exergy Destruction FWH 3: {ed_fwh3} kW")
# print(f"Exergy Destruction Pump 4: {ed_P4} kW")
# print(f"Exergy Destruction in Steam Generator: {ed_sg} kW")

# Overall Second Law Efficiency
Ex_total = ed_ht + ed_lt + ed_c + ed_P1 + ed_fwh1 + ed_P2 + ed_fwh2 + ed_P3 + ed_fwh3 + ed_P4 + ed_sg
eff_ii = W_net / Ex_supplied

print(f"Total Exergy Destruction: {Ex_total} kW")
print(f"Exergetic Efficiency: {eff_ii} %")