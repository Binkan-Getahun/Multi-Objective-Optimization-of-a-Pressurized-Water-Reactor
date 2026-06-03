import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from iapws import IAPWS97

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from components.turbine import expand
from components.pump import compress
from components.heater import get_fwh_coefficients
from properties.steam import get_state




# 1. Wrapper Function 
def compute_exergy(h, s, h0, s0, T0):
    """Computes Exergy."""

    h = np.array(h)     # Vecttorization
    s = np.array(s)

    return (h - h0) - (T0 * (s - s0))


def simulation_engine(p_array):
    """Runs the steam cycle simulation for a given set of 4 pressures."""
    
    # Constant Parameter
    P1 = 6; T1 = 563.15; m1 = 1; eff_T = 0.88
    T3 = 535.15; P4 = 0.008; eff_P = 0.85; TTD = 5 
    P_prim = 15.5; T_prim_in = 594.26; T_prim_out = 560.93

    # Dead state
    T0 = 298.15; P0 = 0.101325  
    dead_state = IAPWS97(T=T0, P=P0)
    h0, s0 = dead_state.h, dead_state.s

    # Map the sorted variables to specific plant architecture
    P18, P2, P11, P12 = p_array 

    try:
        # Computes state calculations
        st1 = get_state("State 1", P=P1, T=T1)
        st18 = expand(st1, P18, eff_T, name="State 18")
        st2 = expand(st1, P2, eff_T, name="State 2")
        st3 = get_state("state 3", P=P2, T=T3)
        st11 = expand(st3, P11, eff_T, name="State 11")
        st12 = expand(st3, P12, eff_T, name="State 12")
        st4 = expand(st3, P4, eff_T, name="State 4")
        st5 = get_state("state 5", P=P4, x=0)
        st6 = compress(st5, P1, eff_P, name="State 6")
        
        st13 = get_state("state 13", P=P12, x=0)
        st15 = compress(st13, P1, eff_P, name="State 15")
        st7 = get_state("state 7", P=P1, T=st13.T-TTD)
        
        st16 = get_state("state 16", P=P11, x=0)
        st17 = compress(st16, P1, eff_P, name="State 17")
        st8 = get_state("state 8", P=st7.P, T=st16.T-TTD)
        
        st20 = get_state("state 20", P=P18, x=0)
        st21 = compress(st20, P1, eff_P, name="State 21")
        st9 = get_state("state 9", P=P1, T=st20.T-TTD)

        # Build and solve matrix
        h1_ext, h1_fwh = get_fwh_coefficients(st12.h, st13.h, st6.h, st7.h)
        h2_ext, h2_fwh = get_fwh_coefficients(st11.h, st16.h, st7.h, st8.h)
        h15_drop = st15.h - st16.h
        h3_ext, h3_fwh = get_fwh_coefficients(st18.h, st20.h, st8.h, st9.h)
        h17_drop = st17.h - st20.h

        A = np.array([
            [0, 0, h1_ext, h1_fwh],
            [0, h2_ext, h15_drop, h2_fwh],
            [h3_ext, h17_drop, h17_drop, h3_fwh],
            [1, 1, 1, 1]
        ])
        B = np.array([0, 0, 0, 1])
        m18, m11, m12, m4 = solve(A, B)

        # Performance Calculation
        Q_in = (m1*(st1.h - st9.h)) + ((m1 - m18)*(st3.h - st2.h)) 
        W_HT = (m1*(st1.h - st18.h)) + ((m1 - m18)*(st18.h - st2.h))
        W_LT = (m4*(st3.h - st4.h)) + (m11*(st3.h - st11.h)) + (m12*(st3.h - st12.h))
        W_P = (m4*(st6.h - st5.h)) + (m12*(st15.h - st13.h)) + (m11*(st17.h - st16.h)) + (m18*(st21.h - st20.h))
        W_net = W_HT + W_LT - W_P
        
        # Exergy calculation
        st_prim_in = IAPWS97(P=P_prim, T=T_prim_in)
        st_prim_out = IAPWS97(P=P_prim, T=T_prim_out)
        e_prim = compute_exergy([st_prim_in.h, st_prim_out.h], [st_prim_in.s, st_prim_out.s], h0, s0, T0)
       
        # Calculate required primary mass flow to satisfy the energy balance
        Q_primary_specific = st_prim_in.h - st_prim_out.h
        m_prim = Q_in / Q_primary_specific
        Ex_supplied = m_prim * (e_prim[0] - e_prim[1])
        
        # First Law Eff and Total Exergy Destruction
        eff = W_net / Q_in
        total_exergy_destruction = Ex_supplied - W_net 

        # Quality and Total extraction check 
        quality_out = st4.x if st4.x is not None else 1.0
        total_extraction = m18 + m11 + m12

        return eff, total_exergy_destruction, quality_out, total_extraction

    except Exception as e:
        # If IAPWS97 fails, return severely penalized values

        return -1, 1e9, 0, 1


# 2. THE NSGA-II PROBLEM CLASS 
class RankineCycleOptimization(ElementwiseProblem):
    '''    Defines Custom Optimization Problem    '''
    
    def __init__(self):
        super().__init__(
            n_var=4,
            n_obj=2,
            n_ieq_constr=2, 
            xl=np.array([0.05, 0.05, 0.05, 0.05]), # Lower bound pressures (MPa)
            xu=np.array([5.9, 5.9, 5.9, 5.9])      # Upper bound pressures (MPa)
        )


    def _evaluate(self, x, out, *args, **kwargs):

        # 1. Enforce Hierarchy
        x_sorted = sorted(x, reverse=True) 

        eff, ex_dest, x_out, sum_y = simulation_engine(x_sorted)

        # 3. Objectives
        f1 = -eff      
        f2 = ex_dest   

        # 4. Constraints
        # TTD is omitted because it is inherently satisfied in the constant modeling parameter 
        g1 = 0.88 - x_out      # Moisture Limit: x_out >= 0.88
        g2 = sum_y - 0.35      # Extraction Limit: sum(y) <= 0.35

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


# 3. EXECUTION, MCDM, PLOT
if __name__ == "__main__":
    problem = RankineCycleOptimization()

    algorithm = NSGA2(
        pop_size=1000,
        n_offsprings=500,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    print("Running NSGA-II Optimization (This may take a few minutes)...")
    res = minimize(problem, algorithm, ('n_gen', 100), seed=1, verbose=True)

    # Extract Pareto Front Data
    pareto_front = res.F
    # Re-invert efficiency for the math
    pareto_front[:, 0] = -pareto_front[:, 0] 

    print("\nOptimization Complete. Applying Shannon Entropy and VIKOR...")


    # SHANNON ENTROPY 

    # Normalize data 
    P_matrix = np.zeros_like(pareto_front)
    P_matrix[:, 0] = pareto_front[:, 0] / np.sum(pareto_front[:, 0])
    
    # Inverse logic for Exergy then Normalize
    inv_exergy = 1.0 / pareto_front[:, 1]
    P_matrix[:, 1] = inv_exergy / np.sum(inv_exergy)

    # Normalize Entropy
    m = len(pareto_front)
    k = 1.0 / np.log(m)

    # Handle log(0) safely
    E_j = -k * np.sum(P_matrix * np.log(P_matrix + 1e-12), axis=0)
    D_j = 1 - E_j
    weights = D_j / np.sum(D_j)
    
    print(f"Calculated Weights: Efficiency = {weights[0]:.4f}, Exergy Destruction = {weights[1]:.4f}")


    # VIKOR METHOD
    f_star = np.array([np.max(pareto_front[:, 0]), np.min(pareto_front[:, 1])]) # Ideal
    f_minus = np.array([np.min(pareto_front[:, 0]), np.max(pareto_front[:, 1])]) # Worst

    S = np.zeros(m)
    R = np.zeros(m)

    for i in range(m):
        # Calculate utility and regret for each point
        dist_eff = weights[0] * (f_star[0] - pareto_front[i, 0]) / (f_star[0] - f_minus[0])
        dist_ex = weights[1] * (pareto_front[i, 1] - f_star[1]) / (f_minus[1] - f_star[1])
        
        S[i] = dist_eff + dist_ex
        R[i] = max(dist_eff, dist_ex)

    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)
    # Balanced Approach
    v = 0.5 

    # Calculate VIKOR Index
    Q = v * (S - S_star) / (S_minus - S_star) + (1 - v) * (R - R_star) / (R_minus - R_star)

    # Find the best solution
    best_index = np.argmin(Q)
    best_variables = sorted(res.X[best_index], reverse=True)
    best_objectives = pareto_front[best_index]

    
    print("\n--- OPTIMAL EXTRACTION SYSTEM DESIGN ---")
    print(f"Optimal Pressures (MPa)")
    print(f"HPT Extraction (P18):   {best_variables[0]:.3f}")
    print(f"Reheat (P2):           {best_variables[1]:.3f}")
    print(f"LPT Extraction 2 (P11): {best_variables[2]:.3f}")
    print(f"LPT Extraction 1 (P12): {best_variables[3]:.3f}")
    print(f"Efficiency:            {best_objectives[0]*100:.2f}%")
    print(f"Total Exergy Destruction: {best_objectives[1]:.2f} kW")

    # Re-run the engine with the best variables to get the constraints
    eff_val, ex_dest_val, x_out, sum_y = simulation_engine(best_variables)
    
    print(f"\nLP Turbine Exhaust Moisture:  {x_out:.4f}")
    print(f"Total Extraction Fraction:    {sum_y:.4f}")
    

    
    #  PLOTTING THE PARETO FRONT
    print("\nGenerating Pareto Front Graph...")
        
    # Extract data for plotting
    efficiencies = pareto_front[:, 0] * 100  
    exergies = pareto_front[:, 1]
        
    # Your Baseline parameters from main.py
    baseline_eff = 35.339
    baseline_exergy = 301.019
        
    plt.figure(figsize=(10, 6))
        
    # Plot Pareto font
    plt.scatter(efficiencies, exergies, color='blue', label='NSGA-II Pareto Front', alpha=0.6)
        
    # Plot the VIKOR Optimal Point
    best_eff = best_objectives[0] * 100
    best_exergy = best_objectives[1]
    plt.scatter(best_eff, best_exergy, color='red', marker='*', s=200, label='Optimal Design', zorder=5)
        
    # Plot the Baseline main.py point
    plt.scatter(baseline_eff, baseline_exergy, color='black', marker='X', s=100, label='Unoptimized Baseline', zorder=5)
        
    # Formatting
    plt.title('Multi-Objective Optimization: Efficiency vs Total Exergy Destruction', fontsize=14)
    plt.xlabel('Thermal Efficiency (%)', fontsize=12)
    plt.ylabel('Total Exergy Destruction (kW)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)

    # Annotation pointing out the baseline
    plt.annotate('Baseline violates\nmoisture limit (x > 0.88)', 
                xy=(baseline_eff, baseline_exergy), xytext=(baseline_eff-0.5, baseline_exergy-2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))

    plt.tight_layout()
    plt.show()

# GENERATE THERMODYNAMIC STATE POINTS
    def print_final_states(p_array):
        print("\n" + "="*75)
        print("OPTIMAL THERMODYNAMIC STATE POINTS")
        print("="*75)
        print(f"{'State':<6} | {'P (MPa)':<9} | {'T (°C)':<9} | {'h (kJ/kg)':<10} | {'s (kJ/kgK)':<10} | {'Quality (x)':<10}")
        print("-" * 75)
        
        # Constant Parameters
        P1 = 6; T1 = 563.15; eff_T = 0.88
        T3 = 535.15; P4 = 0.008; eff_P = 0.85; TTD = 5 
        P18, P2, P11, P12 = p_array 

        try:
            st1 = get_state("State 1", P=P1, T=T1)
            st18 = expand(st1, P18, eff_T, name="State 18")
            st2 = expand(st1, P2, eff_T, name="State 2")
            st3 = get_state("state 3", P=P2, T=T3)
            st11 = expand(st3, P11, eff_T, name="State 11")
            st12 = expand(st3, P12, eff_T, name="State 12")
            st4 = expand(st3, P4, eff_T, name="State 4")
            st5 = get_state("state 5", P=P4, x=0)
            st6 = compress(st5, P1, eff_P, name="State 6")
            
            st13 = get_state("state 13", P=P12, x=0)
            st15 = compress(st13, P1, eff_P, name="State 15")
            st7 = get_state("state 7", P=P1, T=st13.T-TTD)
            
            st16 = get_state("state 16", P=P11, x=0)
            st17 = compress(st16, P1, eff_P, name="State 17")
            st8 = get_state("state 8", P=st7.P, T=st16.T-TTD)
            
            st20 = get_state("state 20", P=P18, x=0)
            st21 = compress(st20, P1, eff_P, name="State 21")
            st9 = get_state("state 9", P=P1, T=st20.T-TTD)

            
            states = {
                "1": st1, "18": st18, "2": st2, "3": st3, "11": st11, 
                "12": st12, "4": st4, "5": st5, "6": st6, "13": st13, 
                "15": st15, "7": st7, "16": st16, "17": st17, "8": st8, 
                "20": st20, "21": st21, "9": st9
            }

            for name, state in states.items():
                T_celsius = state.T - 273.15
                quality = f"{state.x:.4f}" 
                
                print(f"{name:<6} | {state.P:<9.4f} | {T_celsius:<9.2f} | {state.h:<10.2f} | {state.s:<10.4f} | {quality:<10}")

        except Exception as e:
            print(f"Error generating state table: {e}")

    
    print_final_states(best_variables)
    print("~"*75 + "\n")