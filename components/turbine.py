from properties.steam import get_state



def expand(inlet_state, p_out, efficiency, name):
    '''Computes turbine's actual outlet state properties.'''

    # Ideal  state
    s_out = inlet_state.s
    st_s = get_state(f"{name} Isentropic", P=p_out, s=s_out)
    
    # Actual state 
    h_a = inlet_state.h - (efficiency * (inlet_state.h - st_s.h))
    
    return get_state(f"{name} Actual", P=p_out, h=h_a)
