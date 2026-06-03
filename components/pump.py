from properties.steam import get_state



def compress(inlet_state, P_out, efficiency, name):
    '''Computes compressor's actual outlet state properties.'''

    # Ideal state
    s_out = inlet_state.s
    st_s = get_state(f"{name} Isentropic", P=P_out, s=s_out)
    
    # Actual state 
    h_a = inlet_state.h + (st_s.h - inlet_state.h) / efficiency
    
    return get_state(f"{name} Actual", P=P_out, h=h_a)
