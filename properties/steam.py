from iapws import IAPWS97



def get_state(name, **kwargs):
    """Computes all thermodynamic properties for a given state."""
   
    state = IAPWS97(**kwargs)
    
    return state
