def get_fwh_coefficients(extraction_h, drain_out_h, fw_in_h, fw_out_h):
    """Returns energy balance coefficients for a matrix."""

    extract_coeff = extraction_h - drain_out_h
    fw_coeff = fw_in_h - fw_out_h
    
    return extract_coeff, fw_coeff
