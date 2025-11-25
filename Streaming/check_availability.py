def check_availability(tracks): #Función que comprueba qué algoritmos se pueden calcular con las variables disponibles.
    possible_list = []

    if ('ECG_HR' in tracks or 'ABP_HR' in tracks or 'HR' in tracks) and ('ABP_SYS' in tracks or 'BP_SYS' in tracks or 'NIBP_SYS' in tracks):
        possible_list.append('Shock Index')
    if 'PPLAT_CMH2O' in tracks and 'PEEP_CMH2O' in tracks:
        possible_list.append('Driving Pressure')
    if 'TV_EXP' in tracks and 'PIP_CMH2O' in tracks and 'PEEP_CMH2O' in tracks:
        possible_list.append('Dynamic Compliance')
    if 'PLETH_SAT_O2' in tracks and 'FiO2' in tracks:
        possible_list.append('ROX Index')
    if ('BT_CORE' in tracks or 'BT_BLD' in tracks) and ('BT_SKIN' in tracks or 'TEMP' in tracks):
        possible_list.append('Temp Comparison')
    #Variables MostCare
    if 'VOL_BLD_STROKE' in tracks and ('ECG_HR' in tracks or 'ABP_HR' in tracks or 'HR' in tracks):
        possible_list.append('Cardiac Output')
    if ('ABP_MEAN' in tracks or 'BP_MEAN' in tracks or 'NIBP_MEAN' in tracks) and 'CVP_MEAN' in tracks and 'Cardiac Output' in possible_list:
        possible_list.append('Systemic Vascular Resistance')
    if ('ABP_MEAN' in tracks or 'BP_MEAN' in tracks or 'NIBP_MEAN' in tracks) and 'Cardiac Output' in possible_list:
        possible_list.append('Cardiac Power Output')
    if ('ABP_SYS' in tracks or 'BP_SYS' in tracks or 'NIBP_SYS' in tracks) and 'VOL_BLD_STROKE' in tracks:
        possible_list.append('Effective Arterial Elastance')
    #Ver si se pueden añadir más variables MostCare

    #Variables autonomicas
    if 'ECG_I' in tracks or 'ECG_II' in tracks or 'ECG_III' in tracks or 'ECG_V' in tracks:
        possible_list.append('Heart Rate Variability') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'ABP' in tracks:
        possible_list.append('Blood Pressure Variability') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'ART' in tracks:
        possible_list.append('BRS') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'CO2' in tracks or 'RESP' in tracks:
        possible_list.append('RSA') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
        
    #Model
    if 'ICP' in tracks:
        possible_list.append('ICP Model')
    if 'PLETH' in tracks and 'ART' in tracks and 'ABP' in tracks:
        possible_list.append('ABP Model')
    
    #Pendiente Comprobar otros algoritmos

    return possible_list #Esta lista se envía al front para que el usuario seleccione.
