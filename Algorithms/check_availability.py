def check_availability(tracks): #Función que comprueba qué algoritmos se pueden calcular con las variables disponibles.
    possible_list = []

    if ('Intellivue/ECG_HR' in tracks or 'Intellivue/ABP_HR' in tracks or 'Intellivue/HR' in tracks) and ('Intellivue/ABP_SYS' in tracks or 'Intellivue/BP_SYS' in tracks or 'Intellivue/NIBP_SYS' in tracks):
        possible_list.append('Shock Index')
    if 'Intellivue/PPLAT_CMH2O' in tracks and 'Intellivue/PEEP_CMH2O' in tracks:
        possible_list.append('Driving Pressure')
    if 'Intellivue/TV_EXP' in tracks and 'Intellivue/PIP_CMH2O' in tracks and 'Intellivue/PEEP_CMH2O' in tracks:
        possible_list.append('Dynamic Compliance')
    if 'Intellivue/PLETH_SAT_O2' in tracks and 'Intellivue/FiO2' in tracks:
        possible_list.append('ROX Index')
    if ('Intellivue/BT_CORE' in tracks or 'Intellivue/BT_BLD' in tracks) and ('Intellivue/BT_SKIN' in tracks or 'Intellivue/TEMP' in tracks):
        possible_list.append('Temp Comparison')
    #Variables MostCare
    if 'Intellivue/VOL_BLD_STROKE' in tracks and ('Intellivue/ECG_HR' in tracks or 'Intellivue/ABP_HR' in tracks or 'Intellivue/HR' in tracks):
        possible_list.append('Cardiac Output')
    if ('Intellivue/ABP_MEAN' in tracks or 'Intellivue/BP_MEAN' in tracks or 'Intellivue/NIBP_MEAN' in tracks) and 'Intellivue/CVP_MEAN' in tracks and 'Cardiac Output' in possible_list:
        possible_list.append('Systemic Vascular Resistance')
    if ('Intellivue/ABP_MEAN' in tracks or 'Intellivue/BP_MEAN' in tracks or 'Intellivue/NIBP_MEAN' in tracks) and 'Cardiac Output' in possible_list:
        possible_list.append('Cardiac Power Output')
    if ('Intellivue/ABP_SYS' in tracks or 'Intellivue/BP_SYS' in tracks or 'Intellivue/NIBP_SYS' in tracks) and 'Intellivue/VOL_BLD_STROKE' in tracks:
        possible_list.append('Effective Arterial Elastance')
    #Ver si se pueden añadir más variables MostCare

    #Variables autonomicas
    if 'Intellivue/ECG_I' in tracks or 'Intellivue/ECG_II' in tracks or 'Intellivue/ECG_III' in tracks or 'Intellivue/ECG_V' in tracks:
        possible_list.append('Heart Rate Variability') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'Intellivue/ABP' in tracks:
        possible_list.append('Blood Pressure Variability') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'Intellivue/ART' in tracks:
        possible_list.append('BRS') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
    if 'Heart Rate Variability' in possible_list and 'Intellivue/CO2' in tracks or 'Intellivue/RESP' in tracks:
        possible_list.append('RSA') #NO SE PODRÁ MOSTRAR POR PANTALLA, SOLO INVESTIGAÇAO
        
    #Model
    if 'Intellivue/ICP' in tracks:
        possible_list.append('ICP Model')
    if 'Intellivue/PLETH' in tracks and 'Intellivue/ART' in tracks and 'Intellivue/ABP' in tracks:
        possible_list.append('ABP Model')
    
    #Pendiente Comprobar otros algoritmos

    return possible_list #Esta lista se envía al front para que el usuario seleccione.
