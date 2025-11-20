import zarr
from numcodecs import Blosc 
import os
import time
import threading

from utils_Streaming import OUTPUT_DIR
from utils_zarr import leer_zattrs_de_grupo, FRAME_SIGNAL_DEMO, leer_senyal, FORMATO_TIMESTAMP, string_to_epoch1700

ZARR_PATH = os.path.join(OUTPUT_DIR, "session_data.zarr") # Ruta al archivo Zarr de ejemplo

## Ejemplo de uso:
"""
Name = "ECG_HR"

a = list_available_tracks(open_root(ZARR_PATH))  # Obtener la lista de tracks de señales

print("Tracks de señales disponibles en el Zarr:") 
for track in a:
    print(f" - {track}")
"""

## Mirar les spects de tots els tracks
""""
nombres_pistas = get_track_names_simplified(ZARR_PATH)
print("Nombres simplificados de pistas disponibles en el Zarr (con sus specs):")
for names in nombres_pistas:
    print(f" - {names}")
    metadatos = leer_zattrs_de_grupo(ZARR_PATH, FRAME_SIGNAL + names)
    for clave, valor in metadatos.items():
        print(f"    - {clave}: {valor}")
"""
## Prueva simple de lectura de 'last_updated' y cálculo de diferencia de tiempo
"""
metadatos_track = leer_zattrs_de_grupo(ZARR_PATH, FRAME_SIGNAL + track)
last_updated = metadatos_track['last_updated']
diferencia_tiempo = time.mktime(time.strptime(time.strftime(FORMATO_TIMESTAMP), FORMATO_TIMESTAMP)) - time.mktime(time.strptime(last_updated, FORMATO_TIMESTAMP))

print(f"Diferencia de tiempo entre ahora y la última actualización de '{track}': {diferencia_tiempo} segundos")
"""

## Prueva actualización en bucle (no me gusta mucho asi, pero funciona para prueva ràpida) (como no puedo escojer varias tracks a la vez para mirar la actualización ns)
"""
diferencia_tiempo_anterior = 10000

print(f"--- Iniciando Monitoreo de Actualización para la Pista: {track} ---")
print("Presione Ctrl+C para detener.")

while True:
    try:
        # Obtener y convertir el timestamp
        metadatos_track = leer_zattrs_de_grupo(ZARR_PATH, FRAME_SIGNAL + track)
        last_updated_str = metadatos_track.get('last_updated')

        if not last_updated_str:
            print(f"[ERROR] Clave 'last_updated' no encontrada. Esperando 5s...")
            time.sleep(5.0)
            continue
            
        # Calcular la diferencia de tiempo
        diferencia_tiempo = time.mktime(time.strptime(time.strftime(FORMATO_TIMESTAMP), FORMATO_TIMESTAMP)) - time.mktime(time.strptime(last_updated_str, FORMATO_TIMESTAMP))
        print(f"Tiempo sin actualizar: {diferencia_tiempo:.2f} s. ", end="")

        # Verificar si hubo una actualización (Diferencia de tiempo menor que la anterior)
        if diferencia_tiempo < diferencia_tiempo_anterior:
            # Esta condición es cierta solo si 'last_updated' ha avanzado en el Zarr.
            
            # Ignorar la primera iteración cuando diferencia_tiempo_anterior es la predeterminada
            if diferencia_tiempo_anterior != 10000:
                 print(f"\n ¡ACTUALIZACIÓN DETECTADA! Datos de {track} cambiaron.")
                 print(f"   [INFO] Diferencia anterior: {diferencia_tiempo_anterior:.2f} s")
                 print(f"   [ACCIÓN] Aquí iría la llamada a funciones para leer y procesar los nuevos datos.")

        # Determinar el intervalo de espera basado en la diferencia de tiempo actual
        if 0 <= diferencia_tiempo <= 10:
            sleep_time = 5.0
        elif 10 < diferencia_tiempo <= 20:
            sleep_time = 1.0
        else: # diferencia_tiempo > 20
            sleep_time = 0.5

        # Actualizar el valor anterior y esperar
        diferencia_tiempo_anterior = diferencia_tiempo
        
        print(f"Próxima verificación en {sleep_time} s...")
        time.sleep(sleep_time)

    except KeyboardInterrupt:
        # Permite al usuario detener el bucle con Ctrl+C
        print("\n--- Monitoreo detenido por el usuario. ---")
        break
    except ValueError as ve:
        # Captura errores si el formato de fecha es incorrecto
        print(f"\n[ERROR CRÍTICO] Error al parsear fecha: {ve}. Reintentando en 5s.")
        time.sleep(5)
    except Exception as e:
        # Captura cualquier otro error (ej. Zarr no disponible)
        print(f"\n[ERROR CRÍTICO] Ocurrió un error en el bucle: {e}. Reintentando en 5s.")
        time.sleep(5)
"""

## Lo Mismo pero en función

def monitorizar_actualizacion_recursivo(
    diferencia_anterior_segundos: float = 10000
) -> bool:
    """
    Sondea de forma recursiva una pista y ajusta el intervalo de espera.

    Args:
        track_name: Nombre de la pista a monitorizar.
        diferencia_anterior_segundos: La diferencia de tiempo de la iteración anterior.
        
    Returns:
        True si se detecta una actualización, False si se interrumpe o falla.
    """
    try:
        # Obtener y convertir el timestamp
        metadatos = leer_zattrs_de_grupo(ZARR_PATH, "")
        last_updated_str = metadatos.get('last_updated')

        if not last_updated_str:
            print(f"[ERROR] Clave 'last_updated' no encontrada. Esperando 5s y reintentando...")
            time.sleep(5.0)
            return monitorizar_actualizacion_recursivo(diferencia_anterior_segundos) # Volver a llamar

        # Convertir la cadena a objeto datetime y calcular la diferencia
        diferencia_tiempo = time.mktime(time.strptime(time.strftime(FORMATO_TIMESTAMP), FORMATO_TIMESTAMP)) - time.mktime(time.strptime(last_updated_str, FORMATO_TIMESTAMP))
        print(f"Tiempo sin actualizar: {diferencia_tiempo:.2f} s. ", end="")


        # Comprobar si hay una actualización
        # Si la diferencia actual es menor que la anterior, significa que 'last_updated' se actualizó.
        if diferencia_tiempo < diferencia_anterior_segundos:
            if diferencia_anterior_segundos != 10000: # Ignorar la primera llamada
                 # ¡ACTUALIZACIÓN DETECTADA!
                 print(f"\n   ¡ACTUALIZACIÓN DETECTADA! ({diferencia_anterior_segundos:.2f}s -> {diferencia_tiempo:.2f}s)")
                 return True # Señal de éxito: Propaga True hacia arriba.

        # Determinar el intervalo de espera (tiempo de 'sleep')
        if 0 <= diferencia_tiempo <= 10:
            sleep_time = 5.0
        elif 10 < diferencia_tiempo <= 20:
            sleep_time = 2.0
        elif 20 < diferencia_tiempo <= 25:
            sleep_time = 1.0
        else: # diferencia_tiempo > 25
            sleep_time = 0.5
            
        # Actualizar el valor anterior y pausar
        print(f"Próxima verificación en {sleep_time} s...")
        time.sleep(sleep_time)

        # Llamada Recursiva
        # Si la llamada recursiva retorna True, nosotros también retornamos True.
        if monitorizar_actualizacion_recursivo(diferencia_tiempo):
            return True
        
        return False # Esto se alcanzaría si una rama de la recursión se agota o se detiene. (La recursividad devuelve False, nosotros tambien tenemos que hacerlo)

    except RecursionError:
        print("\n[PELIGRO] Se alcanzó el límite de profundidad de recursión de Python. Deteniendo el monitoreo.")
        return False
    except KeyboardInterrupt:
        print("\n--- Monitoreo detenido por el usuario. ---")
        return False
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] Ocurrió un error: {e}. Deteniendo el monitoreo.")
        return False

# Ejemplo de uso de la función recursiva
if __name__ == "__main__":
    try:
        traks_a_monitorizar = ["ECG", "NIBP_DBP", "SPO2_SAT"] # Ejemplo de tracks 

        print(f"--- Iniciando Monitoreo Recursivo para la Pistas ---")

        if monitorizar_actualizacion_recursivo():
            print("\n¡ACTUALIZACIÓN DETECTADA !")
            time.sleep(1.0) # Pequeña pausa antes de leer los metadatos
            metadatos_general = leer_zattrs_de_grupo(ZARR_PATH, "")
            for track in traks_a_monitorizar:
                metadatos_track = leer_zattrs_de_grupo(ZARR_PATH, FRAME_SIGNAL_DEMO + track)
                if metadatos_track is not None:
                    if metadatos_track.get("last_updated") == metadatos_general.get("last_updated"):
                        print(f"   - La pista '{track}' fue actualizada.")
                        print(f"        - Ultima actualización contiene datos de {metadatos_track.get('last_update_secs')} segundos.")
                        segundos = string_to_epoch1700(metadatos_track.get("last_updated"))
                        datos_track = leer_senyal(ZARR_PATH, FRAME_SIGNAL_DEMO + track, segundos - metadatos_track.get("last_update_secs"), segundos)
                        print(f"        - Datos leídos: {datos_track}")
                    else:
                        print(f"   - La pista '{track}' NO fue actualizada.")
                    

    except KeyboardInterrupt:
        print("\n--- Monitoreo detenido por el usuario. ---")
