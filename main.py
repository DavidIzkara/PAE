import time
from modulo_monitor import monitorizar_actualizacion_recurso
from modulo_algoritmos import ejecutar_algoritmos
from modulo_interfaz import actualizar_interfaz

def main():
    modo = input("Selecciona modo (online/offline): ")
    
    if modo == "offline":
        print("Modo offline activado. Esperando acciones...")
        while True:
            time.sleep(1)
    elif modo == "online":
        print("Modo online activado. Iniciando monitoreo y ejecución de algoritmos...")
        while True:
            monitorizar_actualizacion_recurso()
            ejecutar_algoritmos()
            actualizar_interfaz()
            time.sleep(1)  # Para evitar que el bucle consuma mucho CPU
    else:
        print("Modo no reconocido. Saliendo.")

if __name__ == '__main__':
    main()
