# PAE

## Streaming

Encara queden moltes coses a canviar i millorar:
- Comentaris cristians (sense insultar a ningu i legibles pel ull huma)
- De zarr_to_algorithm, falta la part de un cop recollides les dades del update, cridar els algoritmes adients per executarlos
- Fer que puguin ser cridats des de un main (Adaptacio del codi)

Dins de la carpeta de Streaming podem trobar els següents archius:

- Streaming_to_zarr.py: Programa que s'encarrega de la lectura dels fitxers .vital en format Streaming o en format PROVA.
    En format PROVA, configures (manualment) el directori i el archiu que vols utilitzar i el programa simulara que te esta         passant aquell programa com si estiguessis reventlo per Streaming (Simplement tel passa amb les esperes determinades)
  Consta de varies funcions:
    vital_to_zarr_streaming: Modificació del vital_to_zarr original que permet utilitzarlo pel Streaming (Explicada en el doc)
    verificar_y_procesar: Prepara tot per poder executar la funcio anterior
    main_loop: Funcio de execució del programa y la que fa possible que es quedi en loop per tal de anar llegint les dades que
      s'actualitzin al .vital.

- utils_Streaming.py: Programa de ajuda pel Streaming amb funcions utils
- utils_zarr.py: Programa de ajuda pel tractament del zarr (algo diferent del altre perque estic fent proves pero despres sera     el mateix doc o aquesta es la idea)
- zarr_to_algorithms.py: Programa que monitoritza els timestamps del zar per detectar un canvi en ells i poder determinar que     hi ha hagut un update, posteriorment carrega nomes les dades del update procedents del .vital (en format de llista de dataframes) i mira quins algoritmes pot executar amb elles (fins aqui esta fet), llavors hauria de passa les dades als algoritmes possibles i alla farien la seva magia.
  Consta de la funció, monitorizar_actualizacion_recursivo: La qual vigila el zarr fins que hi hagi el canvi de timestamp, per        retornar True
                      
  
