# **MANUAL UTILS\_ZARR**

# **1\. Introducció**

Aquest mòdul defineix totes les funcions necessàries per:

* Llegir i escriure senyals fisiològics en format Zarr

* Convertir fitxers `.vital` de VitalDB

* Gestionar estructures jeràrquiques Zarr

* Obtenir finestres temporals de les dades

* Escriure prediccions de models

* Exportar subconjunts d’un Zarr

**L’objectiu és proporcionar un conjunt d’eines alt-nivell i consistents per treballar amb dades fisiològiques de gran mida, de forma incremental i robusta.**

# **2\. Conceptes clau**

## **2.1 Què és Zarr?**

Zarr és un format per emmagatzemar **arrays multidimensionals** en disc, amb:

* Accés ràpid a parts petites de dades (lazy loading)

* Compressió per chunks

* Capacitat de *redimensionar arrays*

* Estructura jeràrquica tipus filesystem (`group → subgroup → arrays`)

L’estructura típica és:

`mydata.zarr/`  
  `├── signals/`  
  `│     ├── Intellivue/`  
  `│     │       ├── PLETH/`  
  `│     │             └── time_ms`  
  `│     │             └── value`  
  `├── algorithms/`  
  `│     └── rr_pred/`  
  `│             ├── time_ms`  
  `│             └── value`  
  `└── .zattrs`

# **3\. Helpers i configuració**

## **3.1 `open_root(store_path)`**

**Què fa:**  
 Obre o crea un contenidor `.zarr`.

**Arguments:**

* `store_path`: ruta al fitxer `.zarr`.

**Retorna:**

* grup arrel (`zarr.Group`)

**Notes pràctiques:**

* Sempre utilitza `mode="a"` → no sobreescriu.

* Crea carpetes si no existeixen.

**Exemple:**

`root = open_root("data/session01.zarr")`

## **3.2 `epoch1700_to_datetime(ts_seconds)`**

**Què fa:** converteix el format de temps de VitalDB (segons des de 1700\) a `datetime`.

**Arguments:**

* `ts_seconds`: float

**Retorna:**

* `datetime.datetime`

**Quan usar-ho:**  
 Quan vols mostrar o logar els timestamps de forma humana.

## **3.3 `normalize_signal_path(signal: str) -> str:`**

**Què fa:** Normalitza la senyal a un path.

**Arguments:**

* `signal`: str

**Retorna:**

* `path: str`

**Quan usar-ho:**  
 Quan volem aconseguir el path real d’on es guarda una senyal

# **4\. Gestió d’estructura**

# **4.1 `safe_group(root, path)`**

**Què fa:**  
 Crea tots els subgrups necessaris i retorna l’últim. Retorna un subgrup .zarr

**Arguments:**

* `root`: `zarr.Group`  
* `path`: string, ex: `"Intellivue/PLETH"`

**Retorna:**

* grup final

**Notes:**

* És com un `mkdir -p`.

**Error comú evitat:** que no existeixi el camí quan escrius.

## **4.2 `get_group_if_exists(root, path)`**

Igual que `safe_group` però **no crea** res.

**Ús:** lectura segura.

# **5\. Creació i escriptura incremental**

## **5.1 `get_or_create_1d(group, name, …(no necessaris de posar))`**

Crea (o retorna) un array 1D amb:

* `shape=(0,)`  
* `maxshape=(None,)` → appendable  
* compressió Blosc-zstd

**Arguments principals:**

| Argument | Explicació |
| ----- | ----- |
| `group` | Grup pare |
| `name` | Nom del dataset |
| `dtype` | Tipus (`f4`, `i8`) |
| `fill` | Valor per defecte |
| `chunks` | Mida del chunk |

**Retorna:**

* `zarr.Array`

**Notes:**

* Si ja existeix, el retorna sense modificar-lo.  
* Imprescindible per senyals fisiològiques.

## **5.2 `append_1d(arr, data)`**

Afegeix dades al final d’un array existent.

**Arguments:**

* `arr`: Zarr array  
* `data`: vector numpy

**Retorna:** `None`

**Fluxe:**

1. Llegeix mida actual  
2. `resize()`  
3. Afegeix dades

**Errors típics:**

* Intentar passar un array 2D → ha de ser 1D.

## **5.3 get\_or\_create\_signal\_pair(parent\_group, signal\_path)**

Crea l’estructura estàndard d’un senyal. Trobarem en la carpeta d’un senyal les següents arrays:

`time_ms   (int64)`  
`values    (float32)`

**Arguments:**

* `el arxiu .zarr`  
* `signal` ex: `"Intellivue/PLETH"`

**Retorna:**

* `En forma de tuple(time_array,value_array)`

# **6\. Lectura i navegació**

## **6.1 load\_track(root, track\_path)**

Llegeix un senyal complet.

**Arguments:**

* `track`: `una senyal, la que volem`

**Retorna:**

* `t_abs_ms` (absoluts)  
* `t_rel_ms` (relatius al primer punt)  
* `values`

**Errors:**

* `Si no existeix la senyal retorna None`

**Notes pràctiques:**

* Ideal per loading complet abans de slicing.

## **6.2 slice\_by\_seconds(t\_rel\_ms, vals, start\_s, end\_s)**

Retalla segons una finestra temporal.

**Retorna:**

* `t_rel_ms_sub`  
* `vals_sub`

**Notes:**

* Usa `searchsorted` → molt ràpid  
* Treballa en ms però el window s’especifica en **segons**

## **6.3 walk\_arrays(node)**

Recorre tota l’estructura i retorna tots els arrays.

## **6.4 list\_available\_tracks(root)**

Retorna:

* llista de senyals  
* llista de prediccions

Exclou arrays de temps (`_time_ms`).

# **7\. Conversió Vital → Zarr**

## **7.1 vital\_to\_zarr(vital\_file, zarr\_path, tracks, window\_secs=None, chunk\_len=30000)**

**Funció principal per crear Zarr a partir de `.vital`.**

### **Workflow intern:**

1. Validació del fitxer.  
2. Obrir `.vital` amb `VitalFile`.  
3. Obrir o crear `.zarr`.  
4. Crear metadades del root.  
5. Per cada track:  
   * Convertir timestamps a ms.  
   * Convertir valors a float32.  
   * Crear grup `signals/<track>/`.  
   * Obtenir `time_ms` i `value`.  
   * Eliminar timestamps duplicats (només deixar els més nous).  
   * Escriure incrementalment.

### **Retorna:**

`None` (imprimeix resum final)

### **Errors comuns:**

* Tracks inexistents → s’avisa però no s’atura.  
* Samples tots NaN → s’ignoren.

# **8\. Funcions d’alt nivell: lectura**

## **8.1 leer\_senyal(zarr\_path, track\_path, start\_s=None, end\_s=None)**

Fa tot el pipeline:

1. Obrir root  
2. Carregar senyal  
3. Opcional: retallar finestra  
4. Retornar dict

**Retorna dict amb:**

`{`  
  `"t_abs_ms": ...,`  
  `"t_rel_ms": ...,`  
  `"values": ...`  
`}`

## **8.2 leer\_multiples\_senyales(zarr\_path, track\_paths, ...)**

Llegeix diversos senyals a la vegada.

**Retorna:**

`{`  
  `track_path1: { ... },`  
  `track_path2: { ... },`  
  `...`  
`}`

Útil per multimodalitat (ECG \+ PLETH \+ HR).

# **9\. Funcions d’alt nivell: escriptura**

## **9.1 escribir\_senyal(zarr\_path, track\_path, timestamps\_ms, values, metadata=None)**

Escriu manualment un senyal al Zarr.

**Arguments:**

* `track_path`: ex: `"Intellivue/PLETH"`  
* `timestamps_ms`: vector 1D  
* `values`: vector 1D  
* `metadata`: opcional

### **Validacions:**

* mateix tamany arrays  
* conversions a `int64` i `float32`

### **Notes:**

* Utilitza `get_or_create_signal_pair`  
* Afegeix al final del Zarr (append)

## **9.2 escribir\_prediccion(zarr\_path, pred\_name, timestamps\_ms, values, modelo\_info=None)**

Igual que `escribir_senyal` però dins `pred/`.

### **Afegeix automàticament:**

* `model_<clau>`  
* `prediction_created`

## **9.3 obtener\_info\_zarr(zarr\_path)**

Resum general del Zarr:

* nº senyals  
* nº prediccions  
* duració  
* metadades

**Retorna:** dict

## **9.4 escribir\_batch\_senyales(zarr\_path, datos\_dict)**

Rep:

`{`  
  `"Intellivue/ECG": (timestamps, values),`  
  `"Intellivue/PLETH": (timestamps, values),`  
`}`

i escriu totes les senyals.

## **9.5 exportar\_ventana\_temporal(...)**

Pipeline complet:

1. Llegir múltiples senyals  
2. Extreure finestra  
3. Re-escriure en un Zarr nou

**Útil per:**

* fer datasets més petits  
* crear conjunts d’entrenament

# **10\. Workflow recomanat**

## **Escriure dades:**

`root = open_root(...)`  
`time_arr, val_arr = get_or_create_signal_pair(root, track)`  
`append_1d(time_arr, timestamps)`  
`append_1d(val_arr, values)`

## **Llegir dades:**

`root = open_root(...)`  
`t_abs, t_rel, vals = load_track(root, track)`  
`t_sub, v_sub = slice_by_seconds(t_rel, vals, 0, 60)`

## **Conversió automàtica:**

`vital_to_zarr("data.vital", "output.zarr", ["PLETH", "HR"])`

# **11\. Errors típics i com evitar-los**

### **❌ Paths incorrectes**

Zarr és **sensible a la jerarquia**:  
 Correcte:

`signals/Intellivue/PLETH`

### **❌ Timestamps en segons en lloc de ms**

Tot el mòdul usa **mil·lisegons**.

### **❌ Arrays que no són 1D**

`append_1d` només accepta 1D.

### **❌ Tamany desigual entre timestamps i valors**

Produirà `ValueError`.

# **12\. Exemples simples**

## **Escriure un sinus:**

`t = np.arange(0, 10000, 5)`  
`v = np.sin(2*np.pi*t/1000)`  
`escribir_senyal("data.zarr", "test/sine", t, v)`

## **Llegir 30 segons:**

`d = leer_senyal("data.zarr", "signals/Intellivue/PLETH", 0, 30)`  
