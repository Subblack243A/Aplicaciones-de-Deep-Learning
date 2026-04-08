# 🤟 Manito — Lengua de Señas Colombiana (LSC)

Mano robótica controlada por **ESP32-C6** que reproduce las **28 letras** del abecedario en Lengua de Señas Colombiana. Se controla desde cualquier dispositivo con navegador o desde una app de escritorio en Python.

---

## 📦 Estructura del Proyecto

```nn
Manito/
├── ESP32_C6/                  # Firmware PlatformIO
│   ├── src/
│   │   └── main.cpp           # Firmware con servidor HTTP + WebSocket
│   └── platformio.ini         # Configuración de build
├── enviar_letra.py            # App de escritorio (Python/Tkinter)
└── calibracion.json           # Config local (se genera automáticamente)
```

---

## ⚡ Hardware Necesario

| Componente | Cantidad | Notas |
|---|---|---|
| ESP32-C6 DevKitC-1 | 1 | Microcontrolador principal |
| Micro servos (SG90 o similar) | 6 | Uno por dedo/articulación |
| Fuente 5V ≥ 3A | 1 | Para alimentar los servos |

### Conexión de Servos (GPIOs)

| Dedo | GPIO |
|---|---|
| Pulgar Palma | 3 |
| Pulgar Dedo | 2 |
| Índice | 23 |
| Medio | 22 |
| Anular | 6 |
| Meñique | 7 |

> ⚠️ **Importante**: Los servos deben alimentarse con fuente externa de 5V, no directamente del ESP32. Solo la señal PWM va al GPIO.

---

## 🔧 Configuración del Firmware (ESP32-C6)

### Requisitos

- [PlatformIO](https://platformio.org/) instalado (CLI o extensión VS Code)

### 1. Configurar Wi-Fi

Editar las credenciales en `ESP32_C6/src/main.cpp`:

```cpp
const char* ssid     = "TU_RED_WIFI";
const char* password = "TU_CONTRASEÑA";
```

### 2. Compilar

```bash
cd Manito/ESP32_C6
pio run
```

### 3. Subir al ESP32

```bash
pio run --target upload
```

### 4. Verificar conexión

Abrir el monitor serial para ver la IP asignada:

```bash
pio device monitor
```

Verás algo como:

```
Conectando a WiFi...
IP: 192.168.1.50
WebSocket listo en puerto 81
Abrir http://192.168.1.50 en el navegador
```

### 5. Subida OTA (opcional)

Para subir firmware por red sin cable USB, descomentar estas líneas en `platformio.ini`:

```ini
upload_protocol = espota
upload_port = 192.168.1.50   ; IP del ESP32-C6
```

---

## 🌐 Interfaz Web (HTML desde ESP32)

Una vez encendida la ESP32 y conectada al Wi-Fi:

1. **Abrir** `http://<IP_DEL_ESP32>` en cualquier navegador (celular, tablet, PC)
2. La conexión WebSocket se establece automáticamente

### Secciones de la Interfaz

#### 🔤 Vocales y Consonantes

- Toca una letra para que la mano la reproduzca
- Las letras con 🔄 (J, Ñ, Z) incluyen movimiento adicional

#### ✋ Controles

- **Abrir mano**: Todos los dedos a posición 0°
- **Cerrar mano**: Puño completo

#### 🖐️ Probar Dedo Individual

- Un **slider por cada dedo** (Índice, Medio, Anular, Meñique, Pulgar Dedo, Pulgar Palma)
- Mueve el slider y el dedo se mueve **en tiempo real**
- Ideal para verificar que cada servo responde correctamente

#### 🔧 Calibrar Letra

1. **Selecciona una letra** del grid de 28 letras
2. **Ajusta cada dedo** con:
   - **Slider**: arrastrar para ajustar libremente
   - **Botón −**: reduce 5° el ángulo
   - **Botón +**: aumenta 5° el ángulo
3. **▶ Probar**: envía la posición actual sin guardar (vista previa)
4. **💾 Guardar**: persiste los ángulos en la **flash del ESP32** (sobrevive reinicios)

#### 📟 Serial Monitor

- Muestra logs en tiempo real del ESP32 vía WebSocket

---

## 🐍 App de Escritorio (Python)

### Requisitos

```bash
pip install websocket-client
```

> Tkinter viene incluido con Python en la mayoría de distribuciones.

### Ejecutar

```bash
cd Manito
python enviar_letra.py
```

### Configuración

1. Al iniciar, **ingresa la IP** del ESP32 en el campo superior
2. Presiona **Conectar**

### Pestañas

| Pestaña | Función |
|---|---|
| **Principal** | Botones de las 28 letras, abrir/cerrar mano, serial monitor |
| **Probar Dedo** | Slider por dedo para control individual en tiempo real |
| **Calibración** | Seleccionar letra → ajustar ángulos con sliders y botones +/− → Probar → Guardar |
| **Límites** | Configurar rango mínimo/máximo de cada servo (protección mecánica) |

### Persistencia

- **Local**: `calibracion.json` se crea automáticamente en el directorio del proyecto con todas las posiciones calibradas
- **ESP32**: Al presionar "Guardar en ESP32", los valores se almacenan en flash (Preferences) y se sincronizan al reconectar

---

## 📡 Protocolo de Comunicación (WebSocket)

La comunicación entre cliente (web/Python) y ESP32 es vía **WebSocket en puerto 81**.

### Comandos del Cliente → ESP32

| Comando | Acción |
|---|---|
| `a` ... `z`, `ñ` | Ejecutar letra LSC |
| `reset` | Abrir mano (todos a 0°) |
| `cerrar` | Cerrar puño |
| `movefinger\|finger\|angle` | Mover un dedo individual (`finger`: idx, med, anu, men, pd, pp) |
| `preview\|v1,v2,v3,v4,v5,v6` | Vista previa de posición (no guarda) |
| `setletra\|letra\|v1,v2,v3,v4,v5,v6` | Guardar posición de letra en flash |
| `setlimits\|finger\|min\|max` | Guardar límites de un dedo |
| `getconfig` | Solicitar toda la configuración actual |

### Respuestas del ESP32 → Cliente

| Formato | Significado |
|---|---|
| `CONFIG:{json}` | JSON con `limits` y `letras` |
| Texto libre | Log/debug del ESP32 |

---

## 🔄 Orden Natural de Dedos

El firmware mueve los dedos en este orden al formar cada letra:

1. **Índice** → 2. **Medio** → 3. **Anular** → 4. **Meñique** → 5. **Pulgar Dedo** → 6. **Pulgar Palma**

> El pulgar siempre se mueve **al final** para "sellar" la forma de la letra. Esto minimiza la necesidad de reabrir la mano entre letras consecutivas.

---

## 🤝 Créditos

Proyecto para **Aplicaciones de Deep Learning** — Semestre IX  
Universidad — Lengua de Señas Colombiana (LSC)
