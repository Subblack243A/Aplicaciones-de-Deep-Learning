#include <ArduinoJson.h>
#include <ArduinoOTA.h>
#include <WiFiManager.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <WiFi.h>

WebSocketsServer webSocket = WebSocketsServer(81);
WebServer server(80);
Preferences prefs;

#define PIN_PULGAR_PALMA 3
#define PIN_PULGAR_DEDO 2
#define PIN_INDICE 23
#define PIN_MEDIO 22
#define PIN_ANULAR 6
#define PIN_MENIQUE 7

#define SERVO_FREQ 50
#define SERVO_RES 16
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2500
#define SERVO_MIN_DUTY ((SERVO_MIN_US * 65536L) / 20000)
#define SERVO_MAX_DUTY ((SERVO_MAX_US * 65536L) / 20000)

const int servoPins[] = {PIN_PULGAR_PALMA, PIN_PULGAR_DEDO, PIN_INDICE,
                         PIN_MEDIO,        PIN_ANULAR,      PIN_MENIQUE};
#define NUM_SERVOS 6

#define SERVO_MIN 0
#define SERVO_MAX 180

struct FingerLimits {
  int minVal;
  int maxVal;
};

FingerLimits limits[] = {
    {0, 190}, // idx
    {0, 180}, // med
    {0, 180}, // anu
    {0, 180}, // men
    {0, 120}, // pd
    {0, 90},  // pp
};

const char *FINGER_KEYS[] = {"idx", "med", "anu", "men", "pd", "pp"};

int posIndice = 0;
int posMedio = 0;
int posAnular = 0;
int posMenique = 0;
int posPulgarDedo = 0;
int posPulgarPalma = 0;

int delayBetweenFingers = 50;
int delayBetweenLetters = 1500;
int delayBeforeMovements = 200;

#define MAX_MOVEMENTS 8

struct FingerMovement {
  int fingerIdx; // 0=idx,1=med,2=anu,3=men,4=pd,5=pp
  int toAngle;   // delta from current position
  int repeat;
  int delayMs;
};

// pos[6] = [idx, med, anu, men, pd, pp]
// preThumbMask: bitmask of 4-finger indices (bit0=idx..bit3=men)
//   that must close BEFORE the thumb. 0 = default order.
#define NUM_LETRAS (sizeof(letras) / sizeof(letras[0]))
struct LetraPos {
  const char *nombre;
  int pos[6];
  uint8_t preThumbMask;
  FingerMovement movements[MAX_MOVEMENTS];
  int numMovements;
};

// Key string used for Preferences storage
const char *LETRA_KEYS[] = {"a", "b", "c", "d", "e", "f",  "g", "h", "i",
                            "j", "k", "l", "m", "n", "nn", "o", "p", "q",
                            "r", "s", "t", "u", "v", "w",  "x", "y", "z"};

// preThumbMask bits: 0=idx, 1=med, 2=anu, 3=men
//   M: men closes before thumb (0x08)
//   N/Ñ: anu+men close before thumb (0x0C)
LetraPos letras[] = {
    {"a", {180, 180, 180, 180, 60, 0},  0, {}, 0},
    {"b", {0, 0, 0, 0, 120, 50},        0, {}, 0},
    {"c", {90, 90, 90, 90, 45, 0},      0, {}, 0},
    {"d", {0, 180, 180, 180, 60, 25},   0, {}, 0},
    {"e", {90, 90, 95, 90, 0, 37},      0, {}, 0},
    {"f", {120, 0, 0, 0, 90, 30},       0, {}, 0},
    {"g", {0, 180, 180, 180, 0, 0},     0, {{0, 180, 3, 180}}, 1},
    {"h", {0, 0, 180, 180, 120, 50},    0, {{0, 180, 2, 150}, {1, 180, 2, 150}}, 2},
    {"i", {180, 180, 180, 0, 60, 25},   0, {}, 0},
    {"j", {180, 180, 180, 0, 60, 25},   0, {{3, 40, 2, 150}}, 1},
    {"k", {0, 0, 180, 180, 45, 25},     0, {}, 0},
    {"l", {0, 180, 180, 180, 0, 0},     0, {}, 0},
    // M: men first → thumb → idx/med/anu halfway
    {"m", {90, 90, 90, 180, 120, 50},   0x08, {}, 0},
    // N: men+anu first → thumb → idx/med halfway
    {"n", {90, 90, 180, 180, 120, 40},  0x0C, {}, 0},
    // Ñ: same as N + idx/med wiggle
    {"ñ", {90, 90, 180, 180, 120, 40},  0x0C, {{0, -90, 4, 150}, {1, -90, 4, 150}}, 2},
    {"o", {120, 120, 120, 120, 60, 25}, 0, {}, 0},
    {"p", {0, 0, 180, 180, 45, 0},      0, {}, 0},
    {"q", {180, 180, 180, 180, 0, 0},   0, {}, 0},
    {"r", {0, 0, 180, 180, 120, 50},    0, {}, 0},
    // S: slight bend, pp oscillates 3x, idx dips between
    {"s", {30, 30, 30, 30, 0, 0},       0, {{5, 50, 1, 150}, {0, 30, 1, 100}, {5, 50, 1, 150}, {0, 30, 1, 100}, {5, 50, 1, 150}}, 5},
    {"t", {180, 180, 180, 180, 60, 30}, 0, {}, 0},
    {"u", {0, 0, 180, 180, 60, 25},     0, {}, 0},
    {"v", {0, 0, 180, 180, 120, 50},    0, {}, 0},
    {"w", {0, 0, 0, 180, 120, 50},      0, {}, 0},
    {"x", {90, 180, 180, 180, 60, 25},  0, {}, 0},
    {"y", {180, 180, 180, 0, 0, 0},     0, {}, 0},
    // Z: anu+med extend slightly
    {"z", {0, 180, 180, 180, 120, 50},  0, {{2, -30, 2, 150}, {1, -30, 2, 150}}, 2},
};

char _logBuf[256];
bool _wsReady = false;

void wsLog(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(_logBuf, sizeof(_logBuf), fmt, args);
  va_end(args);
  Serial.println(_logBuf);
  if (_wsReady)
    webSocket.broadcastTXT(_logBuf);
}

void servoWrite(int pin, int angulo) {
  angulo = constrain(angulo, SERVO_MIN, SERVO_MAX);
  uint32_t duty =
      SERVO_MIN_DUTY + (long)(SERVO_MAX_DUTY - SERVO_MIN_DUTY) * angulo / 180;
  ledcWrite(pin, duty);
}

void moverServo(int pin, int desde, int hasta) {
  desde = constrain(desde, SERVO_MIN, SERVO_MAX);
  hasta = constrain(hasta, SERVO_MIN, SERVO_MAX);
  int diff = abs(hasta - desde);
  if (diff == 0) return;
  for (int step = 1; step <= diff; step++) {
    int pos = desde + (long)(hasta - desde) * step / diff;
    servoWrite(pin, pos);
    delay(5);
  }
  servoWrite(pin, hasta);
}

// ── Finger pin/position helpers ──────────────────────────────────────
// Ordered arrays for mapping fingerIdx → pin and position pointer
const int fingerPins[] = {PIN_INDICE, PIN_MEDIO, PIN_ANULAR,
                          PIN_MENIQUE, PIN_PULGAR_DEDO, PIN_PULGAR_PALMA};
int *fingerPos[] = {&posIndice, &posMedio, &posAnular,
                    &posMenique, &posPulgarDedo, &posPulgarPalma};

// Open order:  pd(4) → pp(5) → idx(0) → med(1) → anu(2) → men(3)
const int OPEN_ORDER[]  = {4, 5, 0, 1, 2, 3};
// Close order: idx(0) → med(1) → anu(2) → men(3) → pp(5) → pd(4)
const int CLOSE_ORDER[] = {0, 1, 2, 3, 5, 4};

// ── Ordered finger movement ─────────────────────────────────────────
// Open: pd → pp → fingers(idx→men)
// Close: fingers(idx→men) → pp → pd

void ponerPosicion(int idx, int med, int anu, int men, int pd, int pp) {
  int target[6] = {idx, med, anu, men, pd, pp};
  for (int i = 0; i < 6; i++) {
    target[i] = constrain(target[i], limits[i].minVal, limits[i].maxVal);
  }

  // Classify each finger: opening (angle decreases) or closing (angle increases)
  bool isOpening[6];
  for (int i = 0; i < 6; i++) {
    isOpening[i] = (target[i] < *fingerPos[i]);
  }

  // Phase 1: Move fingers that are OPENING (in open order: pd→pp→idx→med→anu→men)
  for (int o = 0; o < 6; o++) {
    int fi = OPEN_ORDER[o];
    if (isOpening[fi]) {
      moverServo(fingerPins[fi], *fingerPos[fi], target[fi]);
      *fingerPos[fi] = target[fi];
      delay(delayBetweenFingers);
    }
  }

  // Phase 2: Move fingers that are CLOSING (in close order: idx→med→anu→men→pp→pd)
  for (int o = 0; o < 6; o++) {
    int fi = CLOSE_ORDER[o];
    if (!isOpening[fi] && target[fi] != *fingerPos[fi]) {
      moverServo(fingerPins[fi], *fingerPos[fi], target[fi]);
      *fingerPos[fi] = target[fi];
      delay(delayBetweenFingers);
    }
  }

  wsLog("Pos: idx=%d med=%d anu=%d men=%d pd=%d pp=%d",
        target[0], target[1], target[2], target[3], target[4], target[5]);
}

// ── Execute repetitive movements for a letter ───────────────────────

void ejecutarMovimientos(LetraPos &letra) {
  if (letra.numMovements <= 0) return;
  delay(delayBeforeMovements);

  for (int m = 0; m < letra.numMovements && m < MAX_MOVEMENTS; m++) {
    FingerMovement &mov = letra.movements[m];
    if (mov.fingerIdx < 0 || mov.fingerIdx >= 6) continue;

    int pin = fingerPins[mov.fingerIdx];
    int *posRef = fingerPos[mov.fingerIdx];
    int origPos = *posRef;
    int targetPos = constrain(origPos + mov.toAngle, SERVO_MIN, SERVO_MAX);

    wsLog("Mov: dedo=%d dest=%d rep=%d", mov.fingerIdx, targetPos, mov.repeat);

    for (int r = 0; r < mov.repeat; r++) {
      moverServo(pin, *posRef, targetPos);
      *posRef = targetPos;
      delay(mov.delayMs);
      moverServo(pin, *posRef, origPos);
      *posRef = origPos;
      delay(mov.delayMs);
    }
  }
}

// ── Thumb transition helpers ─────────────────────────────────────────

void abrirPulgar() {
  // Open thumb: pd first, then pp (open order for thumb)
  moverServo(PIN_PULGAR_DEDO, posPulgarDedo, 0);
  posPulgarDedo = 0;
  delay(delayBetweenFingers);
  moverServo(PIN_PULGAR_PALMA, posPulgarPalma, 0);
  posPulgarPalma = 0;
}

void cerrarPulgar(int targetPd, int targetPp) {
  // Close thumb to letter position: pp first, then pd (close order for thumb)
  targetPp = constrain(targetPp, limits[5].minVal, limits[5].maxVal);
  targetPd = constrain(targetPd, limits[4].minVal, limits[4].maxVal);
  if (targetPp != posPulgarPalma) {
    moverServo(PIN_PULGAR_PALMA, posPulgarPalma, targetPp);
    posPulgarPalma = targetPp;
    delay(delayBetweenFingers);
  }
  if (targetPd != posPulgarDedo) {
    moverServo(PIN_PULGAR_DEDO, posPulgarDedo, targetPd);
    posPulgarDedo = targetPd;
    delay(delayBetweenFingers);
  }
}

void moverDedo(int fi, int target) {
  target = constrain(target, limits[fi].minVal, limits[fi].maxVal);
  if (target != *fingerPos[fi]) {
    moverServo(fingerPins[fi], *fingerPos[fi], target);
    *fingerPos[fi] = target;
    delay(delayBetweenFingers);
  }
}

int findLetra(const char *nombre) {
  for (int i = 0; i < (int)NUM_LETRAS; i++)
    if (strcmp(letras[i].nombre, nombre) == 0) return i;
  if (strcmp(nombre, "nn") == 0)
    for (int i = 0; i < (int)NUM_LETRAS; i++)
      if (strcmp(letras[i].nombre, "ñ") == 0) return i;
  return -1;
}

// Track if previous letter had fingers mounted on thumb
static bool prevWasSpecial = false;

void hacerLetra(const char *nombre) {
  int li = findLetra(nombre);
  if (li < 0) { wsLog("Letra desconocida: %s", nombre); return; }

  wsLog("Sena: %s", letras[li].nombre);
  int *p = letras[li].pos;
  uint8_t mask = letras[li].preThumbMask;

  // ── Phase 1: Safe release from previous letter ──
  if (prevWasSpecial) {
    // Coming from m/n/ñ: mounted fingers release first → thumb → rest
    for (int i = 3; i >= 0; i--) moverDedo(i, 0);  // men→anu→med→idx
    moverDedo(4, 0);  // pd
    moverDedo(5, 0);  // pp
  } else {
    // Thumb always stretches to 0 between letters
    moverDedo(4, 0);  // pd
    moverDedo(5, 0);  // pp
  }

  // ── Phase 2: Form new letter ──
  if (mask) {
    // Special (m, n, ñ): open all fingers → pre-thumb close → thumb → rest
    for (int i = 0; i < 4; i++) moverDedo(i, 0);

    for (int i = 0; i < 4; i++)
      if (mask & (1 << i)) moverDedo(i, p[i]);
    cerrarPulgar(p[4], p[5]);
    for (int i = 0; i < 4; i++)
      if (!(mask & (1 << i))) moverDedo(i, p[i]);

    prevWasSpecial = true;
  } else {
    // Normal: fingers direct → thumb closes
    for (int i = 0; i < 4; i++) moverDedo(i, p[i]);
    cerrarPulgar(p[4], p[5]);

    prevWasSpecial = false;
  }

  wsLog("Pos: idx=%d med=%d anu=%d men=%d pd=%d pp=%d",
        p[0], p[1], p[2], p[3], p[4], p[5]);

  ejecutarMovimientos(letras[li]);
}

void resetMano() {
  wsLog("Reset: mano abierta");
  ponerPosicion(0, 0, 0, 0, 0, 0);
}

void cerrarMano() {
  wsLog("Cerrar: puno completo");
  ponerPosicion(180, 180, 180, 180, 120, 50);
}

// ── Preferences persistence ──────────────────────────────────────────

void guardarConfig() {
  prefs.begin("manito", false);

  // Save settings
  prefs.putInt("dl_fingers", delayBetweenFingers);
  prefs.putInt("dl_letters", delayBetweenLetters);
  prefs.putInt("dl_movs", delayBeforeMovements);

  // Save limits
  for (int i = 0; i < 6; i++) {
    String keyMin = String(FINGER_KEYS[i]) + "_min";
    String keyMax = String(FINGER_KEYS[i]) + "_max";
    prefs.putInt(keyMin.c_str(), limits[i].minVal);
    prefs.putInt(keyMax.c_str(), limits[i].maxVal);
  }

  // Save letter positions
  for (int i = 0; i < (int)NUM_LETRAS; i++) {
    for (int j = 0; j < 6; j++) {
      String key = String(LETRA_KEYS[i]) + "_" + String(j);
      prefs.putInt(key.c_str(), letras[i].pos[j]);
    }
    String keyPt = String(LETRA_KEYS[i]) + "_pt";
    prefs.putInt(keyPt.c_str(), letras[i].preThumbMask);
    String keyNm = String(LETRA_KEYS[i]) + "_nm";
    prefs.putInt(keyNm.c_str(), letras[i].numMovements);
    for (int m = 0; m < letras[i].numMovements && m < MAX_MOVEMENTS; m++) {
      String base = String(LETRA_KEYS[i]) + "_m" + String(m);
      prefs.putInt((base + "f").c_str(), letras[i].movements[m].fingerIdx);
      prefs.putInt((base + "t").c_str(), letras[i].movements[m].toAngle);
      prefs.putInt((base + "r").c_str(), letras[i].movements[m].repeat);
      prefs.putInt((base + "d").c_str(), letras[i].movements[m].delayMs);
    }
  }

  prefs.end();
  wsLog("Config guardada en flash");
}

void cargarConfig() {
  prefs.begin("manito", true);

  // Load settings
  delayBetweenFingers = prefs.getInt("dl_fingers", delayBetweenFingers);
  delayBetweenLetters = prefs.getInt("dl_letters", delayBetweenLetters);
  delayBeforeMovements = prefs.getInt("dl_movs", delayBeforeMovements);

  for (int i = 0; i < 6; i++) {
    String keyMin = String(FINGER_KEYS[i]) + "_min";
    String keyMax = String(FINGER_KEYS[i]) + "_max";
    limits[i].minVal = prefs.getInt(keyMin.c_str(), limits[i].minVal);
    limits[i].maxVal = prefs.getInt(keyMax.c_str(), limits[i].maxVal);
  }

  for (int i = 0; i < (int)NUM_LETRAS; i++) {
    for (int j = 0; j < 6; j++) {
      String key = String(LETRA_KEYS[i]) + "_" + String(j);
      letras[i].pos[j] = prefs.getInt(key.c_str(), letras[i].pos[j]);
    }
    String keyPt = String(LETRA_KEYS[i]) + "_pt";
    letras[i].preThumbMask = prefs.getInt(keyPt.c_str(), letras[i].preThumbMask);
    String keyNm = String(LETRA_KEYS[i]) + "_nm";
    letras[i].numMovements = prefs.getInt(keyNm.c_str(), letras[i].numMovements);
    if (letras[i].numMovements > MAX_MOVEMENTS) letras[i].numMovements = MAX_MOVEMENTS;
    for (int m = 0; m < letras[i].numMovements; m++) {
      String base = String(LETRA_KEYS[i]) + "_m" + String(m);
      letras[i].movements[m].fingerIdx = prefs.getInt((base + "f").c_str(), letras[i].movements[m].fingerIdx);
      letras[i].movements[m].toAngle   = prefs.getInt((base + "t").c_str(), letras[i].movements[m].toAngle);
      letras[i].movements[m].repeat    = prefs.getInt((base + "r").c_str(), letras[i].movements[m].repeat);
      letras[i].movements[m].delayMs   = prefs.getInt((base + "d").c_str(), letras[i].movements[m].delayMs);
    }
  }

  prefs.end();
  wsLog("Config cargada desde flash");
}

void enviarConfigJSON() {
  JsonDocument doc;

  // Settings
  JsonObject jSettings = doc["settings"].to<JsonObject>();
  jSettings["delay_between_fingers_ms"] = delayBetweenFingers;
  jSettings["delay_between_letters_ms"] = delayBetweenLetters;
  jSettings["delay_before_movements_ms"] = delayBeforeMovements;

  // Limits
  JsonObject jLimits = doc["limits"].to<JsonObject>();
  for (int i = 0; i < 6; i++) {
    JsonObject f = jLimits[FINGER_KEYS[i]].to<JsonObject>();
    f["min"] = limits[i].minVal;
    f["max"] = limits[i].maxVal;
  }

  // Letters with positions and movements
  JsonObject jLetras = doc["letras"].to<JsonObject>();
  for (int i = 0; i < (int)NUM_LETRAS; i++) {
    JsonObject letraObj = jLetras[letras[i].nombre].to<JsonObject>();

    JsonArray posArr = letraObj["pos"].to<JsonArray>();
    for (int j = 0; j < 6; j++) posArr.add(letras[i].pos[j]);

    // pre_thumb: array of finger keys that close before thumb
    JsonArray ptArr = letraObj["pre_thumb"].to<JsonArray>();
    for (int b = 0; b < 4; b++)
      if (letras[i].preThumbMask & (1 << b))
        ptArr.add(FINGER_KEYS[b]);

    // Movements
    JsonArray movArr = letraObj["movements"].to<JsonArray>();
    for (int m = 0; m < letras[i].numMovements && m < MAX_MOVEMENTS; m++) {
      JsonObject movObj = movArr.add<JsonObject>();
      movObj["finger"] = FINGER_KEYS[letras[i].movements[m].fingerIdx];
      movObj["to"] = letras[i].movements[m].toAngle;
      movObj["repeat"] = letras[i].movements[m].repeat;
      movObj["delay_ms"] = letras[i].movements[m].delayMs;
    }
  }

  String output = "CONFIG:";
  serializeJson(doc, output);
  webSocket.broadcastTXT(output);
  wsLog("Config enviada (%d bytes)", output.length());
}

void moverDedoIndividual(const char *finger, int angulo) {
  int fi = -1;
  for (int i = 0; i < 6; i++)
    if (strcmp(FINGER_KEYS[i], finger) == 0) { fi = i; break; }
  if (fi < 0) { wsLog("Dedo desconocido: %s", finger); return; }

  angulo = constrain(angulo, limits[fi].minVal, limits[fi].maxVal);
  moverServo(fingerPins[fi], *fingerPos[fi], angulo);
  *fingerPos[fi] = angulo;
  wsLog("Dedo %s -> %d", finger, angulo);
}

// ── Embedded HTML interface ──────────────────────────────────────────

const char HTML_PAGE[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Manito - LSC</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#eaeaea;
       min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:16px}
  h1{font-size:1.8em;margin:8px 0 2px}
  .sub{color:#8a8a9a;font-size:.85em;margin-bottom:12px}
  .status{font-size:.9em;margin-bottom:10px}
  .status.ok{color:#2ecc71} .status.err{color:#e74c3c} .status.wait{color:#f5a623}

  .section-title{color:#8a8a9a;font-size:.75em;font-weight:bold;letter-spacing:1px;
                  margin:14px 0 6px;text-transform:uppercase}

  /* Text input */
  .text-input{display:flex;gap:6px;width:100%;max-width:500px;margin:4px 0}
  .text-input input{flex:1;padding:10px;border:none;border-radius:8px;background:#0f3460;
                    color:#eaeaea;font-family:'Consolas',monospace;font-size:1em;outline:none}
  .text-input button{padding:10px 14px;border:none;border-radius:8px;color:#fff;font-weight:bold;
                     font-size:.9em;cursor:pointer;transition:transform .1s}
  .text-input button:active{transform:scale(.93)}
  .btn-send{background:#3498db} .btn-cancel{background:#e74c3c}
  .btn-cancel:disabled{opacity:.4;cursor:default}
  .queue-status{font-size:.8em;color:#f5a623;margin:4px 0;min-height:1.2em}

  /* Letter grid */
  .letter-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;max-width:420px;margin:0 auto}
  .letter-grid button{padding:10px 0;border:none;border-radius:8px;color:#fff;
                     font-size:1em;font-weight:bold;cursor:pointer;transition:transform .1s,filter .1s}
  .letter-grid button:active{transform:scale(.92)}
  .letter-grid button:hover{filter:brightness(1.3)}

  .controls{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin:8px 0}
  .controls button{padding:10px 18px;border:none;border-radius:10px;color:#fff;font-size:1em;
                   font-weight:600;cursor:pointer;transition:transform .1s}
  .controls button:active{transform:scale(.93)}
  .btn-open{background:#2ecc71} .btn-close{background:#e74c3c}

  .divider{width:80%;height:1px;background:#0f3460;margin:10px auto}

  .finger-panel{width:100%;max-width:500px;background:#16213e;border-radius:12px;padding:14px;margin:6px 0}
  .finger-row{display:flex;align-items:center;gap:8px;margin:6px 0}
  .finger-row label{width:110px;font-size:.85em;color:#8a8a9a}
  .finger-row input[type=range]{flex:1;accent-color:#e94560}
  .finger-row .val{width:36px;text-align:center;font-family:'Consolas',monospace;font-size:.9em;
                    color:#e94560;font-weight:bold}

  .calib-panel{width:100%;max-width:500px;background:#16213e;border-radius:12px;padding:14px;margin:6px 0}
  .letter-sel{display:flex;flex-wrap:wrap;gap:4px;justify-content:center;margin:8px 0}
  .letter-sel button{width:32px;height:32px;border:none;border-radius:6px;background:#0f3460;
                     color:#eaeaea;font-weight:bold;font-size:.85em;cursor:pointer;transition:background .15s}
  .letter-sel button.active{background:#e94560;color:#fff}
  .letter-sel button:hover{filter:brightness(1.3)}
  .calib-row{display:flex;align-items:center;gap:6px;margin:5px 0}
  .calib-row label{width:110px;font-size:.85em;color:#8a8a9a}
  .calib-row .val{width:36px;text-align:center;font-family:'Consolas',monospace;font-size:.9em;
                   color:#f5a623;font-weight:bold}
  .calib-row input[type=range]{flex:1;accent-color:#f5a623}
  .pm-btn{width:30px;height:30px;border:none;border-radius:6px;font-size:1.1em;font-weight:bold;
          cursor:pointer;color:#fff;transition:transform .1s}
  .pm-btn:active{transform:scale(.9)}
  .pm-minus{background:#e74c3c} .pm-plus{background:#2ecc71}
  .calib-btns{display:flex;gap:8px;justify-content:center;margin-top:10px}
  .calib-btns button{padding:8px 16px;border:none;border-radius:8px;color:#fff;font-weight:600;
                     font-size:.9em;cursor:pointer;transition:transform .1s}
  .calib-btns button:active{transform:scale(.93)}
  .btn-preview{background:#3498db} .btn-save{background:#27ae60}

  .log-wrap{width:100%;max-width:500px}
  #log{width:100%;height:100px;background:#0d1117;color:#58a6ff;border:none;border-radius:8px;
       padding:8px;font-family:'Consolas',monospace;font-size:.8em;resize:none;outline:none}
  .log-btn{margin-top:6px;padding:6px 14px;border:none;border-radius:6px;background:#0f3460;
           color:#8a8a9a;font-size:.8em;cursor:pointer}
</style>
</head>
<body>
<h1>Manito</h1>
<p class="sub">Lengua de Senas Colombiana</p>
<p id="status" class="status wait">* Conectando...</p>

<p class="section-title">Escribir Texto</p>
<div class="text-input">
  <input type="text" id="textIn" placeholder="Escribe aqui..." autocomplete="off">
  <button class="btn-send" onclick="sendText()">Enviar</button>
  <button class="btn-cancel" id="btnCancel" disabled onclick="cancelQueue()">CANCELAR</button>
</div>
<p class="queue-status" id="queueStatus"></p>

<div class="divider"></div>
<p class="section-title">Abecedario LSC</p>
<div class="letter-grid" id="letterGrid"></div>

<div class="divider"></div>
<p class="section-title">Controles</p>
<div class="controls">
  <button class="btn-open" onclick="send('reset')">[+] Abrir</button>
  <button class="btn-close" onclick="send('cerrar')">[x] Cerrar</button>
</div>

<div class="divider"></div>
<p class="section-title">Probar Dedo Individual</p>
<div class="finger-panel" id="fingerPanel"></div>

<div class="divider"></div>
<p class="section-title">Calibrar Letra</p>
<div class="calib-panel">
  <div class="letter-sel" id="letterSel"></div>
  <div id="calibSliders"></div>
  <div class="calib-btns">
    <button class="btn-preview" onclick="previewCalib()">>> Probar</button>
    <button class="btn-save" onclick="saveCalib()">[Save] Guardar</button>
  </div>
</div>

<div class="divider"></div>
<p class="section-title">Serial Monitor</p>
<div class="log-wrap">
  <textarea id="log" readonly></textarea>
  <button class="log-btn" onclick="document.getElementById('log').value=''">Limpiar</button>
  <button class="log-btn" style="background:#27ae60;color:#fff;margin-left:6px" onclick="downloadConfig()">Descargar Config</button>
</div>

<script>
const FINGERS = ['idx','med','anu','men','pd','pp'];
const FINGER_NAMES = ['Indice','Medio','Anular','Menique','Pulgar D','Pulgar P'];
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').concat('N~');
const LETTERS_FULL = 'ABCDEFGHIJKLMN'.split('').concat(['\u00d1']).concat('OPQRSTUVWXYZ'.split(''));
const MOV = new Set(['J','\u00d1','Z']);
const VOWELS = {A:'#e94560',E:'#f5a623',I:'#7ed321',O:'#4a90d9',U:'#9b59b6'};

let settings = {delay_between_fingers_ms:50, delay_between_letters_ms:1500, delay_before_movements_ms:200};
let letterData = {};
let lastConfigJSON = null;
const DEFAULTS = {
  a:[180,180,180,180,60,0], b:[0,0,0,0,120,50], c:[90,90,90,90,45,0],
  d:[0,180,180,180,60,25], e:[90,90,95,90,0,37], f:[120,0,0,0,90,30],
  g:[0,180,180,180,0,0], h:[0,0,180,180,120,50], i:[180,180,180,0,60,25],
  j:[180,180,180,0,60,25], k:[0,0,180,180,45,25], l:[0,180,180,180,0,0],
  m:[180,180,180,180,120,50], n:[180,180,180,180,120,40], '\u00f1':[180,180,180,180,120,40],
  o:[120,120,120,120,60,25], p:[0,0,180,180,45,0], q:[180,180,180,180,0,0],
  r:[0,0,180,180,120,50], s:[180,180,180,180,90,50], t:[180,180,180,180,60,30],
  u:[0,0,180,180,60,25], v:[0,0,180,180,120,50], w:[0,0,0,180,120,50],
  x:[90,180,180,180,60,25], y:[180,180,180,0,0,0], z:[0,180,180,180,120,50]
};
for (const k in DEFAULTS) letterData[k] = [...DEFAULTS[k]];

// -- Letter grid (all letters) --
const lg = document.getElementById('letterGrid');
LETTERS_FULL.forEach(c => {
  const b = document.createElement('button');
  const isMov = MOV.has(c);
  b.textContent = isMov ? '~'+c : c;
  const bg = isMov ? '#1abc9c' : (VOWELS[c] || '#2c3e50');
  b.style.background = bg;
  b.onclick = () => enqueue(c === '\u00d1' ? '\u00f1' : c.toLowerCase());
  lg.appendChild(b);
});

// -- Text input --
document.getElementById('textIn').addEventListener('keydown', e => { if(e.key==='Enter') sendText(); });

// ── Finger test panel ──
const fp = document.getElementById('fingerPanel');
FINGERS.forEach((f, i) => {
  const row = document.createElement('div');
  row.className = 'finger-row';
  const lbl = document.createElement('label');
  lbl.textContent = FINGER_NAMES[i];
  const slider = document.createElement('input');
  slider.type = 'range'; slider.min = 0; slider.max = 180; slider.value = 0;
  slider.id = 'ft_' + f;
  const val = document.createElement('span');
  val.className = 'val'; val.textContent = '0'; val.id = 'ftv_' + f;
  slider.oninput = () => {
    val.textContent = slider.value;
    send('movefinger|' + f + '|' + slider.value);
  };
  row.appendChild(lbl); row.appendChild(slider); row.appendChild(val);
  fp.appendChild(row);
});

// ── Calibration panel ──
let currentLetter = 'a';
const ls = document.getElementById('letterSel');
LETTERS.forEach(l => {
  const b = document.createElement('button');
  b.textContent = l;
  b.id = 'ls_' + (l === 'Ñ' ? 'nn' : l.toLowerCase());
  b.onclick = () => selectLetter(l === 'Ñ' ? 'ñ' : l.toLowerCase());
  if (l === 'A') b.className = 'active';
  ls.appendChild(b);
});

const cs = document.getElementById('calibSliders');
FINGERS.forEach((f, i) => {
  const row = document.createElement('div');
  row.className = 'calib-row';
  const lbl = document.createElement('label');
  lbl.textContent = FINGER_NAMES[i];

  const minus = document.createElement('button');
  minus.className = 'pm-btn pm-minus'; minus.textContent = '−';
  minus.onclick = () => adjustCalib(f, -5);

  const slider = document.createElement('input');
  slider.type = 'range'; slider.min = 0; slider.max = 180;
  slider.value = letterData[currentLetter][i];
  slider.id = 'cs_' + f;
  const val = document.createElement('span');
  val.className = 'val'; val.textContent = slider.value; val.id = 'csv_' + f;
  slider.oninput = () => {
    val.textContent = slider.value;
    letterData[currentLetter][i] = parseInt(slider.value);
  };

  const plus = document.createElement('button');
  plus.className = 'pm-btn pm-plus'; plus.textContent = '+';
  plus.onclick = () => adjustCalib(f, 5);

  row.appendChild(lbl); row.appendChild(minus); row.appendChild(slider);
  row.appendChild(val); row.appendChild(plus);
  cs.appendChild(row);
});

function selectLetter(l) {
  currentLetter = l;
  document.querySelectorAll('.letter-sel button').forEach(b => b.classList.remove('active'));
  const id = l === 'ñ' ? 'nn' : l;
  const btn = document.getElementById('ls_' + id);
  if (btn) btn.classList.add('active');
  if (!letterData[l]) letterData[l] = [...DEFAULTS[l] || [0,0,0,0,0,0]];
  FINGERS.forEach((f, i) => {
    const s = document.getElementById('cs_' + f);
    const v = document.getElementById('csv_' + f);
    s.value = letterData[l][i]; v.textContent = letterData[l][i];
  });
}

function adjustCalib(finger, delta) {
  const i = FINGERS.indexOf(finger);
  let v = letterData[currentLetter][i] + delta;
  v = Math.max(0, Math.min(180, v));
  letterData[currentLetter][i] = v;
  document.getElementById('cs_' + finger).value = v;
  document.getElementById('csv_' + finger).textContent = v;
}

function previewCalib() {
  const vals = letterData[currentLetter].join(',');
  send('preview|' + vals);
}

function saveCalib() {
  const l = currentLetter === 'ñ' ? 'ñ' : currentLetter;
  const vals = letterData[currentLetter].join(',');
  send('setletra|' + l + '|' + vals);
  addLog('[Save] ' + currentLetter.toUpperCase() + ' guardada');
}

// ── Letter Queue ──
let queue = [];
let queueActive = false;
let queueCancelled = false;
const qsEl = document.getElementById('queueStatus');
const cancelBtn = document.getElementById('btnCancel');
const validLetters = new Set('abcdefghijklmnopqrstuvwxyz\u00f1'.split(''));

function enqueue(letter) {
  queue.push(letter);
  updateQueueUI();
  if (!queueActive) processQueue();
}

function sendText() {
  const inp = document.getElementById('textIn');
  const txt = inp.value.trim().toLowerCase();
  inp.value = '';
  if (!txt) return;
  for (const ch of txt) {
    if (validLetters.has(ch)) queue.push(ch);
    // map n~ to \u00f1
    if (ch === '~' && queue.length > 0 && queue[queue.length-1] === 'n') {
      queue[queue.length-1] = '\u00f1';
    }
  }
  updateQueueUI();
  if (!queueActive) processQueue();
}

function cancelQueue() {
  queue = [];
  queueCancelled = true;
  updateQueueUI();
  addLog('>>> Cola cancelada');
}

function processQueue() {
  if (!queue.length || queueCancelled) {
    queueActive = false;
    queueCancelled = false;
    cancelBtn.disabled = true;
    updateQueueUI();
    return;
  }
  queueActive = true;
  cancelBtn.disabled = false;
  const letter = queue.shift();
  updateQueueUI();
  send(letter);
  setTimeout(processQueue, settings.delay_between_letters_ms);
}

function updateQueueUI() {
  if (queue.length) {
    const preview = queue.slice(0, 20).map(l => l.toUpperCase()).join(' ');
    qsEl.textContent = 'Cola: ' + preview + (queue.length > 20 ? ' ...' : '');
  } else {
    qsEl.textContent = '';
  }
}

// ── WebSocket ──
let ws;
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');

function addLog(t) { logEl.value += t + '\n'; logEl.scrollTop = logEl.scrollHeight; }

function connect() {
  statusEl.textContent = '● Conectando...';
  statusEl.className = 'status wait';
  ws = new WebSocket('ws://' + location.hostname + ':81');
  ws.onopen = () => {
    statusEl.textContent = '● Conectado';
    statusEl.className = 'status ok';
    addLog('Conectado');
    send('getconfig');
  };
  ws.onclose = () => {
    statusEl.textContent = '● Desconectado';
    statusEl.className = 'status err';
    addLog('Desconectado');
    setTimeout(connect, 3000);
  };
  ws.onerror = () => { ws.close(); };
  ws.onmessage = e => {
    if (e.data.startsWith('CONFIG:')) {
      try {
        const cfg = JSON.parse(e.data.substring(7));
        lastConfigJSON = cfg;
        if (cfg.settings) Object.assign(settings, cfg.settings);
        if (cfg.letras) {
          for (const k in cfg.letras) {
            if (cfg.letras[k].pos) letterData[k] = [...cfg.letras[k].pos];
            else if (Array.isArray(cfg.letras[k])) letterData[k] = [...cfg.letras[k]];
          }
          selectLetter(currentLetter);
        }
        addLog('[Config sincronizada]');
      } catch(ex) { addLog('[Config error: '+ex+']'); }
    } else {
      addLog('ESP32> ' + e.data);
    }
  };
}

function send(cmd) {
  if (!ws || ws.readyState !== 1) { alert('Sin conexión'); return; }
  ws.send(cmd);
  if (!cmd.startsWith('movefinger') && !cmd.startsWith('getconfig'))
    addLog('>>> ' + cmd.toUpperCase());
}

function downloadConfig() {
  if (!lastConfigJSON) { alert('Aún no se ha recibido config del ESP32'); return; }
  const out = {settings: lastConfigJSON.settings, limits: lastConfigJSON.limits, letters: lastConfigJSON.letras};
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'config.json';
  a.click();
  URL.revokeObjectURL(a.href);
  addLog('[Config descargada como config.json]');
}

connect();
</script>
</body>
</html>
)rawliteral";

// ── Command handler ──────────────────────────────────────────────────

void procesarComando(String msg) {
  msg.trim();
  if (msg.length() == 0)
    return;

  wsLog("CMD: %s", msg.c_str());

  // Single letter commands (a-z, ñ)
  if (msg.length() == 1 && msg[0] >= 'a' && msg[0] <= 'z') {
    hacerLetra(msg.c_str());
    return;
  }

  // Handle UTF-8 "ñ" (2 bytes: 0xC3 0xB1)
  if (msg.length() == 2 && (uint8_t)msg[0] == 0xC3 && (uint8_t)msg[1] == 0xB1) {
    hacerLetra("ñ");
    return;
  }

  // Named commands
  if (msg == "reset") {
    resetMano();
    return;
  }
  if (msg == "cerrar") {
    cerrarMano();
    return;
  }
  // movefinger|finger|angle - move a single finger
  if (msg.startsWith("movefinger|")) {
    String rest = msg.substring(11);
    int sep = rest.indexOf('|');
    if (sep > 0) {
      String finger = rest.substring(0, sep);
      int angle = rest.substring(sep + 1).toInt();
      moverDedoIndividual(finger.c_str(), angle);
    }
    return;
  }
  if (msg == "getconfig") {
    enviarConfigJSON();
    return;
  }

  // preview|v1,v2,v3,v4,v5,v6
  if (msg.startsWith("preview|")) {
    String vals = msg.substring(8);
    int v[6] = {0};
    int idx = 0;
    while (vals.length() > 0 && idx < 6) {
      int comma = vals.indexOf(',');
      if (comma < 0) {
        v[idx++] = vals.toInt();
        break;
      }
      v[idx++] = vals.substring(0, comma).toInt();
      vals = vals.substring(comma + 1);
    }
    if (idx == 6) {
      wsLog("Preview: %d,%d,%d,%d,%d,%d", v[0], v[1], v[2], v[3], v[4], v[5]);
      ponerPosicion(v[0], v[1], v[2], v[3], v[4], v[5]);
    }
    return;
  }

  // setletra|nombre|v1,v2,v3,v4,v5,v6
  if (msg.startsWith("setletra|")) {
    int sep1 = msg.indexOf('|', 9);
    if (sep1 < 0)
      return;
    String nombre = msg.substring(9, sep1);
    String vals = msg.substring(sep1 + 1);
    int letraIdx = findLetra(nombre.c_str());
    if (letraIdx < 0) {
      wsLog("Letra desconocida para setletra: %s", nombre.c_str());
      return;
    }
    int v[6] = {0};
    int vi = 0;
    while (vals.length() > 0 && vi < 6) {
      int comma = vals.indexOf(',');
      if (comma < 0) {
        v[vi++] = vals.toInt();
        break;
      }
      v[vi++] = vals.substring(0, comma).toInt();
      vals = vals.substring(comma + 1);
    }
    if (vi == 6) {
      for (int j = 0; j < 6; j++)
        letras[letraIdx].pos[j] = v[j];
      guardarConfig();
      wsLog("Letra %s guardada: %d,%d,%d,%d,%d,%d", letras[letraIdx].nombre,
            v[0], v[1], v[2], v[3], v[4], v[5]);
    }
    return;
  }

  // setlimits|finger|min|max
  if (msg.startsWith("setlimits|")) {
    String rest = msg.substring(10);
    int s1 = rest.indexOf('|');
    int s2 = rest.indexOf('|', s1 + 1);
    if (s1 < 0 || s2 < 0)
      return;
    String finger = rest.substring(0, s1);
    int mn = rest.substring(s1 + 1, s2).toInt();
    int mx = rest.substring(s2 + 1).toInt();
    for (int i = 0; i < 6; i++) {
      if (finger == FINGER_KEYS[i]) {
        limits[i].minVal = mn;
        limits[i].maxVal = mx;
        wsLog("Limites %s: %d-%d", FINGER_KEYS[i], mn, mx);
        break;
      }
    }
    guardarConfig();
    return;
  }

  // setsettings|delay_fingers|delay_letters|delay_movements
  if (msg.startsWith("setsettings|")) {
    String rest = msg.substring(12);
    int s1 = rest.indexOf('|');
    int s2 = rest.indexOf('|', s1 + 1);
    if (s1 > 0 && s2 > 0) {
      delayBetweenFingers = rest.substring(0, s1).toInt();
      delayBetweenLetters = rest.substring(s1 + 1, s2).toInt();
      delayBeforeMovements = rest.substring(s2 + 1).toInt();
      guardarConfig();
      wsLog("Settings: fingers=%dms letters=%dms movements=%dms",
            delayBetweenFingers, delayBetweenLetters, delayBeforeMovements);
    }
    return;
  }

  // setmovements|letra|fingerIdx,toAngle,repeat,delayMs;fingerIdx,toAngle,repeat,delayMs;...
  if (msg.startsWith("setmovements|")) {
    int sep1 = msg.indexOf('|', 13);
    if (sep1 < 0) return;
    String nombre = msg.substring(13, sep1);
    String data = msg.substring(sep1 + 1);
    int letraIdx = findLetra(nombre.c_str());
    if (letraIdx < 0) {
      wsLog("Letra desconocida para setmovements: %s", nombre.c_str());
      return;
    }

    // Clear existing movements
    letras[letraIdx].numMovements = 0;

    // Parse: "fi,to,rep,del;fi,to,rep,del;..." (empty string = no movements)
    if (data.length() > 0) {
      int mi = 0;
      while (data.length() > 0 && mi < MAX_MOVEMENTS) {
        int semi = data.indexOf(';');
        String chunk = (semi < 0) ? data : data.substring(0, semi);

        // Parse "fingerIdx,toAngle,repeat,delayMs"
        int vals[4] = {0};
        int vi = 0;
        while (chunk.length() > 0 && vi < 4) {
          int comma = chunk.indexOf(',');
          if (comma < 0) { vals[vi++] = chunk.toInt(); break; }
          vals[vi++] = chunk.substring(0, comma).toInt();
          chunk = chunk.substring(comma + 1);
        }
        if (vi == 4) {
          letras[letraIdx].movements[mi].fingerIdx = vals[0];
          letras[letraIdx].movements[mi].toAngle   = vals[1];
          letras[letraIdx].movements[mi].repeat    = vals[2];
          letras[letraIdx].movements[mi].delayMs   = vals[3];
          mi++;
        }

        if (semi < 0) break;
        data = data.substring(semi + 1);
      }
      letras[letraIdx].numMovements = mi;
    }

    guardarConfig();
    wsLog("Movimientos para %s actualizados: %d", letras[letraIdx].nombre,
          letras[letraIdx].numMovements);
    return;
  }

  // setprethumb|letra|idx,men  (finger keys that close before thumb)
  if (msg.startsWith("setprethumb|")) {
    int sep1 = msg.indexOf('|', 12);
    if (sep1 < 0) return;
    String nombre = msg.substring(12, sep1);
    String data = msg.substring(sep1 + 1);
    int letraIdx = findLetra(nombre.c_str());
    if (letraIdx < 0) return;

    uint8_t mask = 0;
    while (data.length() > 0) {
      int comma = data.indexOf(',');
      String key = (comma < 0) ? data : data.substring(0, comma);
      key.trim();
      for (int i = 0; i < 4; i++)
        if (key == FINGER_KEYS[i]) mask |= (1 << i);
      if (comma < 0) break;
      data = data.substring(comma + 1);
    }
    letras[letraIdx].preThumbMask = mask;
    guardarConfig();
    wsLog("PreThumb %s = 0x%02X", letras[letraIdx].nombre, mask);
    return;
  }

  // pushconfig|{json} - receive full config from PC (source of truth)
  if (msg.startsWith("pushconfig|")) {
    String jsonStr = msg.substring(11);
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, jsonStr);
    if (err) { wsLog("pushconfig JSON error: %s", err.c_str()); return; }

    if (doc.containsKey("settings")) {
      JsonObject s = doc["settings"];
      delayBetweenFingers  = s["delay_between_fingers_ms"]  | delayBetweenFingers;
      delayBetweenLetters  = s["delay_between_letters_ms"]  | delayBetweenLetters;
      delayBeforeMovements = s["delay_before_movements_ms"] | delayBeforeMovements;
    }
    if (doc.containsKey("limits")) {
      JsonObject lim = doc["limits"];
      for (int i = 0; i < 6; i++) {
        if (lim.containsKey(FINGER_KEYS[i])) {
          limits[i].minVal = lim[FINGER_KEYS[i]]["min"] | limits[i].minVal;
          limits[i].maxVal = lim[FINGER_KEYS[i]]["max"] | limits[i].maxVal;
        }
      }
    }
    if (doc.containsKey("letters")) {
      JsonObject letters = doc["letters"];
      for (int i = 0; i < (int)NUM_LETRAS; i++) {
        const char *name = letras[i].nombre;
        if (!letters.containsKey(name)) continue;
        JsonObject lObj = letters[name];

        if (lObj.containsKey("pos")) {
          JsonArray posArr = lObj["pos"];
          for (int j = 0; j < 6 && j < (int)posArr.size(); j++)
            letras[i].pos[j] = posArr[j];
        }

        // pre_thumb: ["men", "anu"]
        letras[i].preThumbMask = 0;
        if (lObj.containsKey("pre_thumb")) {
          JsonArray ptArr = lObj["pre_thumb"];
          for (int p = 0; p < (int)ptArr.size(); p++) {
            const char *fk = ptArr[p];
            for (int b = 0; b < 4; b++)
              if (strcmp(fk, FINGER_KEYS[b]) == 0) letras[i].preThumbMask |= (1 << b);
          }
        }

        letras[i].numMovements = 0;
        if (lObj.containsKey("movements")) {
          JsonArray movArr = lObj["movements"];
          for (int m = 0; m < (int)movArr.size() && m < MAX_MOVEMENTS; m++) {
            JsonObject mv = movArr[m];
            const char *fk = mv["finger"] | "idx";
            int fi = 0;
            for (int b = 0; b < 6; b++)
              if (strcmp(fk, FINGER_KEYS[b]) == 0) { fi = b; break; }
            letras[i].movements[m].fingerIdx = fi;
            letras[i].movements[m].toAngle   = mv["to"] | 0;
            letras[i].movements[m].repeat    = mv["repeat"] | 1;
            letras[i].movements[m].delayMs   = mv["delay_ms"] | 150;
            letras[i].numMovements++;
          }
        }
      }
    }

    guardarConfig();
    wsLog("pushconfig OK");
    enviarConfigJSON();
    return;
  }

  wsLog("Comando desconocido: %s", msg.c_str());
}

// ── WebSocket events ─────────────────────────────────────────────────

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t *payload,
                      size_t length) {
  switch (type) {
  case WStype_CONNECTED:
    wsLog("[WS] Cliente %u conectado", num);
    break;
  case WStype_DISCONNECTED:
    wsLog("[WS] Cliente %u desconectado", num);
    break;
  case WStype_TEXT: {
    if (length > 0) {
      char buf[length + 1];
      memcpy(buf, payload, length);
      buf[length] = '\0';
      procesarComando(String(buf));
    }
    break;
  }
  default:
    break;
  }
}

// ── HTTP handler ─────────────────────────────────────────────────────

void handleRoot() { server.send_P(200, "text/html", HTML_PAGE); }
void handleCaptivePortal() { server.sendHeader("Location", "http://192.168.4.1/"); server.send(302); }

// ── Setup & Loop ─────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);

  // WiFiManager
  WiFiManager wm;
  // wm.resetSettings(); // Descomentar para borrar redes guardadas
  
  bool res = wm.autoConnect("Manito-LSC", "asereje78");
  if (!res) {
    Serial.println("Fallo conexion WiFi o se alcanzo timeout");
  } else {
    Serial.printf("\nConectado! IP: %s\n", WiFi.localIP().toString().c_str());
  }
  
  // Iniciar mDNS
  if (!MDNS.begin("manito")) {
    Serial.println("Error configurando mDNS!");
  } else {
    Serial.println("mDNS iniciado: manito.local");
  }

  // Initialize servos with LEDC
  Serial.println("Inicializando servos...");
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.printf("  Servo %d: GPIO %d attach...", i, servoPins[i]);
    ledcAttach(servoPins[i], SERVO_FREQ, SERVO_RES);
    Serial.print(" write...");
    servoWrite(servoPins[i], 0);
    Serial.println(" OK");
    delay(200);
  }
  Serial.println("Servos OK");

  // Load saved config from flash
  Serial.println("Cargando config...");
  cargarConfig();
  Serial.println("Config OK");

  // Initial position: open hand
  Serial.println("Reset mano...");
  resetMano();
  Serial.println("Reset OK");

  // OTA
  ArduinoOTA.setHostname("manito-esp32c6");
  ArduinoOTA.begin();

  // HTTP server (serves HTML interface)
  server.on("/", handleRoot);
  server.begin();
  Serial.println("HTTP server en puerto 80");

  // WebSocket server
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  _wsReady = true;
  wsLog("WebSocket listo en puerto 81");
  wsLog("Abrir http://manito.local o http://%s en el navegador",
    WiFi.localIP().toString().c_str());
}

void loop() {
  ArduinoOTA.handle();
  server.handleClient();
  webSocket.loop();

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    procesarComando(cmd);
  }
}
