#include <WiFi.h>
#include <ArduinoOTA.h>
#include <WebSocketsServer.h>

const char* ssid     = "holi:3";
const char* password = "Sayejoda78";

WebSocketsServer webSocket = WebSocketsServer(81);

// Pines de los servos
// GPIO seguros en ESP32-C6 (evitar 0,9=strapping/BOOT; 18,19=USB; 16,17=UART)
#define PIN_PULGAR_PALMA   3
#define PIN_PULGAR_DEDO    2
#define PIN_INDICE         23
#define PIN_MEDIO          22
#define PIN_ANULAR         6
#define PIN_MENIQUE        7

// PWM config para servos: 50Hz, 16 bits de resolución
#define SERVO_FREQ       50
#define SERVO_RES        16
// Pulso mínimo y máximo en us (0° y 180°)
#define SERVO_MIN_US     500
#define SERVO_MAX_US     2500
// Duty cycle correspondiente (resolución 16 bits = 65536, periodo 20000us)
#define SERVO_MIN_DUTY   ((SERVO_MIN_US * 65536L) / 20000)  // ~1638
#define SERVO_MAX_DUTY   ((SERVO_MAX_US * 65536L) / 20000)  // ~8192

// Pines de servo en array para facilitar operaciones
const int servoPins[] = {
    PIN_PULGAR_PALMA, PIN_PULGAR_DEDO,
    PIN_INDICE, PIN_MEDIO, PIN_ANULAR, PIN_MENIQUE
};
#define NUM_SERVOS 6

// Rangos máximos por dedo (el pulgar y anular recorren menos)
#define MAX_PULGAR_PALMA  90
#define MAX_PULGAR_DEDO   180
#define MAX_INDICE        190
#define MAX_MEDIO         180
#define MAX_ANULAR        180
#define MAX_MENIQUE       180

// Tope real del hardware del servo (nunca escribir más de esto)
#define SERVO_MIN 0
#define SERVO_MAX 180

// Posición actual de cada servo
int posPulgarPalma = 0;
int posPulgarDedo  = 0;
int posIndice      = 0;
int posMedio       = 0;
int posAnular      = 0;
int posMenique     = 0;

// ── Log inalámbrico: Serial + WebSocket ──

char _logBuf[256];

// Envía un mensaje tanto a Serial como a todos los clientes WebSocket
void wsLog(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(_logBuf, sizeof(_logBuf), fmt, args);
    va_end(args);

    Serial.println(_logBuf);
    webSocket.broadcastTXT(_logBuf);
}

// Convierte ángulo (0-180) a duty cycle LEDC y escribe al pin
void servoWrite(int pin, int angulo) {
    angulo = constrain(angulo, SERVO_MIN, SERVO_MAX);
    uint32_t duty = SERVO_MIN_DUTY + (long)(SERVO_MAX_DUTY - SERVO_MIN_DUTY) * angulo / 180;
    ledcWrite(pin, duty);
}

// Escribe un valor seguro al servo (clampeado a 0-180)
void servoWriteSafe(int pin, int pos) {
    servoWrite(pin, constrain(pos, SERVO_MIN, SERVO_MAX));
}

// Mueve un único servo suavemente de 'desde' hasta 'hasta'
void moverServo(int pin, int desde, int hasta) {
    desde = constrain(desde, SERVO_MIN, SERVO_MAX);
    hasta = constrain(hasta, SERVO_MIN, SERVO_MAX);
    int diff = abs(hasta - desde);
    if (diff == 0) return;
    for (int step = 1; step <= diff; step++) {
        int pos = desde + (long)(hasta - desde) * step / diff;
        servoWriteSafe(pin, pos);
        delay(5);
    }
    servoWriteSafe(pin, hasta);
}

// Mueve un grupo de servos en paralelo hasta sus targets
void moverParalelo(int pins[], int starts[], int targets[], int n) {
    int maxDiff = 0;
    for (int i = 0; i < n; i++) {
        starts[i]  = constrain(starts[i],  SERVO_MIN, SERVO_MAX);
        targets[i] = constrain(targets[i], SERVO_MIN, SERVO_MAX);
        int d = abs(targets[i] - starts[i]);
        if (d > maxDiff) maxDiff = d;
    }
    if (maxDiff == 0) return;
    for (int step = 1; step <= maxDiff; step++) {
        for (int i = 0; i < n; i++) {
            int pos = starts[i] + (long)(targets[i] - starts[i]) * step / maxDiff;
            servoWriteSafe(pins[i], pos);
        }
        delay(5);
    }
    for (int i = 0; i < n; i++) servoWriteSafe(pins[i], targets[i]);
}

// Establece todos los dedos a posiciones específicas.
void ponerPosicion(int pp, int pd, int idx, int med, int anu, int men) {
    pp  = constrain(pp,  0, MAX_PULGAR_PALMA);
    pd  = constrain(pd,  0, MAX_PULGAR_DEDO);
    idx = constrain(idx, 0, MAX_INDICE);
    med = constrain(med, 0, MAX_MEDIO);
    anu = constrain(anu, 0, MAX_ANULAR);
    men = constrain(men, 0, MAX_MENIQUE);


    // 3. Resto de dedos secuencial
    moverServo(PIN_INDICE, posIndice, idx);
    posIndice = idx;
    delay(50);

    moverServo(PIN_MEDIO, posMedio, med);
    posMedio = med;
    delay(50);

    moverServo(PIN_ANULAR, posAnular, anu);
    posAnular = anu;
    delay(50);

    moverServo(PIN_MENIQUE, posMenique, men);
    posMenique = men;
    delay(50);

    // 1. Pulgar dedo (se recoge primero)
    moverServo(PIN_PULGAR_DEDO, posPulgarDedo, pd);
    posPulgarDedo = pd;

    // 2. Pulgar palma (gira después)
    moverServo(PIN_PULGAR_PALMA, posPulgarPalma, pp);
    posPulgarPalma = pp;

    wsLog("Pos: pp=%d pd=%d idx=%d med=%d anu=%d men=%d",
        pp, pd, idx, med, anu, men);
}

// ── Vocales en Lengua de Señas Colombiana (LSC) ──

void hacerLetraA() {
    wsLog("Sena: A");
    ponerPosicion(MAX_PULGAR_PALMA*3/4, 0, MAX_INDICE, MAX_MEDIO, MAX_ANULAR, MAX_MENIQUE);
}

void hacerLetraE() {
    wsLog("Sena: E");
    ponerPosicion(0, MAX_PULGAR_DEDO*4/5, MAX_INDICE/2, MAX_MEDIO/2, MAX_ANULAR/2, MAX_MENIQUE/2);
}

void hacerLetraI() {
    wsLog("Sena: I");
    ponerPosicion(MAX_PULGAR_PALMA/2, MAX_PULGAR_DEDO/2, MAX_INDICE, MAX_MEDIO, MAX_ANULAR, 0);
}

void hacerLetraO() {
    wsLog("Sena: O");
    ponerPosicion(MAX_PULGAR_PALMA/2, MAX_PULGAR_DEDO/2, MAX_INDICE*2/3, MAX_MEDIO*2/3, MAX_ANULAR*2/3, MAX_MENIQUE*2/3);
}

void hacerLetraU() {
    wsLog("Sena: U");
    ponerPosicion(MAX_PULGAR_PALMA/2, MAX_PULGAR_DEDO/2, 0, MAX_MEDIO, MAX_ANULAR, 0);
}

void resetMano() {
    wsLog("Reset: mano abierta");
    ponerPosicion(0, 0, 0, 0, 0, 0);
}

void cerrarMano() {
    wsLog("Cerrar: puno completo");
    ponerPosicion(MAX_PULGAR_PALMA, MAX_PULGAR_DEDO, MAX_INDICE, MAX_MEDIO, MAX_ANULAR, MAX_MENIQUE);
}

// ── Diagnóstico: prueba cada pin uno por uno ──

struct PinInfo {
    const char* nombre;
    int pin;
};

void testPines() {
    PinInfo pines[] = {
        {"PULGAR_PALMA", PIN_PULGAR_PALMA},
        {"PULGAR_DEDO",  PIN_PULGAR_DEDO},
        {"INDICE",       PIN_INDICE},
        {"MEDIO",        PIN_MEDIO},
        {"ANULAR",       PIN_ANULAR},
        {"MENIQUE",      PIN_MENIQUE},
    };
    int n = sizeof(pines) / sizeof(pines[0]);

    wsLog("== DIAGNOSTICO DE PINES ==");
    wsLog("Cada servo: 0 -> 90 -> 0");

    for (int i = 0; i < n; i++) {
        wsLog("[%d/%d] %s (GPIO %d)...", i+1, n, pines[i].nombre, pines[i].pin);

        servoWriteSafe(pines[i].pin, 0);
        delay(500);

        for (int pos = 0; pos <= 90; pos++) {
            servoWriteSafe(pines[i].pin, pos);
            delay(10);
        }
        wsLog("  -> en 90, esperando 1s");
        delay(1000);

        for (int pos = 90; pos >= 0; pos--) {
            servoWriteSafe(pines[i].pin, pos);
            delay(10);
        }
        delay(500);

        wsLog("  -> %s GPIO %d LISTO", pines[i].nombre, pines[i].pin);
    }

    wsLog("== DIAGNOSTICO COMPLETO ==");
    for (int i = 0; i < n; i++) {
        wsLog("  %s = GPIO %d", pines[i].nombre, pines[i].pin);
    }
}

// ── WebSocket + Serial command handler ──

void procesarComando(String msg) {
    msg.trim();
    msg.toLowerCase();
    if (msg.length() == 0) return;

    wsLog("CMD: %s", msg.c_str());

    if (msg == "a")           hacerLetraA();
    else if (msg == "e")      hacerLetraE();
    else if (msg == "i")      hacerLetraI();
    else if (msg == "o")      hacerLetraO();
    else if (msg == "u")      hacerLetraU();
    else if (msg == "reset")  resetMano();
    else if (msg == "cerrar") cerrarMano();
    else if (msg == "test")   testPines();
    else wsLog("Comando desconocido: %s", msg.c_str());
}

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            wsLog("[WS] Cliente %u conectado", num);
            break;
        case WStype_DISCONNECTED:
            wsLog("[WS] Cliente %u desconectado", num);
            break;
        case WStype_TEXT: {
            if (length > 0) {
                // Construcción segura del String desde payload
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

void setup() {
    Serial.begin(115200);

    WiFi.begin(ssid, password);
    Serial.print("Conectando a WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\nIP: %s\n", WiFi.localIP().toString().c_str());

    // Inicializar servos con LEDC
    for (int i = 0; i < NUM_SERVOS; i++) {
        ledcAttach(servoPins[i], SERVO_FREQ, SERVO_RES);
        servoWrite(servoPins[i], 0);
        delay(200);
    }

    // Posicion inicial: mano abierta
    resetMano();

    ArduinoOTA.setHostname("manito-esp32c6");
    ArduinoOTA.begin();

    webSocket.begin();
    webSocket.onEvent(onWebSocketEvent);
    wsLog("WebSocket listo en puerto 81");
}

void loop() {
    ArduinoOTA.handle();
    webSocket.loop();

    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        procesarComando(cmd);
    }
}
