#include <WiFi.h>
#include <ArduinoOTA.h>
#include <WebSocketsServer.h>
#include <ESP32Servo.h>

const char* ssid     = "M@STV-SARA-BELLA";
const char* password = "P4tac0n_cOn_Qu3s0";

WebSocketsServer webSocket = WebSocketsServer(81);

// Pines de los servos
// GPIO seguros en ESP32-C6 (evitar 0,9=strapping/BOOT; 18,19=USB; 16,17=UART)
#define PIN_PULGAR_PALMA   2
#define PIN_PULGAR_DEDO   13
#define PIN_INDICE        12
#define PIN_MEDIO         21
#define PIN_ANULAR         3
#define PIN_MENIQUE        0

Servo servoPulgarPalma;
Servo servoPulgarDedo;
Servo servoIndice;
Servo servoMedio;
Servo servoAnular;
Servo servoMenique;

// Rangos máximos por dedo (el pulgar y anular recorren menos)
#define MAX_PULGAR_PALMA  50
#define MAX_PULGAR_DEDO   120
#define MAX_INDICE        200
#define MAX_MEDIO         180
#define MAX_ANULAR        190
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

// Escribe un valor seguro al servo (clampeado a 0-180)
void servoWriteSafe(Servo& servo, int pos) {
    servo.write(constrain(pos, SERVO_MIN, SERVO_MAX));
}

// Mueve un único servo suavemente de 'desde' hasta 'hasta'
void moverServo(Servo& servo, int desde, int hasta) {
    desde = constrain(desde, SERVO_MIN, SERVO_MAX);
    hasta = constrain(hasta, SERVO_MIN, SERVO_MAX);
    int diff = abs(hasta - desde);
    if (diff == 0) return;
    for (int step = 1; step <= diff; step++) {
        int pos = desde + (long)(hasta - desde) * step / diff;
        servoWriteSafe(servo, pos);
        delay(5);
    }
    servoWriteSafe(servo, hasta);
}

// Mueve un grupo de servos en paralelo hasta sus targets
void moverParalelo(Servo* servos[], int starts[], int targets[], int n) {
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
            servoWriteSafe(*servos[i], pos);
        }
        delay(5);
    }
    for (int i = 0; i < n; i++) servoWriteSafe(*servos[i], targets[i]);
}

// Establece todos los dedos a posiciones específicas.
// Orden de movimiento: 1) pulgar dedo  2) pulgar palma  3) resto en paralelo
void ponerPosicion(int pp, int pd, int idx, int med, int anu, int men) {
    pp  = constrain(pp,  0, MAX_PULGAR_PALMA);
    pd  = constrain(pd,  0, MAX_PULGAR_DEDO);
    idx = constrain(idx, 0, MAX_INDICE);
    med = constrain(med, 0, MAX_MEDIO);
    anu = constrain(anu, 0, MAX_ANULAR);
    men = constrain(men, 0, MAX_MENIQUE);

    // 1. Pulgar dedo (se recoge primero)
    moverServo(servoPulgarDedo, posPulgarDedo, pd);
    posPulgarDedo = pd;

    // 2. Pulgar palma (gira después)
    moverServo(servoPulgarPalma, posPulgarPalma, pp);
    posPulgarPalma = pp;

    // 3. Resto de dedos en paralelo
    Servo* servos[4] = {&servoIndice, &servoMedio, &servoAnular, &servoMenique};
    int starts[4]    = {posIndice, posMedio, posAnular, posMenique};
    int targets[4]   = {idx, med, anu, men};
    moverParalelo(servos, starts, targets, 4);
    posIndice = idx;
    posMedio  = med;
    posAnular = anu;
    posMenique = men;

    wsLog("Pos: pp=%d pd=%d idx=%d med=%d anu=%d men=%d",
        pp, pd, idx, med, anu, men);
}

// ── Vocales en Lengua de Señas Colombiana (LSC) ──

void hacerLetraA() {
    wsLog("Sena: A");
    ponerPosicion(0, MAX_PULGAR_DEDO/2, MAX_INDICE, MAX_MEDIO, MAX_ANULAR, MAX_MENIQUE);
}

void hacerLetraE() {
    wsLog("Sena: E");
    ponerPosicion(MAX_PULGAR_PALMA*3/4, 0, MAX_INDICE/2, MAX_MEDIO/2, MAX_ANULAR/2, MAX_MENIQUE/2);
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
    ponerPosicion(MAX_PULGAR_PALMA/2, MAX_PULGAR_DEDO/2, 0, 0, MAX_ANULAR, MAX_MENIQUE);
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
    Servo* servo;
};

void testPines() {
    PinInfo pines[] = {
        {"PULGAR_PALMA", PIN_PULGAR_PALMA, &servoPulgarPalma},
        {"PULGAR_DEDO",  PIN_PULGAR_DEDO,  &servoPulgarDedo},
        {"INDICE",       PIN_INDICE,       &servoIndice},
        {"MEDIO",        PIN_MEDIO,        &servoMedio},
        {"ANULAR",       PIN_ANULAR,       &servoAnular},
        {"MENIQUE",      PIN_MENIQUE,      &servoMenique},
    };
    int n = sizeof(pines) / sizeof(pines[0]);

    wsLog("== DIAGNOSTICO DE PINES ==");
    wsLog("Cada servo: 0 -> 90 -> 0");

    for (int i = 0; i < n; i++) {
        wsLog("[%d/%d] %s (GPIO %d)...", i+1, n, pines[i].nombre, pines[i].pin);

        servoWriteSafe(*pines[i].servo, 0);
        delay(500);

        for (int pos = 0; pos <= 90; pos++) {
            servoWriteSafe(*pines[i].servo, pos);
            delay(10);
        }
        wsLog("  -> en 90, esperando 1s");
        delay(1000);

        for (int pos = 90; pos >= 0; pos--) {
            servoWriteSafe(*pines[i].servo, pos);
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

    // Inicializar servos
    servoPulgarPalma.attach(PIN_PULGAR_PALMA);
    servoPulgarDedo.attach(PIN_PULGAR_DEDO);
    servoIndice.attach(PIN_INDICE);
    servoMedio.attach(PIN_MEDIO);
    servoAnular.attach(PIN_ANULAR);
    servoMenique.attach(PIN_MENIQUE);

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