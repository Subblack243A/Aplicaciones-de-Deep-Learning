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
#define PIN_MEDIO         21   // era 9 (BOOT) → cambiado a 4
#define PIN_ANULAR         3
#define PIN_MENIQUE        0   // era 0 (strapping) → cambiado a 6

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

// Posición actual de cada servo
int posPulgarPalma = 0;
int posPulgarDedo  = 0;
int posIndice      = 0;
int posMedio       = 0;
int posAnular      = 0;
int posMenique     = 0;

// Mueve un único servo suavemente de 'desde' hasta 'hasta'
void moverServo(Servo& servo, int desde, int hasta) {
    int diff = abs(hasta - desde);
    if (diff == 0) return;
    for (int step = 1; step <= diff; step++) {
        int pos = desde + (long)(hasta - desde) * step / diff;
        servo.write(pos);
        delay(5);
    }
    servo.write(hasta);
}

// Mueve un grupo de servos en paralelo hasta sus targets
void moverParalelo(Servo* servos[], int starts[], int targets[], int n) {
    int maxDiff = 0;
    for (int i = 0; i < n; i++) {
        int d = abs(targets[i] - starts[i]);
        if (d > maxDiff) maxDiff = d;
    }
    if (maxDiff == 0) return;
    for (int step = 1; step <= maxDiff; step++) {
        for (int i = 0; i < n; i++) {
            int pos = starts[i] + (long)(targets[i] - starts[i]) * step / maxDiff;
            servos[i]->write(pos);
        }
        delay(5);
    }
    for (int i = 0; i < n; i++) servos[i]->write(targets[i]);
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

    Serial.printf("Pos: pp=%d pd=%d idx=%d med=%d anu=%d men=%d\n",
        pp, pd, idx, med, anu, men);
}

// ── Vocales en Lengua de Señas Colombiana (LSC) ──

void hacerLetraA() {
    // A: Puño cerrado, pulgar al lado (no cruza)
    Serial.println("Sena: A");
    ponerPosicion(
        0,                  // pulgar_palma: al lado
        MAX_PULGAR_DEDO/2,  // pulgar_dedo: ligeramente doblado
        MAX_INDICE,         // indice: cerrado
        MAX_MEDIO,          // medio: cerrado
        MAX_ANULAR,         // anular: cerrado
        MAX_MENIQUE         // menique: cerrado
    );
}

void hacerLetraE() {
    // E: Dedos doblados a la mitad, pulgar cruzado al frente
    Serial.println("Sena: E");
    ponerPosicion(
        MAX_PULGAR_PALMA * 3/4,  // pulgar_palma: cruza hacia adentro
        0,                        // pulgar_dedo: estirado
        MAX_INDICE / 2,           // indice: doblado a mitad
        MAX_MEDIO / 2,            // medio: doblado a mitad
        MAX_ANULAR / 2,           // anular: doblado a mitad
        MAX_MENIQUE / 2           // menique: doblado a mitad
    );
}

void hacerLetraI() {
    // I: Puño cerrado, meñique extendido
    Serial.println("Sena: I");
    ponerPosicion(
        MAX_PULGAR_PALMA / 2,  // pulgar_palma: cruza sobre puno
        MAX_PULGAR_DEDO / 2,   // pulgar_dedo: doblado
        MAX_INDICE,             // indice: cerrado
        MAX_MEDIO,              // medio: cerrado
        MAX_ANULAR,             // anular: cerrado
        0                       // menique: extendido
    );
}

void hacerLetraO() {
    // O: Todos los dedos curvados formando un circulo con el pulgar
    Serial.println("Sena: O");
    ponerPosicion(
        MAX_PULGAR_PALMA / 2,  // pulgar_palma: hacia adelante
        MAX_PULGAR_DEDO / 2,   // pulgar_dedo: curvado
        MAX_INDICE * 2/3,      // indice: curvado
        MAX_MEDIO * 2/3,       // medio: curvado
        MAX_ANULAR * 2/3,      // anular: curvado
        MAX_MENIQUE * 2/3      // menique: curvado
    );
}

void hacerLetraU() {
    // U: Indice y medio extendidos juntos, resto cerrado
    Serial.println("Sena: U");
    ponerPosicion(
        MAX_PULGAR_PALMA / 2,  // pulgar_palma: cruza
        MAX_PULGAR_DEDO / 2,   // pulgar_dedo: doblado
        0,                      // indice: extendido
        0,                      // medio: extendido
        MAX_ANULAR,             // anular: cerrado
        MAX_MENIQUE             // menique: cerrado
    );
}

void resetMano() {
    Serial.println("Reset: mano abierta");
    ponerPosicion(0, 0, 0, 0, 0, 0);
}

void cerrarMano() {
    Serial.println("Cerrar: puno completo");
    ponerPosicion(MAX_PULGAR_PALMA, MAX_PULGAR_DEDO, MAX_INDICE, MAX_MEDIO, MAX_ANULAR, MAX_MENIQUE);
}

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            Serial.printf("[WS] Cliente %u conectado\n", num);
            break;
        case WStype_DISCONNECTED:
            Serial.printf("[WS] Cliente %u desconectado\n", num);
            break;
        case WStype_TEXT: {
            if (length > 0) {
                String msg = String((char*)payload).substring(0, length);
                msg.toLowerCase();
                Serial.printf("Mensaje recibido: %s\n", msg.c_str());

                if (msg == "a")           hacerLetraA();
                else if (msg == "e")      hacerLetraE();
                else if (msg == "i")      hacerLetraI();
                else if (msg == "o")      hacerLetraO();
                else if (msg == "u")      hacerLetraU();
                else if (msg == "reset")  resetMano();
                else if (msg == "cerrar") cerrarMano();
                else Serial.printf("Comando desconocido: %s\n", msg.c_str());
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
    Serial.println("WebSocket listo en puerto 81");
}

void loop() {
    ArduinoOTA.handle();
    webSocket.loop();
}
