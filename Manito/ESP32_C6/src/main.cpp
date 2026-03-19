#include <WiFi.h>
#include <ArduinoOTA.h>
#include <WebSocketsServer.h>
#include <ESP32Servo.h>

const char* ssid     = "M@STV-SARA-BELLA";
const char* password = "P4tac0n_cOn_Qu3s0";

WebSocketsServer webSocket = WebSocketsServer(81);

// Pines de los servos
#define PIN_PULGAR_PALMA  2
#define PIN_PULGAR_DEDO   13
#define PIN_INDICE        12
#define PIN_MEDIO          7
#define PIN_ANULAR         3
#define PIN_MENIQUE        6

Servo servoPulgarPalma;
Servo servoPulgarDedo;
Servo servoIndice;
Servo servoMedio;
Servo servoAnular;
Servo servoMenique;

// Posición actual de cada servo (inician en 0°)
int posPulgarPalma = 0;
int posPulgarDedo  = 0;
int posIndice      = 0;
int posMedio       = 0;
int posAnular      = 0;
int posMenique     = 0;

void resetTodosLosDedos() {
    posPulgarPalma = 0;
    posPulgarDedo  = 0;
    posIndice      = 0;
    posMedio       = 0;
    posAnular      = 0;
    posMenique     = 0;

    servoPulgarPalma.write(0);
    servoPulgarDedo.write(0);
    servoIndice.write(0);
    servoMedio.write(0);
    servoAnular.write(0);
    servoMenique.write(0);

    Serial.println("Todos los dedos vuelven a 0°");
}

void moverTodosLosDedos(int grados) {
    posPulgarPalma = constrain(posPulgarPalma + grados, 0, 180);
    posPulgarDedo  = constrain(posPulgarDedo  + grados, 0, 180);
    posIndice      = constrain(posIndice      + grados, 0, 180);
    posMedio       = constrain(posMedio       + grados, 0, 180);
    posAnular      = constrain(posAnular      + grados, 0, 180);
    posMenique     = constrain(posMenique     + grados, 0, 180);

    servoPulgarPalma.write(posPulgarPalma);
    servoPulgarDedo.write(posPulgarDedo);
    servoIndice.write(posIndice);
    servoMedio.write(posMedio);
    servoAnular.write(posAnular);
    servoMenique.write(posMenique);

    Serial.printf("Dedos movidos +%d°: pulgar_palma=%d, pulgar_dedo=%d, indice=%d, medio=%d, anular=%d, menique=%d\n",
        grados, posPulgarPalma, posPulgarDedo, posIndice, posMedio, posAnular, posMenique);
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

                if (msg == "ya") {
                    moverTodosLosDedos(50);
                } else if (msg == "no") {
                    resetTodosLosDedos();
                }
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

    // Posición inicial: todos a 0°
    servoPulgarPalma.write(0);
    servoPulgarDedo.write(0);
    servoIndice.write(0);
    servoMedio.write(0);
    servoAnular.write(0);
    servoMenique.write(0);
    Serial.println("Servos inicializados en 0°");

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
