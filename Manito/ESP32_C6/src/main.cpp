#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <WebSocketsServer.h>

// ── Credenciales WiFi ────────────────────────────────────────────────────────
const char* WIFI_SSID     = "M@STV-SARABELLA";
const char* WIFI_PASSWORD = "p4taC0n_cOn_Qu3sO";

// ── WebSocket en puerto 81 ───────────────────────────────────────────────────
WebSocketsServer webSocket = WebSocketsServer(81);

// ── OTA ──────────────────────────────────────────────────────────────────────
void setupOTA() {
    ArduinoOTA.setHostname("esp32-c6-mano");

    ArduinoOTA.onStart([]() {
        Serial.println("[OTA] Iniciando actualización...");
    });
    ArduinoOTA.onEnd([]() {
        Serial.println("\n[OTA] Actualización finalizada con éxito.");
    });
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("[OTA] Progreso: %u%%\r", progress * 100 / total);
    });
    ArduinoOTA.onError([](ota_error_t error) {
        Serial.printf("[OTA] Error [%u]: ", error);
        if      (error == OTA_AUTH_ERROR)    Serial.println("Error de autenticación");
        else if (error == OTA_BEGIN_ERROR)   Serial.println("Error al iniciar");
        else if (error == OTA_CONNECT_ERROR) Serial.println("Error de conexión");
        else if (error == OTA_RECEIVE_ERROR) Serial.println("Error al recibir");
        else if (error == OTA_END_ERROR)     Serial.println("Error al finalizar");
    });

    ArduinoOTA.begin();
}

// ── Evento WebSocket ─────────────────────────────────────────────────────────
void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            Serial.printf("[WS] Cliente #%u conectado\n", num);
            break;
        case WStype_DISCONNECTED:
            Serial.printf("[WS] Cliente #%u desconectado\n", num);
            break;
        case WStype_TEXT:
            Serial.printf("[WS] Comando recibido para la mano: %s\n", payload);
            // Aquí procesaremos los movimientos de los dedos más adelante
            break;
        default:
            break;
    }
}

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n[BOOT] Iniciando ESP32-C6...");

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("[WiFi] Conectando");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n[WiFi] Conectado!");
    Serial.print("[WiFi] IP para PlatformIO: ");
    Serial.println(WiFi.localIP());

    setupOTA();
    Serial.println("[OTA] Listo en 'esp32-c6-mano'");

    webSocket.begin();
    webSocket.onEvent(onWebSocketEvent);
    Serial.println("[WS] WebSocket escuchando en puerto 81");
}

// ── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    ArduinoOTA.handle();  // Revisa si hay una nueva versión disponible
    webSocket.loop();     // Revisa si hay nuevas letras enviadas desde la PC
}
