import websocket

# Sustituye por la IP que sale en el monitor serial de tu ESP32
esp32_ip = "192.168.1.50"
uri = f"ws://{esp32_ip}:81"

def iniciar_consola():
    try:
        ws = websocket.create_connection(uri)
        print("Conectado a la Manito ESP32-C6.")
        print("  - Escribe una letra y presiona Enter para enviarla.")
        print("  - Escribe 'ya' para mover todos los dedos 20°.")
        print("  - Escribe 'no' para regresar todos los dedos a 0°.")
        print("  - Ctrl+C para salir.")

        while True:
            entrada = input("Comando > ").strip()
            if not entrada:
                continue

            if entrada.lower() == "ya":
                ws.send("ya")
                print("Enviado: ya  →  todos los dedos se mueven 20°")
            elif entrada.lower() == "no":
                ws.send("no")
                print("Enviado: no  →  todos los dedos vuelven a 0°")
            else:
                ws.send(entrada[0])
                print(f"Enviado: {entrada[0]}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ws' in locals():
            ws.close()

if __name__ == "__main__":
    iniciar_consola()