import websocket

# Sustituye por la IP que sale en el monitor serial de tu ESP32
esp32_ip = "192.168.1.50"
uri = f"ws://{esp32_ip}:81"

def iniciar_consola():
    try:
        ws = websocket.create_connection(uri)
        print("Conectado a la Manito ESP32-C6. Escribe una letra y presiona Enter (Ctrl+C para salir):")
        
        while True:
            letra = input("Letra > ")
            if len(letra) > 0:
                ws.send(letra[0]) # Enviar solo el primer caracter
                print(f"Enviado: {letra[0]}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ws' in locals(): ws.close()

if __name__ == "__main__":
    iniciar_consola()