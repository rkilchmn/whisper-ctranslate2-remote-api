import sounddevice as sd

def list_input_devices():
    devices = sd.query_devices()

    print("All available input audio devices:")
    for device in devices:
        if device['max_input_channels'] > 0:
            print(f"Device Name: {device['name']}, Index: {device['index']}, Max Input Channels: {device['max_input_channels']}")

if __name__ == "__main__":
    list_input_devices()