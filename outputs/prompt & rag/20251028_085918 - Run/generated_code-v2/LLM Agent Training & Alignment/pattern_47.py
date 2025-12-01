import json

def simulate_troubleshooting_data():
    """
    Simulates a small dataset of technical troubleshooting demonstrations.
    Each demonstration consists of a sequence of (system_state, agent_action) pairs.
    """
    demonstrations = [
        {
            "scenario": "Wi-Fi Router Setup - Basic Connection",
            "dialog": [
                {"state": "Customer: My internet isn't working. How do I set up my new Wi-Fi router?", "action": "Agent: Please connect your router to the modem using an Ethernet cable and plug in the power adapter. Wait 2 minutes for it to boot up."},
                {"state": "Customer: Okay, the lights are blinking. What next?", "action": "Agent: Please open a web browser and navigate to the router's default IP address, usually 192.168.1.1 or 192.168.0.1."},
                {"state": "Customer: I see a login page. What are the credentials?", "action": "Agent: The default username is 'admin' and the password is 'password'. If that doesn't work, check the sticker on your router."},
                {"state": "Customer: I'm logged in. Now what?", "action": "Agent: Navigate to the 'Wireless Settings' or 'Wi-Fi Setup' section. Look for an option to configure the Wi-Fi name (SSID) and password."},
                {"state": "Customer: I've set up the Wi-Fi name and password. Is that it?", "action": "Agent: Yes, save the settings and try connecting a device to the new Wi-Fi network. You should now have internet access."},
            ]
        },
        {
            "scenario": "Software Error - Cache Clear",
            "dialog": [
                {"state": "Customer: My application XYZ is crashing frequently. It was working fine yesterday.", "action": "Agent: Please try clearing the application's cache. Go to Settings > Apps > XYZ > Storage and tap 'Clear Cache'."},
                {"state": "Customer: I cleared the cache. Should I restart the app?", "action": "Agent: Yes, please restart the application and see if the issue persists."},
                {"state": "Customer: It seems to be working now. Thanks!", "action": "Agent: Great! If the issue returns, try restarting your device as well."},
            ]
        },
        {
            "scenario": "Printer Offline",
            "dialog": [
                {"state": "Customer: My printer is showing 'offline' and won't print.", "action": "Agent: First, ensure the printer is turned on and connected to your computer or Wi-Fi network. Check the power and network cables."},
                {"state": "Customer: It's on and connected to Wi-Fi. Still offline.", "action": "Agent: Please restart both your computer and the printer. Sometimes a simple restart resolves connectivity issues."},
                {"state": "Customer: I restarted both, but it's still offline. Do I need to reinstall drivers?", "action": "Agent: Let's try updating the printer driver. Go to Device Manager, find your printer, right-click, and select 'Update driver'."},
            ]
        }
    ]

    # Convert demonstrations into a format suitable for language model training
    # Input: "[CLS] Customer: ... System: ... [SEP]"
    # Target: "Agent: ..."
    formatted_data = []
    for demo in demonstrations:
        for i in range(len(demo["dialog"])): # Iterate through each turn in the dialog
            current_state = demo["dialog"][i]["state"]
            current_action = demo["dialog"][i]["action"]

            # For simplicity, we'll use the customer's current query as the primary input state
            # In a real system, 'state' would also include previous turns or system outputs.
            # We'll prepend 'Customer:' to the state for clarity in input.
            # The model learns to generate 'Agent: ...' given this context.
            input_text = f"[CLS] {current_state} [SEP]"
            target_text = f"{current_action}"
            formatted_data.append({"input": input_text, "target": target_text})
            
    with open("troubleshooting_data.jsonl", "w") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {len(formatted_data)} data points and saved to troubleshooting_data.jsonl")
    return "troubleshooting_data.jsonl"

if __name__ == "__main__":
    simulate_troubleshooting_data()
