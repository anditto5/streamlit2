import subprocess
import time
from pyngrok import ngrok

# Start the Streamlit app in background
subprocess.Popen(["streamlit", "run", "Profile.py"])

# Wait for the app to spin up
time.sleep(5)

# Open ngrok tunnel to port 8501 (default streamlit port)
public_url = ngrok.connect(8501)
print("Public URL:", public_url)