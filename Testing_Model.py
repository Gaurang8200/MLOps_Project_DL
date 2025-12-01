import base64
from google.cloud import aiplatform

PROJECT_ID = "mlops-project-479512"          
LOCATION = "europe-west3"              
ENDPOINT_ID = "6025044444259024896"  

endpoint_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"

# ---- 1) Encode local image as base64  ----
image_path = "10212.jpg" 

with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

instance = {
    "data": {
        "b64": b64
    }
}
instances = [instance]

# ---- 2) Created client and call endpoint ----
client = aiplatform.gapic.PredictionServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
)

response = client.predict(
    endpoint=endpoint_name,
    instances=instances,
    parameters=None,
)

print("Raw response:", response)
print("Predictions:", response.predictions)