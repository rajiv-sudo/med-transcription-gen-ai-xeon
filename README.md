# Insights from Medical Transcription using Generative AI LLMs on Intel Xeon Scalable Processors
This repository demonstrates how to leverage Generative AI with the Huggingface model Meta-Llama-3.1-8B to extract insights from medical transcription. It showcases a scalable solution running on Intel Xeon Scalable Processors, utilizing Docker, Huggingface Transformers, and Ray for distributed processing.

We will use an M7i.8xlarge EC2 instance in AWS to run the test. The instance is based on the 4th generation of Intel Xeon Scalable processors called Sapphire Rapids.

## Pre-requisites
- Access to an AWS account and privilege to setup EC2 instances
- Knowledge of setting up EC2 instances
- Basic familiarity with running CLI commands, SSH, and Docker

## Steps for Setup and Execution
In this section we will demonstrate the steps for setup and execution.

### Step-01

Launch an EC2 instance from AWS Console. Select US-East-1 region, since it is likely to have better availability in that region. Specs for the EC2 instance as follows.

> M7i.8xlrage
>
> Ubuntu 22.04
>
> 300 GB disk storage
>
> Enable public IP
>
> Inside your existing VPC or default VPC
>
> Create a key pair or use your existing key pair
>
> Create a security group or use your existing security group. Inbound rules needed for security group are:
>> open port 22 for SSH from your desired IP address
>>
>
> Wait until the EC2 instance is fully launched and ready

### Step-02
Log into the EC2 instance via SSH. You will need the EC2 key that you created or reused in the earlier step. Since the OS is ubuntu 22.04, your ssh login username will be ubuntu.

You can use your favorite SSH clients like putty or bitvise or even log in from your command line.

Once logged in you will be at the directory `/home/ubuntu`

### Step-03
We will install docker on this machine, we will use instructions for Ubuntu 22.04. Recommend running one command at a time.

```
sudo apt update

sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update

apt-cache policy docker-ce

sudo apt install -y docker-ce

sudo systemctl status docker
```
The last command shows Docker service running. Press `Control + C` on a Windows keyboard to break out of the docker status and come back to the command prompt.

### Step-04
In this step, we will start a base docker container for Ubuntu 22.04, install packages in the docker container. Then will lauch a Ray application, which is designed to deploy a generative language model using Ray Serve on Intel Xeon processors. It loads the Huggingface model "Meta-Llama-3.1-8B" and provides functionality to generate text responses based on user input. The model is deployed with a specified number of CPUs (32 in this case) and handles requests via HTTP, accepting input text and generation configurations. The program supports both batch text generation and token-by-token streaming. It integrates with the Huggingface Hub for model access and authentication, and it uses Docker, Huggingface Transformers, and Ray for distributed processing.

#### Use a standard Ubuntu-based Docker image or your local environment

Elevate to root user.
```
sudo su
```

```
docker run -it --net=host --ipc=host ubuntu:22.04
```

```
apt-get update
```

```
apt-get install -y nano
```

#### Update and install dependencies
```
apt-get update && apt-get install -y python3-pip git
```

#### Install Ray, Hugging Face Transformers, and necessary dependencies
To make sure everything installs correctly, we recommend you run one instruction at a time.
```
pip install ray[tune,serve]==2.20.0
pip install transformers
pip install git+https://github.com/huggingface/optimum.git
pip install accelerate
```

### Step-05
Start the Ray application
```
ray start --head
```

Once Ray starts successfully, you will see something like this.
```
--------------------
Ray runtime started.
--------------------

Next steps
  To add another node to this Ray cluster, run
    ray start --address='172.31.28.86:6379'

  To connect to this Ray cluster:
    import ray
    ray.init()

  To submit a Ray job using the Ray Jobs CLI:
    RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python my_script.py

  See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html
  for more information on submitting Ray jobs to the Ray cluster.

  To terminate the Ray runtime, run
    ray stop

  To view the status of the cluster, use
    ray status

  To monitor and debug Ray, view the dashboard at
    127.0.0.1:8265

  If connection to the dashboard fails, check your firewall settings and network configuration.
root@ip-172-31-28-86:/#
```

### Step-06
Set your Hugging Face Hub token. Before running the command below, replace `your_huggingface_token` with your own Huggingface token. The model `meta-llama/Meta-Llama-3.1-8B` we are using is a gated model. Please make sure you have accepted terms for using this model on Higgingface and is approved to use this model. Make sure you are running this command inside the container.

```
export HUGGINGFACE_HUB_TOKEN="your_huggingface_token"
```

### Step-07
In this step we will create the `intel_cpu_inference_serve.py` program that will be served by Ray for the LLM inferencing.

***Open the terminal editor and create the program***

```
nano intel_cpu_inference_serve.py
```

Paste in the code below on the editor. Once you paste in the code, there is a line in the code `huggingface_token = os.getenv('HUGGINGFACE_TOKEN', 'your_huggingface_token')`. In that line replace the `your_huggingface_token` with your own Huggingface token. At the least, your token should have the access to read models from Huggingface. Once you have pasted in the code, use `Control + X` on a Windows keyboard. Press `Y` and press `Enter` key to save the file under the name `intel_cpu_inference_serve.py`. This is the process to save a file in the `nano` editor on an Ubuntu terminal

**Code for intel_cpu_inference_serve.py**
```
import asyncio
from functools import partial
from queue import Empty
from typing import Dict, Any, List
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
import torch
from ray import serve
from huggingface_hub import login
import os

# Authenticate to Hugging Face
def authenticate_to_huggingface(token: str):
    if token:
        login(token=token)
    else:
        raise ValueError("Hugging Face token is not provided.")

# Replace 'your_huggingface_token' with your actual Hugging Face token or set it as an environment variable
huggingface_token = os.getenv('HUGGINGFACE_TOKEN', 'your_huggingface_token')
authenticate_to_huggingface(huggingface_token)

@serve.deployment(ray_actor_options={"num_cpus": 32})
class LlamaModel:
    def __init__(self, model_id_or_path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

        # Set device
        self.device = torch.device("cpu")

        # Initialize event loop
        self.loop = asyncio.get_event_loop()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            use_fast=False,
            use_auth_token=huggingface_token
        )

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            pad_token_added = True
        else:
            pad_token_added = False

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_auth_token=huggingface_token
        ).to(self.device)

        # Resize embeddings if pad token was added
        if pad_token_added:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Set tokenizer configurations
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # Set model to evaluation mode
        self.model.eval()

    def tokenize(self, prompt: str):
        """
        Tokenize the input prompt.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        return input_ids, attention_mask

    def generate(self, prompt: str, generation_config: Dict[str, Any]):
        """
        Generate text based on the input prompt.
        """
        input_ids, attention_mask = self.tokenize(prompt)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output

    async def stream_generate(self, prompt: str, generation_config: Dict[str, Any]):
        """
        Stream generated text token by token.
        """
        input_ids, attention_mask = self.tokenize(prompt)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            **generation_config
        )
        # Run generation in a separate thread to avoid blocking
        generation_task = partial(self.model.generate, **generation_kwargs)
        self.loop.run_in_executor(None, generation_task)
        async for text in streamer:
            yield text

    async def __call__(self, request: Request):
        """
        Handle incoming HTTP requests.
        """
        request_json = await request.json()
        prompt = request_json.get("text", "")
        generation_config = request_json.get("config", {})
        stream = request_json.get("stream", False)

        # Set default generation parameters if not provided
        default_config = {
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2
        }
        # Merge default and provided configs
        generation_config = {**default_config, **generation_config}

        if stream:
            return StreamingResponse(
                self.stream_generate(prompt, generation_config),
                media_type="text/plain"
            )
        else:
            output = self.generate(prompt, generation_config)
            return JSONResponse({"generated_text": output})

# Deploy the model with specified model ID or path
model_id = "meta-llama/Meta-Llama-3.1-8B"  # Replace with your desired model
entrypoint = LlamaModel.bind(model_id)
```
### Step-08
Server the Ray application. Wait until you see `Deployed app 'default' successfully` message on the terminal.

```
serve run intel_cpu_inference_serve:entrypoint
```

You should see something like this. Do not press `Control + C` or any keys on this terminal since the Ray application is running from this terminal.

```
(ServeController pid=6789) WARNING 2024-09-12 14:05:38,658 controller 6789 deployment_state.py:2164 - Deployment 'LlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Downloading shards:  75%|███████▌  | 3/4 [00:29<00:09,  9.87s/it]
Downloading shards: 100%|██████████| 4/4 [00:32<00:00,  8.03s/it]
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:01,  1.93it/s]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.93it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:01<00:00,  1.92it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:01<00:00,  2.27it/s]
2024-09-12 14:05:54,682 INFO handle.py:126 -- Created DeploymentHandle 'gims3bs5' for Deployment(name='LlamaModel', app='default').
2024-09-12 14:05:54,682 INFO api.py:584 -- Deployed app 'default' successfully.
```

### Step-08
In this step, we will create another Python script, that will send the transcription file and the user query to the Ray application to perform the LLM inference.

Open another shell terminal and log into the `home/ubuntu` folder. Then run the following commands.

```
nano send_request.py
```

Paste in the code below on the editor. Once you have pasted in the code, use `Control + X` on a Windows keyboard. Press `Y` and press `Enter` key to save the file under the name `send_request.py`. This is the process to save a file in the `nano` editor on an Ubuntu terminal

**Code for send_request.py**

```
import requests
import json
import argparse
import sys
import time  # Import time module

# Define the server URL
SERVER_URL = "http://127.0.0.1:8000/"

# Use argparse to get the file name from the command line
parser = argparse.ArgumentParser(description="Send a request with context from a file.")
parser.add_argument("file_name", type=str, help="The file containing the context.")
args = parser.parse_args()

# Read the context from the file
try:
    with open(args.file_name, 'r') as file:
        context = file.read()
except FileNotFoundError:
    print(f"Error: The file '{args.file_name}' was not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading the file: {e}")
    sys.exit(1)

# Define the prompt
prompt = (
    "What were the key findings from the patient's ECG?"
    "Please provide a brief explanation."
)

# Combine context and prompt
full_prompt = context + prompt

# Define generation configuration
generation_config = {
    "max_new_tokens": 300,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "do_sample": True
}

# Prepare the request payload
payload = {
    "text": full_prompt,
    "config": generation_config,
    "stream": False  # Set to True if you want streaming response
}

# Send the request to the server
try:
    # Start timing before sending the request
    start_time = time.time()

    response = requests.post(SERVER_URL, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    
    # End timing after receiving the response
    end_time = time.time()
    
    # Calculate response time
    response_time = end_time - start_time
    
    generated_text = result.get("generated_text", "")
    
    # Remove the full prompt from the generated text
    if generated_text.startswith(full_prompt):
        generated_text = generated_text[len(full_prompt):].strip()
    
    print("\nGenerated Response:\n")
    print(generated_text)
    
    # Print the response time
    print(f"\nResponse Time: {response_time:.2f} seconds")
    
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except json.JSONDecodeError:
    print("Failed to decode JSON response.")
```

### Step-09
We will use a sample fictitious transcription notes for a fictitious patient. We will save this fictitious transcription into a filename called `transcription_file.txt`. We will pass this file into the `send_request.py` script, for answering questions based on the transcription file being passed as the context using the LLM which is being served by the Ray application.

You should be in the shell terminal and in the folder `home/ubuntu` folder. Then run the following commands.

```
nano transcription_file.txt
```

Paste in the sample transcription below on the editor. Once you have pasted in the code, use `Control + X` on a Windows keyboard. Press `Y` and press `Enter` key to save the file under the name `send_request.py`. This is the process to save a file in the `nano` editor on an Ubuntu terminal

**Sample transcription for transcription_file.txt**

```
Patient: John Doe  
Date of Birth: 05/14/1972  
Date of Emergency Room Visit: 09/12/2024  
Time of Arrival: 14:35  

Patient is a 52-year-old male presenting to the emergency department with complaints of acute chest pain, described as pressure-like and radiating to the left arm. The pain began approximately 2 hours ago while the patient was at rest and has been persistent since onset. The patient denies recent physical exertion, trauma, or new medications. Past medical history is significant for hypertension and type 2 diabetes, both of which are currently managed with medication. The patient denies any known family history of cardiovascular disease.

Upon arrival, vital signs were as follows: blood pressure 150/90 mmHg, heart rate 95 beats per minute, respiratory rate 18 breaths per minute, oxygen saturation 98% on room air. The patient was immediately placed on continuous cardiac monitoring, and an ECG was performed.

ECG findings revealed ST-segment elevation in leads II, III, and aVF, consistent with acute inferior myocardial infarction. There were also T-wave inversions in leads V5 and V6, suggesting ischemia in the lateral region. No significant arrhythmias were noted during the examination. The patient was given sublingual nitroglycerin and aspirin, which resulted in mild alleviation of chest pain.

Plan includes immediate consultation with cardiology for further evaluation and potential coronary angiography. Troponin levels were drawn and are pending. The patient remains hemodynamically stable and will be admitted for continuous monitoring and further management.
```

### Step-10
In this step, we will run the script `send_request.py` with the transcription file `transcription_file.txt`. The sample question in the script is asking `What were the key findings from the patient's ECG?`. The LLM should be able to use the transcription file as the context and come back with a meaningful response.

Run the command below to execute the script.

```
python3 send_request.py transcription_file.txt
```

Wait for the processing to complete. In our case, we see output like below.

```
Generated Response:

The key finding from this patient’s electrocardiogram (ECG) was ST segment elevations in lead II, III, and aVF indicating an acute inferoposterior wall MI. These changes suggest injury or damage to specific areas within the right ventricle and posterior portion of the septum resulting in blockage of blood flow through these regions leading to tissue death if not treated promptly. Additionally there were t wave inversion observed in leads v5-v6 suggestive of possible lateral ischemic change although additional testing may be required confirm diagnosis.

Response Time: 27.78 seconds
```
## Conclusion
In this example, we see how easy it is to use Intel Xeon Scalable processors for generating meaningful insights from medical transcription files using LLMs. This can be used in many use cases in the medical field.

On the Linux terminal, make sure you are in the `/home/ubuntu` folder, and run the following commands.

Elevate to root user.
```
sudo su
```

## Clean up Docker container and images
**Stop all running containers:**
```
docker stop $(docker ps -q)
```

**Remove all stopped containers:**
```
docker rm $(docker ps -a -q)
```

**Optional: If you also want to remove all Docker images:**
```
docker rmi $(docker images -q)
```

## Destroy AWS Resources
Please remember to destroy the AWS resources to avoid billing
