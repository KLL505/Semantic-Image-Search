import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel

class Settings:
    def __init__(self, settings_file="./data/settings.json"):
        self.settings_file = settings_file
        self.default_model = "openai/clip-vit-base-patch32"
        # Automatically load the saved model ID (and create the file if missing)
        self.current_model_id = self.load_settings()

    def load_settings(self):
        # If the file exists, read it and grab the model ID
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    return settings.get("model_id", self.default_model)
            except Exception as e:
                print(f"Error loading settings: {e}")
                return self.default_model
        else:
            # If the file DOES NOT exist, create it automatically with the default model
            print("Settings file not found. Creating a default settings.json...")
            self.save_settings(self.default_model)
            return self.default_model

    def save_settings(self, model_id):
        # Ensure the data folder exists, then write the new setting to disk
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        with open(self.settings_file, "w") as f:
            # Added indent=4 so the JSON file is easily readable by humans
            json.dump({"model_id": model_id}, f, indent=4)
        # Update the active tracker
        self.current_model_id = model_id

    def initialize_backend(self, model_id=None):
        # Fallback to the saved model if no specific one is provided
        if model_id is None:
            model_id = self.current_model_id
            
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        print(f"Using {device} device for {model_id}")

        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.eval()
        processor.use_fast = False

        return device, model, processor