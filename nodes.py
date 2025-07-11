import subprocess
import sys
import re
import comfy.model_management as model_management
import gc
import torch
import requests
import time
import random

class EbuLMStudioLoadModel:
    ERROR_NO_MODEL_FOUND = "no model by that name found"
    ERROR_CANT_CONNECT = "can't connect to lmstudio"
    _currently_loaded_model_path = None  # Store the path of the loaded model

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING",),
                "model_search_string": ("STRING", {"default": "llama"}),
                "offload": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number"
                }),
                "context_length": ("INT", {"default": 4096, "min": 1000, "max": 200000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "unload_image_models_first": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("export_string", "model_selected", "models_found",)
    FUNCTION = "load_model"
    CATEGORY = "LMStudio"

    def get_currently_loaded_model_path(self):
        """Retrieves the path of the currently loaded model from 'lms ps'."""
        try:
            command = ['lms', 'ps']
            process = self.run_command(command)
            if process and "LOADED MODELS" in process.stdout:
                for line in process.stdout.splitlines():
                    if "Path:" in line:
                        loaded_path = line.split("Path:")[1].strip()
                        return loaded_path
            return None
        except Exception as e:
            print(f"Error getting loaded model path: {e}", file=sys.stderr)
            return None

    def check_models_loaded(self):
        """Check if any models are currently loaded in LMStudio"""
        return self.get_currently_loaded_model_path() is not None

    def run_command(self, command):
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False
            )
            stdout_bytes, stderr_bytes = process.communicate()
            stdout = stdout_bytes.decode('utf-8', errors='replace')
            stderr = stderr_bytes.decode('utf-8', errors='replace')
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            result = subprocess.CompletedProcess(
                args=command,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error executing command {command}: {e}", file=sys.stderr)
            return None
        except FileNotFoundError:
            print("Error: 'lms' command not found. Make sure LM Studio is installed and the 'lms' command is in your system's PATH.", file=sys.stderr)
            return None

    def unload_image_models(self):
        print("Unload Image Models:")
        print(" - Unloading all image models...")
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        try:
            print(" - Clearing Cache...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except:
            print("    - Unable to clear cache")

    def find_matching_model(self, search_term):
        try:
            if '|' in search_term:
                search_options = [s.strip() for s in search_term.split('|') if s.strip()]
                if search_options:
                    search_term = random.choice(search_options)
                    print(f"Multiple search strings provided. Randomly selected: '{search_term}'")

            print("Fetching model list...")
            command = ['lms', 'ls', '--detailed']
            list_process = self.run_command(command)

            if not list_process:
                return self.ERROR_CANT_CONNECT, ""

            result = list_process.stdout.replace("\r\n", "\n").replace("\r", "\n")
            search_parts = [re.escape(part) for part in search_term.lower().split()]
            search_pattern = '.*'.join(search_parts)
            matched_models = []

            for line in result.split("\n"):
                line = line.strip()
                if line.startswith('/'):
                    model_path = line.split()[0].lstrip("/")
                    if re.search(search_pattern, model_path.lower()):
                        matched_models.append(model_path)

            if not matched_models:
                print(f"No models found matching '{search_term}'. Available models:\n{result}")
                return self.ERROR_NO_MODEL_FOUND, result

            return matched_models[0], result

        except subprocess.CalledProcessError:
            print("Error: Unable to fetch the model list. Is LMStudio running?", file=sys.stderr)
            return self.ERROR_CANT_CONNECT, ""
        except FileNotFoundError:
            print("Error: 'lms' command not found. Make sure LM Studio is installed and the 'lms' command is in your system's PATH.", file=sys.stderr)
            return self.ERROR_CANT_CONNECT, ""
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return self.ERROR_CANT_CONNECT, ""

    def load_model(self, input_string, model_search_string, context_length, seed, unload_image_models_first):
        try:
            # Get the path of the currently loaded model
            current_loaded_path = self.get_currently_loaded_model_path()

            # Find the model path to be loaded
            model_path_to_load, models_found = self.find_matching_model(model_search_string)
            if model_path_to_load in [self.ERROR_NO_MODEL_FOUND, self.ERROR_CANT_CONNECT]:
                return (input_string, model_path_to_load, models_found)

            # Log the comparison strings
            print(f"Comparing loaded path: '{current_loaded_path}' with target path: '{model_path_to_load}'")

            # Check if the resolved model path is already loaded
            if current_loaded_path and model_path_to_load in current_loaded_path:
                print(f"Model '{model_path_to_load}' (resolved from '{model_search_string}') is already loaded in memory.")
                return (input_string, model_path_to_load, models_found)
            else:
                # Unload LLMs only if a different model needs to be loaded
                if self.check_models_loaded():
                    print("Unloading LLMs in memory...")
                    subprocess.run(['lms', 'unload', '--all'],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        check=True)
                    self._currently_loaded_model_path = None # Reset loaded model path

            # Only unload image models if specified
            if unload_image_models_first:
                print("Unloading ComfyUI image models to save memory (as requested)")
                self.unload_image_models()

            print(f"Loading model: {model_path_to_load}")
            # Run the 'lms load' command to load the model
            command = ['lms', 'load', model_path_to_load, '-y', f'--context-length={context_length}', f'--gpu={offload}']
            load_process = self.run_command(command)

            if load_process and load_process.returncode == 0:
                print(f"Model loaded: {model_path_to_load}")
                self._currently_loaded_model_path = model_path_to_load
                return (input_string, model_path_to_load, models_found)
            else:
                print(f"Warning: Issue loading model: {model_path_to_load}. Try to reduce offload")
                return (input_string, self.ERROR_NO_MODEL_FOUND, models_found)

        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)

class EbuLMStudioUnload:
    ERROR_CANT_CONNECT = "can't connect to lmstudio"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("export_string",)
    FUNCTION = "unload_all"
    CATEGORY = "LMStudio"

    def unload_all(self, input_string, seed):
        try:
            print("Calling lms unload --all")
            result = subprocess.run(['lms', 'unload', '--all'],
                                        capture_output=True,
                                        text=True,
                                        check=True)
            print("Command output:", result.stdout)
            if result.stderr:
                print("", result.stderr, file=sys.stderr)
            EbuLMStudioLoadModel._currently_loaded_model_path = None
            return (input_string,)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}", file=sys.stderr)
            return (input_string,)
        except FileNotFoundError:
            print("Error: 'lms' command not found. Make sure LM Studio is installed and the 'lms' command is in your system's PATH.", file=sys.stderr)
            return (input_string,)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            return (input_string,)


# initially inspired by the excellent LLM node from CrasH Utils Custom Nodes
class EbuLMStudioMakeRequest:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", { "multiline": True, "default": "" }),
                "system_message": ("STRING", { "multiline": True, "default": "You are an assistant designed to craft AI image prompts for an AI image generator that uses natural language prompts. Follow the instructions you are given, or use the guidelines, to create a detailed prompt that includes creativity and amazing visual details for an unforgetable image. Respond with just your new prompt." }),
                "url": ("STRING", { "multiline": False, "default": "http://127.0.0.1:1234/v1/chat/completions" }),
                "context_length": ("INT", { "default": 4096, "min": 512, "max": 65536, "display": "slider" }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 300, "min": 10, "max": 100000}),
                "temp": ("FLOAT", {
                    "default": 0.7,
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "display": "number"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": "number"
                }),
                "utf8_safe_replace": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generateText"
    OUTPUT_NODE = False
    CATEGORY = "LMStudio"

    def sanitize_utf8(self, text):
        print("Sanitizing the response into UTF-8 safe characters as requested...")
        try:
            from unicodedata import normalize
            return normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        except Exception as e:
            print(f"Warning: UTF-8 sanitization failed: {str(e)}")
            return text.encode('ascii', 'replace').decode('ascii')

    def generateText(self, prompt, system_message, url, context_length, seed, max_tokens, temp, top_p, utf8_safe_replace):
        description = self.call_api(prompt, system_message, url, context_length, seed, max_tokens, temp, top_p, utf8_safe_replace)
        return (description,)

    def call_api(self, prompt_text, system_message, url, context_length, seed, max_tokens, temp, top_p, utf8_safe_replace):
        payload = {
            "messages": [
                { "role": "system", "content": f"{system_message}\n" },
                { "role": "user", "content": f"{prompt_text}\n" }],
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "seed": seed
        }
        start_time = time.time()
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result_json = response.json()
                generatedText = result_json["choices"][0]["message"]["content"]
                if utf8_safe_replace:
                    generatedText = self.sanitize_utf8(generatedText)
                    print("Response (UTF-8 sanitized): " + generatedText)
                else:
                    print("Response: " + generatedText)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"\n[Response generation time: {elapsed_time:.2f} seconds]\n")
                return generatedText
            else:
                error_message = f"Error {response.status_code}: {response.text}"
                print(error_message)
                return error_message
        except requests.exceptions.RequestException as e:
            error_message = f"Request failed: {str(e)}"
            print(error_message)
            return error_message

class EbuLMStudioUnloadGuider:
    ERROR_CANT_CONNECT = "can't connect to lmstudio"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "guider": ("GUIDER",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "unload_all"
    CATEGORY = "LMStudio"

    def unload_all(self, guider, seed):
        try:
            print("Calling lms unload --all")
            result = subprocess.run(['lms', 'unload', '--all'],
                                        capture_output=True,
                                        text=True,
                                        check=True)
            print("Command output:", result.stdout)
            if result.stderr:
                print("", result.stderr, file=sys.stderr)
            EbuLMStudioLoadModel._currently_loaded_model_path = None
            return (guider,)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}", file=sys.stderr)
            return (guider,)
        except FileNotFoundError:
            print("Error: 'lms' command not found. Make sure LM Studio is installed and the 'lms' command is in your system's PATH.", file=sys.stderr)
            return (guider,)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            return (guider,)

NODE_CLASS_MAPPINGS = {
    "EbuLMStudioLoadModel": EbuLMStudioLoadModel,
    "EbuLMStudioUnload": EbuLMStudioUnload,
    "EbuLMStudioMakeRequest": EbuLMStudioMakeRequest,
    "EbuLMStudioUnloadGuider": EbuLMStudioUnloadGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EbuLMStudioLoadModel": "EBU LMStudio Load",
    "EbuLMStudioUnload": "EBU LMStudio Unload All",
    "EbuLMStudioMakeRequest": "EBU LMStudio Make Request",
    "EbuLMStudioUnloadGuider": "EBU LMStudio Unload (Guider)",
}
