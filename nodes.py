import subprocess
import sys
import re
import comfy.model_management as model_management
import gc
import torch
import requests
import time

class EbuLMStudioLoadModel:
    ERROR_NO_MODEL_FOUND = "no model by that name found"
    ERROR_CANT_CONNECT = "can't connect to lmstudio"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING",),
                "model_search_string": ("STRING", {"default": "llama"}),
                "context_length": ("INT", {"default": 4096, "min": 1000, "max": 200000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "unload_image_models_first": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("export_string", "model_selected", "models_found",)
    FUNCTION = "load_model"
    CATEGORY = "LMStudio"

    def check_models_loaded(self):
        """Check if any models are currently loaded in LMStudio"""
        try:
            # Run the 'lms status' command
            command = ['lms', 'status']
            status_process = self.run_command(command)

            if not status_process:
                return True  # Assume models are loaded if we can't check

            # Look for the "No Models Loaded" indicator in the output
            return "No Models Loaded" not in status_process.stdout

        except Exception as e:
            print(f"Error checking model status: {str(e)}", file=sys.stderr)
            return True  # Assume models are loaded if check fails

    def run_command(self, command):
        try:
            # Use utf-8 encoding with errors='replace' to handle problematic characters
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False  # Get bytes instead of string
            )

            stdout_bytes, stderr_bytes = process.communicate()

            # Decode with 'utf-8' and replace invalid characters
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
            print("   - Unable to clear cache")

    def find_matching_model(self, search_term):
        try:
            print("Fetching model list...")

            # Run the 'lms ls --detailed' command to get available models
            command = ['lms', 'ls', '--detailed']
            list_process = self.run_command(command)

            if not list_process:
                return self.ERROR_CANT_CONNECT, ""

            result = list_process.stdout.replace("\r\n", "\n").replace("\r", "\n")

            # Split search term into parts and escape special regex characters
            search_parts = [re.escape(part) for part in search_term.lower().split()]

            # Create a pattern that matches all parts in any order
            # This allows for flexible matching with spaces treated as wildcards
            search_pattern = '.*'.join(search_parts)
            matched_models = []

            # Iterate through the output and find matching models
            for line in result.split("\n"):
                line = line.strip()

                # Only match lines that start with '/', these are the llm full names
                if line.startswith('/'):
                    model_path = line.split()[0].lstrip("/")
                    if re.search(search_pattern, model_path.lower()):
                        matched_models.append(model_path)

            if not matched_models:
                print(f"No models found matching '{search_term}'. Available models:\n{result}")
                return self.ERROR_NO_MODEL_FOUND, result

            # Return the first matched model or None if no match
            return matched_models[0], result

        except subprocess.CalledProcessError:
            print("Error: Unable to fetch the model list. Is LMStudio running?", file=sys.stderr)
            return self.ERROR_CANT_CONNECT, ""
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return self.ERROR_CANT_CONNECT, ""

    def load_model(self, input_string, model_search_string, context_length, seed, unload_image_models_first):
        try:
            # Only unload image models if specified
            if unload_image_models_first:
                print("Unloading ComfyUI image models to save memory (as requested)")
                self.unload_image_models()

            # Always unload all LLM models first if there are any in memory
            if self.check_models_loaded():
	            print("Unloading LLMs in memory...")
	            subprocess.run(['lms', 'unload', '--all'],
	                           stdout=subprocess.DEVNULL,
	                           stderr=subprocess.DEVNULL,
	                           check=True)

            print(f"Loading model: {model_search_string}")
            model_path, models_found = self.find_matching_model(model_search_string)
            if model_path in [self.ERROR_NO_MODEL_FOUND, self.ERROR_CANT_CONNECT]:
                return (input_string, model_path, models_found)

            # Run the 'lms load' command to load the model
            command = ['lms', 'load', model_path, '-y', f'--context-length={context_length}', f'--gpu=1']
            load_process = self.run_command(command)

            if load_process and load_process.returncode == 0:  # Check return code for success
                print(f"Model loaded: {model_path}")
                return (input_string, model_path, models_found)
            else:
                print(f"Warning: Issue loading model: {model_path}")
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

            # Run the command 'lms unload --all' to unload all llms
            result = subprocess.run(['lms', 'unload', '--all'],
                                    capture_output=True,
                                    text=True,
                                    check=True)

            # Print the output of the command
            print("Command output:", result.stdout)

            # If there's any error output, print it as well
            if result.stderr:
                print("", result.stderr, file=sys.stderr)

            # Return the input string unchanged
            return (input_string,)

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}", file=sys.stderr)
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
                "context_length": ("INT", { "default": 4096, "min": 512, "max": 8192, "display": "slider" }),
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
        # Sanitize text to ensure it's valid UTF-8 using unicode normalization
        print("Sanitizing the response into UTF-8 safe characters as requested...")
        try:
            from unicodedata import normalize
            # Normalize to decomposed form and convert to ASCII, dropping diacritics
            return normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        except Exception as e:
            print(f"Warning: UTF-8 sanitization failed: {str(e)}")
            # Fallback to simple replacement if something goes wrong
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

NODE_CLASS_MAPPINGS = {
    "EbuLMStudioLoadModel": EbuLMStudioLoadModel,
    "EbuLMStudioUnload": EbuLMStudioUnload,
    "EbuLMStudioMakeRequest": EbuLMStudioMakeRequest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EbuLMStudioLoadModel": "EBU LMStudio Load",
    "EbuLMStudioUnload": "EBU LMStudio Unload All",
    "EbuLMStudioMakeRequest": "EBU LMStudio Make Request",
}
