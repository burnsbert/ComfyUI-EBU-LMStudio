import subprocess
import sys
import re
import os
import comfy.model_management as model_management
import gc
import torch
import requests
import time
import random
import json
from urllib.parse import urlparse

class EbuLMStudioLoadModel:
    """
    Updated for LM Studio 0.4.x:
      - `lms ps` output format changed; use `lms ps --json`
      - OpenAI-compatible endpoints expect a required `model` field; load with `--identifier`
    """
    ERROR_NO_MODEL_FOUND = "no model by that name found"
    ERROR_CANT_CONNECT = "can't connect to lmstudio"

    _currently_loaded_model_key = None
    _currently_loaded_model_path = None
    _currently_loaded_identifier = None

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

    def run_command(self, command, timeout=180):
        try:
            process = subprocess.run(
                [str(c) for c in command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False
            )
            if process.returncode != 0:
                print(f"[lms-node] Command failed: {command}", file=sys.stderr)
                if process.stdout:
                    print(f"[lms-node] STDOUT:\n{process.stdout}", file=sys.stderr)
                if process.stderr:
                    print(f"[lms-node] STDERR:\n{process.stderr}", file=sys.stderr)
                return None
            return process
        except FileNotFoundError:
            print("Error: 'lms' command not found. Make sure LM Studio is installed and in PATH.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"[lms-node] Unexpected error running {command}: {e}", file=sys.stderr)
            return None

    def _list_models_json(self):
        """
        Return list[dict] from `lms ls --llm --json` (preferred) or `lms ls --json` as fallback.
        """
        for cmd in (['lms', 'ls', '--llm', '--json'], ['lms', 'ls', '--json'], ['lms', 'models', 'ls', '--json']):
            p = self.run_command(cmd)
            if not (p and p.stdout and p.stdout.strip().startswith(('[', '{'))):
                continue
            try:
                data = json.loads(p.stdout)
            except Exception:
                continue
            items = data
            if isinstance(items, dict):
                items = items.get('models') or items.get('data') or items.get('items') or []
            if isinstance(items, dict):
                items = items.get('models') or items.get('data') or []
            if not isinstance(items, list):
                continue
            llms = [m for m in items if isinstance(m, dict) and (m.get("type") in (None, "llm", "LLM") or m.get("isEmbedding") is False)]
            return llms or [m for m in items if isinstance(m, dict)]
        return []

    def _model_info_for_key(self, key):
        for m in self._list_models_json():
            if not isinstance(m, dict):
                continue
            if m.get('modelKey') == key or m.get('key') == key or m.get('identifier') == key:
                return m
        return None

    def _path_for_model_key(self, key):
        m = self._model_info_for_key(key) or {}
        return m.get('path')

    def _ps_models(self):
        """
        Return a list of dicts describing loaded models, normalized across versions.
        Prefer `lms ps --json` (0.4+).
        """
        p = self.run_command(['lms', 'ps', '--json'])
        if p and p.stdout and p.stdout.strip().startswith(('[', '{')):
            try:
                data = json.loads(p.stdout)
                items = data
                if isinstance(items, dict):
                    items = items.get('models') or items.get('data') or items.get('items') or []
                if isinstance(items, dict):
                    items = items.get('models') or items.get('data') or []
                if not isinstance(items, list):
                    items = []
            except Exception:
                items = []
            out = []
            for m in items:
                if not isinstance(m, dict):
                    continue
                out.append({
                    "identifier": m.get("identifier") or m.get("id") or m.get("model") or m.get("modelKey"),
                    "modelKey": m.get("modelKey") or m.get("model_key") or m.get("key") or m.get("id"),
                    "path": m.get("path"),
                    "type": m.get("type"),
                    "contextLength": m.get("contextLength") or m.get("context_length") or m.get("context"),
                    "maxContextLength": m.get("maxContextLength") or m.get("max_context_length"),
                    "sizeBytes": m.get("sizeBytes") or m.get("size_bytes"),
                    "status": m.get("status") or m.get("generationStatus") or m.get("state"),
                    "raw": m,
                })
            return out

        # Fallback: parse human output (Identifier/Path/etc.)
        p = self.run_command(['lms', 'ps'])
        txt = p.stdout if p else ""
        if not txt:
            return []

        models = []
        current = {}
        for ln in txt.splitlines():
            line = ln.strip()
            if not line:
                continue

            m = re.match(r'^Identifier:\s*(.+)$', line, flags=re.IGNORECASE)
            if m:
                if current:
                    models.append(current)
                current = {"identifier": m.group(1).strip()}
                continue

            m = re.match(r'^[•\-\*]\s*Type:\s*(.+)$', line, flags=re.IGNORECASE)
            if m:
                current["type"] = m.group(1).strip()
                continue
            m = re.match(r'^[•\-\*]\s*Path:\s*(.+)$', line, flags=re.IGNORECASE)
            if m:
                current["path"] = m.group(1).strip()
                continue

        if current:
            models.append(current)

        for m in models:
            m.setdefault("modelKey", m.get("identifier"))
        return models

    def check_models_loaded(self):
        return bool(self._ps_models())

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
        except Exception:
            print("    - Unable to clear cache")

    def find_matching_model(self, search_term):
        """
        Returns (chosen_model_key, raw_listing_json_text).
        Spaces act like wildcards (tokens may appear anywhere).
        """
        try:
            if '|' in (search_term or ''):
                options = [s.strip() for s in search_term.split('|') if s.strip()]
                if options:
                    search_term = random.choice(options)
                    print(f"Multiple search strings provided. Randomly selected: '{search_term}'")

            models = self._list_models_json()
            listing = json.dumps(models, indent=2)

            tokens = [t for t in re.split(r'\s+', (search_term or '').strip().lower()) if t]
            if not tokens:
                return self.ERROR_NO_MODEL_FOUND, listing

            candidates = []
            for m in models:
                if not isinstance(m, dict):
                    continue
                key = (m.get('modelKey') or m.get('key') or m.get('name') or
                       m.get('identifier') or m.get('displayName') or "")
                disp = m.get('displayName', "") or ""
                path = m.get('path', "") or ""
                arch = m.get('architecture', "") or ""
                params = m.get('paramsString', "") or ""
                q = m.get('quantization') or {}
                qname = (q.get('name') if isinstance(q, dict) else "") or ""
                qbits = (str(q.get('bits')) if isinstance(q, dict) and q.get('bits') is not None else "")

                hay = " ".join([str(key), disp, path, arch, params, qname, qbits]).lower()
                if all(t in hay for t in tokens):
                    in_order = bool(re.search(".*".join(map(re.escape, tokens)), hay))
                    candidates.append({
                        "key": str(m.get('modelKey', key)),
                        "in_order": in_order,
                        "key_len": len(str(m.get('modelKey', key)) or ""),
                        "hay_len": len(hay),
                    })

            if not candidates:
                print(f"No models found matching '{search_term}'.")
                return self.ERROR_NO_MODEL_FOUND, listing

            candidates.sort(key=lambda c: (c["in_order"], c["key_len"], -c["hay_len"]), reverse=True)
            return candidates[0]["key"], listing

        except Exception as e:
            print(f"[lms-node] Unexpected error in find_matching_model: {e}", file=sys.stderr)
            return self.ERROR_CANT_CONNECT, ""

    def _safe_identifier_for_key(self, model_key: str) -> str:
        ident = (model_key or "comfyui-model").strip()
        ident = re.sub(r'[^A-Za-z0-9._\-]+', '-', ident)
        ident = re.sub(r'-{2,}', '-', ident).strip('-')
        return ident[:64] if len(ident) > 64 else ident

    def _build_loaded_metadata(self, model_key, chosen_path, context_length):
        info = self._model_info_for_key(model_key) or {}
        identifier = self._currently_loaded_identifier

        ps = self._ps_models()
        match = None
        for r in ps:
            if r.get("modelKey") == model_key or r.get("identifier") == identifier or r.get("identifier") == model_key:
                match = r
                break

        return {
            "modelKey": model_key,
            "identifier": (match or {}).get("identifier") or identifier,
            "displayName": info.get("displayName"),
            "path": info.get("path") or chosen_path,
            "architecture": info.get("architecture"),
            "paramsString": info.get("paramsString"),
            "quantization": info.get("quantization"),
            "maxContextLength": info.get("maxContextLength"),
            "loadedContextLength": int(context_length),
            "loadedVia": "path" if chosen_path else "key",
            "status": (match or {}).get("status"),
        }

    def load_model(self, input_string, model_search_string, context_length, seed, unload_image_models_first):
        try:
            model_key_to_load, models_found = self.find_matching_model(model_search_string)
            if model_key_to_load in [self.ERROR_NO_MODEL_FOUND, self.ERROR_CANT_CONNECT]:
                return (input_string, model_key_to_load, models_found)

            desired_identifier = self._safe_identifier_for_key(model_key_to_load)

            # Already loaded?
            for r in self._ps_models():
                if r.get("modelKey") == model_key_to_load or r.get("identifier") in (model_key_to_load, desired_identifier):
                    print(f"Model already loaded: {model_key_to_load} (identifier={r.get('identifier')})")
                    self._currently_loaded_model_key = model_key_to_load
                    self._currently_loaded_model_path = r.get("path")
                    self._currently_loaded_identifier = r.get("identifier") or desired_identifier
                    meta = self._build_loaded_metadata(model_key_to_load, None, context_length)
                    print("[lms-node] Loaded Model Info:\n" + json.dumps(meta, indent=2))
                    short_name = (meta.get("path") or model_key_to_load).replace("\\", "/").split("/")[-1]
                    return (input_string, short_name, models_found)

            if self.check_models_loaded():
                print("Unloading LLMs in memory...")
                self.run_command(['lms', 'unload', '--all'])

            if unload_image_models_first:
                print("Unloading ComfyUI image models to save memory (as requested)")
                self.unload_image_models()

            print(f"Loading model by key: {model_key_to_load} (identifier={desired_identifier})")
            chosen_path = None

            key_cmd = [
                'lms', 'load', model_key_to_load,
                '-y',
                '--identifier', desired_identifier,
                '--context-length', str(int(context_length)),
                '--gpu', 'max',
            ]
            key_proc = self.run_command(key_cmd, timeout=600)

            # Fallback: load by path if key load fails
            if key_proc is None:
                chosen_path = self._path_for_model_key(model_key_to_load)
                if chosen_path:
                    print(f"[lms-node] Key load failed. Attempting path load: {chosen_path}")
                    path_cmd = [
                        'lms', 'load', chosen_path,
                        '-y',
                        '--identifier', desired_identifier,
                        '--context-length', str(int(context_length)),
                        '--gpu', 'max',
                    ]
                    key_proc = self.run_command(path_cmd, timeout=600)

            if key_proc is None:
                print(f"Warning: Issue loading model: {model_key_to_load}")
                return (input_string, self.ERROR_NO_MODEL_FOUND, models_found)

            ps_after = self._ps_models()
            match = None
            for r in ps_after:
                if r.get("identifier") == desired_identifier or r.get("modelKey") == model_key_to_load or r.get("identifier") == model_key_to_load:
                    match = r
                    break

            self._currently_loaded_model_key = model_key_to_load
            self._currently_loaded_model_path = (match or {}).get("path") or chosen_path
            self._currently_loaded_identifier = (match or {}).get("identifier") or desired_identifier

            meta = self._build_loaded_metadata(model_key_to_load, chosen_path, context_length)
            print("[lms-node] Loaded Model Info:\n" + json.dumps(meta, indent=2))

            short_name = (meta.get("path") or model_key_to_load).replace("\\", "/").split("/")[-1]
            return (input_string, short_name, models_found)

        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            return (input_string, self.ERROR_CANT_CONNECT, str(e))

class EbuLMStudioUnload:
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
            result = subprocess.run(['lms', 'unload', '--all'], capture_output=True, text=True, check=True)
            print("Command output:", result.stdout)
            if result.stderr:
                print("", result.stderr, file=sys.stderr)
            EbuLMStudioLoadModel._currently_loaded_model_key = None
            EbuLMStudioLoadModel._currently_loaded_model_path = None
            EbuLMStudioLoadModel._currently_loaded_identifier = None
            return (input_string,)
        except Exception as e:
            print(f"[lms-node] unload_all failed: {e}", file=sys.stderr)
            return (input_string,)

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
                "temp": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05, "display": "number"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01, "display": "number"}),
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

    def _auth_headers(self):
        token = (os.environ.get("LMSTUDIO_API_KEY") or
                 os.environ.get("LM_API_TOKEN") or
                 os.environ.get("LM_API_KEY"))
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    def _base_v1_from_url(self, url: str) -> str:
        if not url:
            return "http://127.0.0.1:1234/v1"
        idx = url.find("/v1/")
        if idx != -1:
            return url[:idx + 3]
        if url.endswith("/v1"):
            return url
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}/v1"

    def _pick_model_id(self, url: str):
        ident = EbuLMStudioLoadModel._currently_loaded_identifier
        if ident:
            return ident

        base_v1 = self._base_v1_from_url(url)
        models_url = f"{base_v1}/models"
        try:
            r = requests.get(models_url, headers=self._auth_headers(), timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, list) and items:
                first = items[0]
                if isinstance(first, dict):
                    return first.get("id") or first.get("model") or first.get("identifier")
                if isinstance(first, str):
                    return first
        except Exception:
            return None
        return None

    def generateText(self, prompt, system_message, url, context_length, seed, max_tokens, temp, top_p, utf8_safe_replace):
        return (self.call_api(prompt, system_message, url, context_length, seed, max_tokens, temp, top_p, utf8_safe_replace),)

    def call_api(self, prompt_text, system_message, url, context_length, seed, max_tokens, temp, top_p, utf8_safe_replace):
        payload = {
            "messages": [
                { "role": "system", "content": f"{system_message}\n" },
                { "role": "user", "content": f"{prompt_text}\n" }
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "seed": seed
        }

        model_id = self._pick_model_id(url)
        if model_id:
            payload["model"] = model_id

        start_time = time.time()
        try:
            response = requests.post(url, json=payload, headers=self._auth_headers(), timeout=600)
            if response.status_code == 200:
                result_json = response.json()
                content = None
                try:
                    content = result_json["choices"][0]["message"]["content"]
                except Exception:
                    content = None

                if isinstance(content, list):
                    content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])

                generatedText = content if isinstance(content, str) else json.dumps(result_json, indent=2)

                if utf8_safe_replace:
                    generatedText = self.sanitize_utf8(generatedText)

                elapsed_time = time.time() - start_time
                print(f"\n[Response generation time: {elapsed_time:.2f} seconds]\n")
                return generatedText

            err = f"Error {response.status_code}: {response.text}"
            print(err, file=sys.stderr)

            # Back-compat: if server is older and doesn't require `model`
            if "model" in payload:
                payload2 = dict(payload)
                payload2.pop("model", None)
                try:
                    response2 = requests.post(url, json=payload2, headers=self._auth_headers(), timeout=600)
                    if response2.status_code == 200:
                        rj = response2.json()
                        txt = rj["choices"][0]["message"]["content"]
                        return txt if isinstance(txt, str) else json.dumps(rj, indent=2)
                except Exception:
                    pass

            return err

        except requests.exceptions.RequestException as e:
            error_message = f"Request failed: {str(e)}"
            print(error_message, file=sys.stderr)
            return error_message

class EbuLMStudioUnloadGuider:
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
            result = subprocess.run(['lms', 'unload', '--all'], capture_output=True, text=True, check=True)
            print("Command output:", result.stdout)
            if result.stderr:
                print("", result.stderr, file=sys.stderr)
            EbuLMStudioLoadModel._currently_loaded_model_key = None
            EbuLMStudioLoadModel._currently_loaded_model_path = None
            EbuLMStudioLoadModel._currently_loaded_identifier = None
            return (guider,)
        except Exception as e:
            print(f"[lms-node] unload_all (guider) failed: {e}", file=sys.stderr)
            return (guider,)

class EbuLMStudioBrainstormer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "topic": ("STRING", {"default": "Fantasy Setting Magic Spells"}),
                "for_each_idea": ("STRING", {"default": "A magic spell that a wizard or sorceress might cast in a fantasy novel. I want the name of the spell and a brief description of what it does.", "multiline": True}),
                "raw_list_size": ("INT", {"default": 20, "min": 1}),
                "return_list_size": ("INT", {"default": 10, "min": 0}),
                "ignore_the_first": ("INT", {"default": 0, "min": 0}),
                "additional_notes_1": ("STRING", {"default": "", "multiline": True}),
                "additional_notes_2": ("STRING", {"default": "", "multiline": True}),
                "url": ("STRING", {"multiline": False, "default": "http://127.0.0.1:1234/v1/chat/completions"}),
                "context_length": ("INT", {"default": 4096, "min": 512, "max": 65536}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 300, "min": 10, "max": 100000}),
                "temp": ("FLOAT", {"default": 0.70, "min": 0.00, "max": 3.00, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.00, "max": 1.00, "step": 0.01}),
                "utf8_safe_replace": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "full_list")
    FUNCTION = "brainstorm"
    CATEGORY = "LMStudio"

    def brainstorm(self, topic, for_each_idea, raw_list_size, return_list_size, ignore_the_first,
                  additional_notes_1, additional_notes_2,
                  url, context_length, seed, max_tokens, temp, top_p, utf8_safe_replace):

        system_message = (
            "You are an expert brainstorming assistant whose sole job is to produce numbered lists of suggestions. "
            "Respond with the list only—no titles, no commentary, no extra text. Each suggestion must be on one line."
        )

        prompt = (
            f"Generate exactly {raw_list_size} ideas on the topic '{topic}'.\n\n"
            f"For each idea, provide exactly this information and no more: {for_each_idea}.\n"
            "Do not include any additional context, background, or commentary beyond that.\n"
            f"Return exactly {raw_list_size} lines. Each line must begin with '1. ', '2. ', etc., followed by the suggestion. "
            "There must be a real line break after each item, and no literal '\\n' sequences."
        )

        notes = []
        if additional_notes_1.strip() or additional_notes_2.strip():
            notes.append("=== ADDITIONAL NOTES ===")
            if additional_notes_1.strip():
                notes.append(additional_notes_1.strip())
            if additional_notes_2.strip():
                notes.append(additional_notes_2.strip())
            prompt += "\n\n" + "\n".join(notes)

        make_request = EbuLMStudioMakeRequest()
        full_response = make_request.generateText(
            prompt, system_message,
            url, context_length, seed,
            max_tokens, temp, top_p,
            utf8_safe_replace
        )[0]
        full_list = full_response.strip()

        lines = [line for line in full_list.splitlines() if line.strip()]
        if ignore_the_first > 0:
            if ignore_the_first >= len(lines):
                raise ValueError(f"ignore_the_first ({ignore_the_first}) must be less than the total number of ideas ({len(lines)})")
            lines = lines[ignore_the_first:]

        if return_list_size > len(lines):
            raise ValueError(f"return_list_size ({return_list_size}) cannot be greater than available ideas ({len(lines)}) after ignoring the first {ignore_the_first}")

        if return_list_size == 0:
            result = ""
        elif return_list_size == 1:
            choice = random.choice(lines)
            result = re.sub(r'^\s*\d+[\.|\)]\s*', '', choice).strip()
        else:
            sampled = random.sample(lines, return_list_size)
            stripped = [re.sub(r'^\s*\d+[\.|\)]\s*', '', l).strip() for l in sampled]
            result_lines = [f"{i+1}. {stripped[i]}" for i in range(len(stripped))]
            result = "\n".join(result_lines)

        return (result, full_list)

NODE_CLASS_MAPPINGS = {
    "EbuLMStudioLoadModel": EbuLMStudioLoadModel,
    "EbuLMStudioUnload": EbuLMStudioUnload,
    "EbuLMStudioMakeRequest": EbuLMStudioMakeRequest,
    "EbuLMStudioUnloadGuider": EbuLMStudioUnloadGuider,
    "EbuLMStudioBrainstormer": EbuLMStudioBrainstormer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EbuLMStudioLoadModel": "EBU LMStudio Load",
    "EbuLMStudioUnload": "EBU LMStudio Unload All",
    "EbuLMStudioMakeRequest": "EBU LMStudio Make Request",
    "EbuLMStudioUnloadGuider": "EBU LMStudio Unload (Guider)",
    "EbuLMStudioBrainstormer": "EBU LMStudio Brainstormer",
}