import subprocess
import sys
import re
import comfy.model_management as model_management
import gc
import torch
import requests
import time
import random
import json

class EbuLMStudioLoadModel:
    ERROR_NO_MODEL_FOUND = "no model by that name found"
    ERROR_CANT_CONNECT = "can't connect to lmstudio"
    _currently_loaded_model_path = None  # Best-effort; may be a key if no path is available

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

    # ---------- low-level helpers ----------

    def run_command(self, command, timeout=180):
        import subprocess, sys
        try:
            process = subprocess.run(
                [str(c) for c in command],
                capture_output=True,
                text=True,
                encoding="utf-8",   # <- force UTF-8
                errors="replace",   # <- avoid decode crashes
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

    def _list_models(self):
        """
        Return ('json'|'text', stdout) from an `lms ls` variant, or (None,"") on failure.
        """
        p = self.run_command(['lms', 'ls', '--json'])
        if p and p.stdout and p.stdout.strip().startswith(('[', '{')):
            return 'json', p.stdout

        p = self.run_command(['lms', 'ls'])
        if p and p.stdout:
            return 'text', p.stdout

        # Legacy/alt
        p = self.run_command(['lms', 'models', 'ls'])
        if p and p.stdout:
            return 'text', p.stdout

        return None, ""

    def _models_index(self):
        """Return list of JSON model dicts (or empty list)."""
        import json
        p = self.run_command(['lms', 'ls', '--json'])
        if not (p and p.stdout and p.stdout.strip().startswith(('[', '{'))):
            return []
        try:
            data = json.loads(p.stdout)
            items = data if isinstance(data, list) else data.get('models', [])
            if isinstance(items, dict):
                items = items.get('models', [])
            return items if isinstance(items, list) else []
        except Exception:
            return []

    def _model_info_for_key(self, key):
        """Return the model JSON dict for modelKey (or None)."""
        for m in self._models_index():
            if isinstance(m, dict) and m.get('modelKey') == key:
                return m
        return None

    def _path_for_model_key(self, key):
        """Return GGUF path for a modelKey using JSON (or None)."""
        m = self._model_info_for_key(key)
        return (m or {}).get('path')

    def _parse_keys_from_text(self, s):
        """
        Parse first column under the 'LLM' header in `lms ls` table output.
        """
        import re
        keys = []
        header_seen = False
        for raw in (s or "").splitlines():
            line = raw.rstrip()
            if not line.strip():
                continue
            if not header_seen:
                if re.match(r'^\s*LLM\s+', line):
                    header_seen = True
                continue
            if re.match(r'^\s*EMBEDDING\s+', line):
                break
            m = re.match(r'^(\S[^\s].*?)\s{2,}', line)
            if m:
                keys.append(m.group(1).strip())
        return keys

    def _get_loaded_ps_stdout(self):
        p = self.run_command(['lms', 'ps'])
        return p.stdout if p else ""

    def _ps_rows(self):
        """
        Parse `lms ps` into rows of {identifier, model, status, size, context, ttl}.
        Very simple column-split on 2+ spaces.
        """
        import re
        out = []
        ps = self._get_loaded_ps_stdout()
        lines = [ln for ln in (ps or "").splitlines() if ln.strip()]
        if len(lines) < 2:
            return out
        # skip header line
        for ln in lines[1:]:
            cols = re.split(r'\s{2,}', ln.strip())
            if len(cols) >= 2:
                row = {
                    "identifier": cols[0],
                    "model": cols[1],
                    "status": cols[2] if len(cols) > 2 else "",
                    "size": cols[3] if len(cols) > 3 else "",
                    "context": cols[4] if len(cols) > 4 else "",
                    "ttl": cols[5] if len(cols) > 5 else "",
                }
                out.append(row)
        return out

    # ---------- compatibility shim ----------

    def get_currently_loaded_model_path(self):
        """
        Best-effort: returns the MODEL key from `lms ps`.
        """
        rows = self._ps_rows()
        return rows[0]["model"] if rows else None

    def check_models_loaded(self):
        """Check if any models are currently loaded in LM Studio (via `lms ps`)."""
        return bool(self._ps_rows())

    def unload_image_models(self):
        import gc, torch
        from comfy import model_management  # assumes ComfyUI env
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

    # ---------- search (spaces = wildcards) ----------

    def find_matching_model(self, search_term):
        """
        Returns (chosen_model_key, raw_listing).
        - Case-insensitive
        - Spaces act like wildcards (tokens may appear anywhere, in any order)
        - Searches across key, displayName, path, architecture, paramsString, quantization.name/bits
        On failure: (ERROR_*, raw_listing_or_reason)
        """
        import re, random, json, sys

        try:
            # allow "a|b|c" forms
            if '|' in search_term:
                options = [s.strip() for s in search_term.split('|') if s.strip()]
                if options:
                    search_term = random.choice(options)
                    print(f"Multiple search strings provided. Randomly selected: '{search_term}'")

            print("Fetching model list...")
            mode, listing = self._list_models()
            if not listing:
                return self.ERROR_CANT_CONNECT, "No output from `lms ls` variants."

            # Prefer JSON (richer matching)
            if mode != 'json':
                pjson = self.run_command(['lms', 'ls', '--json'])
                if pjson and pjson.stdout and pjson.stdout.strip().startswith(('[', '{')):
                    mode, listing = 'json', pjson.stdout

            tokens = [t for t in re.split(r'\s+', (search_term or '').strip().lower()) if t]
            if not tokens:
                return self.ERROR_NO_MODEL_FOUND, listing

            if mode == 'json':
                try:
                    data = json.loads(listing)
                    items = data if isinstance(data, list) else data.get('models', [])
                    if isinstance(items, dict):
                        items = items.get('models', [])
                except Exception as e:
                    print(f"[lms-node] JSON parse error (matching): {e}", file=sys.stderr)
                    items = []

                candidates = []
                for m in items or []:
                    if not isinstance(m, dict):
                        continue
                    key  = (m.get('modelKey') or m.get('key') or m.get('name') or
                            m.get('identifier') or m.get('displayName') or "")
                    disp = m.get('displayName', "") or ""
                    path = m.get('path', "") or ""
                    arch = m.get('architecture', "") or ""
                    params = m.get('paramsString', "") or ""
                    q = m.get('quantization') or {}
                    qname = (q.get('name') if isinstance(q, dict) else "") or ""
                    qbits = (str(q.get('bits')) if isinstance(q, dict) and q.get('bits') is not None else "")

                    hay = " ".join([str(key), disp, path, arch, params, qname, qbits]).lower()

                    # all tokens must appear
                    tok_hits = [t in hay for t in tokens]
                    score = sum(tok_hits)
                    if score == len(tokens):
                        in_order = bool(re.search(".*".join(map(re.escape, tokens)), hay))
                        candidates.append({
                            "key": str(m.get('modelKey', key)),
                            "hay": hay,
                            "in_order": in_order,
                            "key_len": len(str(m.get('modelKey', key)) or ""),
                            "hay_len": len(hay),
                        })

                if not candidates:
                    print(f"No models found matching '{search_term}'.")
                    return self.ERROR_NO_MODEL_FOUND, listing

                candidates.sort(key=lambda c: (c["in_order"], c["key_len"], c["hay_len"]), reverse=True)
                return candidates[0]["key"], listing

            else:
                # Text fallback over parsed keys only
                keys = self._parse_keys_from_text(listing)
                if not keys:
                    print("[lms-node] Parser could not extract any model keys from text listing.", file=sys.stderr)
                    return self.ERROR_NO_MODEL_FOUND, listing

                def all_tokens_in(s: str) -> bool:
                    s = s.lower()
                    return all(t in s for t in tokens)

                hits = [k for k in keys if all_tokens_in(k)]
                if not hits:
                    print(f"No models found matching '{search_term}'.")
                    return self.ERROR_NO_MODEL_FOUND, listing

                return max(hits, key=len), listing

        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return self.ERROR_CANT_CONNECT, ""

    # ---------- load + rich metadata ----------

    def _build_loaded_metadata(self, model_key, chosen_path, context_length):
        """
        Build a rich metadata dict for the loaded model, including identifier & load method.
        """
        import os, re, copy

        info = self._model_info_for_key(model_key) or {}
        meta = {
            "modelKey": model_key,
            "displayName": info.get("displayName"),
            "path": info.get("path") or chosen_path,
            "architecture": info.get("architecture"),
            "paramsString": info.get("paramsString"),
            "quantization": info.get("quantization"),
            "maxContextLength": info.get("maxContextLength"),
            "loadedContextLength": int(context_length),
            "loadedVia": "path" if chosen_path else "key",
            "identifier": None,   # filled from `lms ps`
            "status": None,
            "size": None,
        }

        # Try to extract current identifier/status/size/context from ps
        rows = self._ps_rows()
        # Heuristics to find the matching row
        match = None
        if rows:
            # prefer model key match
            for r in rows:
                if r.get("model") and model_key in r["model"]:
                    match = r
                    break
            # else, fallback: match by chosen_path filename stem
            if not match and chosen_path:
                stem = os.path.basename(chosen_path)
                stem = re.sub(r'\.gguf$', '', stem, flags=re.IGNORECASE)
                for r in rows:
                    if r.get("model") and (stem in r["model"]):
                        match = r
                        break
        if match:
            meta["identifier"] = match.get("identifier")
            meta["status"] = match.get("status")
            meta["size"] = match.get("size")
            # keep ps's reported context if present
            try:
                meta["reportedContext"] = int(match.get("context")) if match.get("context") else None
            except Exception:
                meta["reportedContext"] = match.get("context")

        return meta

    def load_model(self, input_string, model_search_string, context_length, seed, unload_image_models_first):
        import sys, re, json, os
        try:
            # 1) Resolve candidate key using your wildcard search
            model_key_to_load, models_found = self.find_matching_model(model_search_string)
            if model_key_to_load in [self.ERROR_NO_MODEL_FOUND, self.ERROR_CANT_CONNECT]:
                return (input_string, model_key_to_load, models_found)

            # 2) If already loaded, short-circuit (but still emit rich logs)
            if any(model_key_to_load in r.get("model", "") for r in self._ps_rows()):
                print(f"Model '{model_key_to_load}' already loaded.")
                self._currently_loaded_model_path = model_key_to_load
                meta = self._build_loaded_metadata(model_key_to_load, None, context_length)
                print("[lms-node] Loaded Model Info:\n" + json.dumps(meta, indent=2))
                short_name = None
                if meta.get("path"):
                    short_name = meta["path"].replace("\\", "/").split("/")[-1]
                if not short_name:
                    short_name = model_key_to_load
                return (input_string, short_name, models_found)

            # 3) Prep memory
            if self.check_models_loaded():
                print("Unloading LLMs in memory...")
                self.run_command(['lms', 'unload', '--all'])

            if unload_image_models_first:
                print("Unloading ComfyUI image models to save memory (as requested)")
                self.unload_image_models()

            # 4) Load by KEY (normal case). If it fails or key contains '@', attempt PATH fallback explicitly.
            print(f"Loading model by key: {model_key_to_load}")
            key_proc = None
            path_proc = None
            chosen_path = None

            # Attempt by key unless we *know* '@' keys won't work and you prefer to try anyway for visibility.
            key_cmd = ['lms', 'load', model_key_to_load, '-y',
                       '--context-length', str(int(context_length)), '--gpu', 'max']
            key_proc = self.run_command(key_cmd, timeout=600)

            # Clear, explicit fallback logging
            if key_proc is None or ('@' in model_key_to_load):
                print(f"[lms-node] Key load failed or contained '@'. Attempting PATH fallback...")
                chosen_path = self._path_for_model_key(model_key_to_load)
                if chosen_path:
                    print(f"[lms-node] Fallback path resolved: {chosen_path}")
                    path_cmd = ['lms', 'load', '--exact', '-y',
                                '--context-length', str(int(context_length)), '--gpu', 'max',
                                chosen_path]
                    path_proc = self.run_command(path_cmd, timeout=600)
                    if path_proc:
                        print(f"[lms-node] PATH fallback succeeded for {model_key_to_load}")
                    else:
                        print(f"[lms-node] PATH fallback FAILED for {model_key_to_load}", file=sys.stderr)
                else:
                    print(f"[lms-node] No path found for key '{model_key_to_load}' in JSON listing.", file=sys.stderr)

            if not (key_proc or path_proc):
                print(f"Warning: Issue loading model (key and path attempts failed): {model_key_to_load}")
                return (input_string, self.ERROR_NO_MODEL_FOUND, models_found)

            # 5) Verify via `lms ps`
            rows_after = self._ps_rows()
            loaded_ok = any(model_key_to_load in r.get("model", "") for r in rows_after)
            if not loaded_ok and chosen_path:
                stem = os.path.basename(chosen_path)
                stem = re.sub(r'\.gguf$', '', stem, flags=re.IGNORECASE)
                loaded_ok = any(stem in r.get("model", "") for r in rows_after)

            # 6) Build and print verbose metadata
            meta = self._build_loaded_metadata(model_key_to_load, chosen_path, context_length)
            if loaded_ok:
                print(f"Model loaded: {model_key_to_load}")
                self._currently_loaded_model_path = model_key_to_load
                print("[lms-node] Loaded Model Info:\n" + json.dumps(meta, indent=2))
            else:
                print("Warning: Model load attempted, but not visible in `lms ps`.")
                print("[lms-node] Post-load Model Info (not confirmed in ps):\n" + json.dumps(meta, indent=2))

            # 7) Return: only the filename portion of the path (fallback to key if missing)
            short_name = None
            if meta.get("path"):
                short_name = meta["path"].replace("\\", "/").split("/")[-1]
            if not short_name:
                short_name = model_key_to_load

            return (input_string, short_name, models_found)

        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            return (input_string, self.ERROR_CANT_CONNECT, str(e))

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
        # Hard-coded system message
        system_message = (
            "You are an expert brainstorming assistant whose sole job is to produce numbered lists of suggestions. "
            "Respond with the list only—no titles, no commentary, no extra text. Each suggestion must be on one line."
        )
        # Base prompt with strict list instruction and positive/negative framing
        prompt = (
            f"Generate exactly {raw_list_size} ideas on the topic '{topic}'.\n\n"
            f"For each idea, provide exactly this information and no more: {for_each_idea}.\n"
            f"Each list item must consist of exactly the information specified above. "
            "Do not include any additional context, background, or commentary beyond that.\n"
            f"Return exactly {raw_list_size} lines. Each line must begin with '1. ', '2. ', etc., followed by the suggestion. "
            "There must be a real line break after each item, and no literal '\\n' sequences."
        )
        # Append additional notes if provided
        notes = []
        if additional_notes_1.strip() or additional_notes_2.strip():
            notes.append("=== ADDITIONAL NOTES ===")
            if additional_notes_1.strip():
                notes.append(additional_notes_1.strip())
            if additional_notes_2.strip():
                notes.append(additional_notes_2.strip())
            prompt += "\n\n" + "\n".join(notes)

        # Call LMStudio via existing node logic
        make_request = EbuLMStudioMakeRequest()
        full_response = make_request.generateText(
            prompt, system_message,
            url, context_length, seed,
            max_tokens, temp, top_p,
            utf8_safe_replace
        )[0]
        full_list = full_response.strip()

        # Process full_list into result
        lines = [line for line in full_list.splitlines() if line.strip()]
        # Skip the first N lines if requested
        if ignore_the_first > 0:
            if ignore_the_first >= len(lines):
                raise ValueError(f"ignore_the_first ({ignore_the_first}) must be less than the total number of ideas ({len(lines)})")
            lines = lines[ignore_the_first:]
        # Validate return_list_size against available lines
        if return_list_size > len(lines):
            raise ValueError(f"return_list_size ({return_list_size}) cannot be greater than available ideas ({len(lines)}) after ignoring the first {ignore_the_first}")
        # Generate result
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
