"""
Metadata Processor for ETL Pipeline

Extracts structured metadata from model descriptions.
This module is integrated into the ETL pipeline to avoid dependencies on SEA-App.

Location: mlentory-etl-pipeline/code/load/mlentory_load/core/metadata_processor.py
"""

import json
import re
from html import unescape
from pathlib import Path
from typing import Optional, Dict, Any, List

# Optional spaCy import for NER/POS-based fallbacks
try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.pipeline import EntityRuler
except Exception:  # pragma: no cover
    spacy = None
    Matcher = None
    EntityRuler = None


def strip_markup(text: str) -> str:
    """Remove most markdown/HTML while keeping plain text."""
    # Remove fenced code blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove images ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    # Convert links [text](url) -> 'text url'
    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1 \2", text)
    # Remove residual HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def load_nlp():
    if spacy is None:
        return None
    try:
        # Try to reuse a cached singleton on the module
        global _NLP
        if '_NLP' in globals() and _NLP is not None:
            return _NLP
        _NLP = spacy.load('en_core_web_sm')
        # Add EntityRuler with light patterns (non-invasive)
        ruler = _NLP.add_pipe("entity_ruler", config={"overwrite_ents": False})
        patterns = [
            {"label": "ARCH", "pattern": "RoPE"},
            {"label": "ARCH", "pattern": "RMSNorm"},
            {"label": "ARCH", "pattern": "SwiGLU"},
            {"label": "ARCH", "pattern": "QKV"},
            {"label": "ARCH", "pattern": "Transformer"},
            {"label": "ARCH", "pattern": "Transformers"},
            {"label": "ARCH", "pattern": "SDXL"},
            {"label": "ARCH", "pattern": "LLaMA"},
            {"label": "ARCH", "pattern": "Mixtral"},
            {"label": "ARCH", "pattern": "Illustrious"},
            {"label": "DATASET", "pattern": "Corpus"},
            {"label": "DATASET", "pattern": "Dataset"},
            {"label": "DATASET", "pattern": "Datasets"},
            {"label": "DATASET", "pattern": "Data Source"},
            {"label": "DATASET", "pattern": "Training Data"},
        ]
        ruler.add_patterns(patterns)
        return _NLP
    except Exception:
        return None

def spacy_extract_architecture(text: str) -> Optional[str]:
    nlp = load_nlp()
    if nlp is None:
        return None
    doc = nlp(text)
    # Heuristic 1: find sentences with label cue and take NP after colon
    for sent in doc.sents:
        s = sent.text.strip()
        if re.search(r"\bArchitecture\b\s*[:\-—–]", s, re.IGNORECASE):
            # Take first NOUN/PROPN sequence after the label
            after = re.split(r"\bArchitecture\b\s*[:\-—–]", s, flags=re.IGNORECASE, maxsplit=1)
            if len(after) == 2:
                tail = after[1].strip()
                # Return concise tail
                return tail.split(' - ')[0][:200].strip() or None
    # Heuristic 2: look for verbs like based/finetuned and capture following proper-noun chunk
    for sent in doc.sents:
        if re.search(r"\b(based|finetune|fine\-tune|retrained|derived)\b", sent.text, re.IGNORECASE):
            # Collect contiguous PROPN/UPPER tokens as candidate
            tokens = list(sent)
            for i, t in enumerate(tokens):
                if t.lower_ in {"based", "finetune", "fine-tune", "retrained", "derived"}:
                    # Look ahead for prepositions and then proper noun span
                    j = i
                    while j < len(tokens) and tokens[j].lower_ not in {"from", "of", "on"}:
                        j += 1
                    if j < len(tokens):
                        k = j + 1
                        start = k
                        while k < len(tokens) and (tokens[k].pos_ in {"PROPN", "NOUN"} or tokens[k].shape_ == tokens[k].shape_.upper()):
                            k += 1
                        if k > start:
                            span = sent[start:k].text.strip()
                            if span:
                                return span[:120]
    return None

def spacy_extract_dataset(text: str) -> Optional[str]:
    nlp = load_nlp()
    if nlp is None:
        return None
    doc = nlp(text)
    # Prefer sentences with dataset/corpus/source cues
    for sent in doc.sents:
        if re.search(r"\b(dataset|datasets|corpus|data\s*source|data\s*sources|training\s*data|training\s*set)\b",
                     sent.text, re.IGNORECASE):
            # Extract dominant proper-noun chunk not including numeric-only spans
            proper_chunks = []
            chunk = []
            for tok in sent:
                if tok.pos_ in {"PROPN", "NOUN"} and not tok.like_num:
                    chunk.append(tok)
                else:
                    if chunk:
                        text_chunk = doc[chunk[0].i:chunk[-1].i+1].text.strip()
                        proper_chunks.append(text_chunk)
                        chunk = []
            if chunk:
                text_chunk = doc[chunk[0].i:chunk[-1].i+1].text.strip()
                proper_chunks.append(text_chunk)
            # Return the first plausible chunk (short and not generic words)
            generic_terms = {"dataset", "datasets", "data", "corpus", "source", "sources", "pictures", "images", "tokens", "samples"}
            for c in proper_chunks:
                c_norm = c.strip().strip('-–—:,')
                # Exclude chunks containing digits or unit-only tokens
                if any(ch.isdigit() for ch in c_norm):
                    continue
                if re.search(r"\b([KMGT]B?|M)\b", c_norm, re.IGNORECASE):
                    continue
                # Exclude pure generic terms
                if c_norm.lower() in generic_terms:
                    continue
                # Require at least one uppercase letter and alphabetic ratio
                if not re.search(r"[A-Z]", c_norm):
                    continue
                letters = sum(ch.isalpha() for ch in c_norm)
                if letters == 0 or letters / max(1, len(c_norm)) < 0.6:
                    continue
                # Prefer multi-word proper names but allow single well-formed tokens
                return c_norm
    return None


def load_modality_config(config_path: Optional[str] = None) -> Dict[str, List[str]]:
    """Load modality inference configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "field_config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("modality_inference", {})
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to default configuration
        return {
            "text_keywords": ["text", "language", "nlp", "conversation", "chat", "generation", "translation", "summarization", "sentiment"],
            "image_keywords": ["image", "vision", "visual", "object-detection", "segmentation", "object", "face"],
            "audio_keywords": ["audio", "speech", "voice", "sound", "music", "acoustic"],
            "video_keywords": ["video", "motion", "action", "frame"],
            "time_series_keywords": ["time-series", "time series", "forecasting", "prediction", "temporal-data", "sequential-data"],
            "multimodal_keywords": ["multimodal", "cross-modal", "crossmodal", "multi-modal"]
        }

def load_architecture_patterns(config_path: Optional[str] = None) -> Dict[str, List[str]]:
    """Load architecture detection patterns from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "field_config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("architecture_patterns", {})
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to comprehensive default patterns
        return {
            "transformer_variants": [
                "Transformer", "BERT", "GPT", "T5", "BART", "RoBERTa", "ELECTRA", 
                "ALBERT", "DeBERTa", "XLNet", "Reformer", "Longformer", "BigBird",
                "Perceiver", "Synthesizer", "Linformer", "Performer"
            ],
            "llm_families": [
                "LLaMA", "Llama", "Alpaca", "Vicuna", "Mistral", "Mixtral", "Qwen",
                "Phi", "Gemma", "ChatGLM", "Baichuan", "Yi", "Falcon", "MPT",
                "StableLM", "RedPajama", "OpenLLaMA", "CodeLlama", "WizardLM",
                "Orca", "Platypus", "Guanaco", "GPT-J", "GPT-NeoX", "OPT", "BLOOM"
            ],
            "vision_models": [
                "ViT", "Vision Transformer", "DeiT", "BEiT", "MAE", "DINO", "CLIP",
                "Swin", "ConvNeXt", "ResNet", "EfficientNet", "MobileNet", "DenseNet",
                "VGG", "Inception", "NASNet", "RegNet", "CvT", "PVT", "Twins",
                "InternVL", "InternViT", "Pix2Struct", "Donut"
            ],
            "diffusion_models": [
                "Stable Diffusion", "SDXL", "SD", "Latent Diffusion", "DALL-E",
                "Imagen", "Midjourney", "ControlNet", "LoRA", "DreamBooth",
                "Illustrious", "Flux", "Cascade"
            ],
            "multimodal_models": [
                "BLIP", "Flamingo", "Kosmos", "PaLM-E", "GPT-4V", "LLaVA",
                "InstructBLIP", "MiniGPT", "Otter", "Qwen-VL", "CogVLM",
                "InternLM-XComposer", "mPLUG"
            ],
            "audio_models": [
                "Whisper", "Wav2Vec", "HuBERT", "WavLM", "UniSpeech", "CLAP",
                "AudioLM", "MusicGen", "AudioGen", "EnCodec"
            ],
            "rl_models": [
                "PPO", "DQN", "A3C", "SAC", "TD3", "TRPO", "Actor-Critic",
                "Rainbow", "Ape-X", "R2D2"
            ],
            "architecture_components": [
                "RoPE", "RMSNorm", "LayerNorm", "SwiGLU", "GeLU", "QKV",
                "Multi-Head Attention", "Cross-Attention", "Self-Attention",
                "FlashAttention", "GroupedQuery", "MoE", "Mixture of Experts"
            ]
        }

def infer_modalities_from_task(task: str) -> List[str]:
    """
    Infer modalities from task name using configurable patterns.
    This replaces the hardcoded modalities_map with dynamic inference.
    """
    task_lower = task.lower()
    modalities = []
    
    # Load configuration
    modality_config = load_modality_config()
    
    # Text-related tasks
    if any(keyword in task_lower for keyword in modality_config.get("text_keywords", [])):
        modalities.append('text')
    
    # Image-related tasks
    if any(keyword in task_lower for keyword in modality_config.get("image_keywords", [])):
        modalities.append('image')
    
    # Audio-related tasks
    if any(keyword in task_lower for keyword in modality_config.get("audio_keywords", [])):
        modalities.append('audio')
    
    # Video-related tasks
    if any(keyword in task_lower for keyword in modality_config.get("video_keywords", [])):
        modalities.append('video')
    
    # Time series-related tasks
    if any(keyword in task_lower for keyword in modality_config.get("time_series_keywords", [])):
        modalities.append('time-series')
    
    # Multimodal tasks
    if any(keyword in task_lower for keyword in modality_config.get("multimodal_keywords", [])):
        # For multimodal tasks, we need to infer the specific modalities
        # This is a heuristic - in practice, you might want to look at the description
        if 'text' not in modalities:
            modalities.append('text')
        if 'image' not in modalities:
            modalities.append('image')
    
    # Cross-modal tasks (e.g., text-to-image, image-to-text)
    if 'to' in task_lower:
        parts = task_lower.split('to')
        if len(parts) == 2:
            source_modality = parts[0].strip()
            target_modality = parts[1].strip()
            
            # Map common terms to modalities
            if source_modality in ['text', 'language']:
                modalities.append('text')
            elif source_modality in ['image', 'visual']:
                modalities.append('image')
            elif source_modality in ['audio', 'speech']:
                modalities.append('audio')
                
            if target_modality in ['text', 'language']:
                modalities.append('text')
            elif target_modality in ['image', 'visual']:
                modalities.append('image')
            elif target_modality in ['audio', 'speech']:
                modalities.append('audio')
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(modalities))


def run_extraction(data: Dict[str, Any]) -> Dict[str, Any]:

    desc = data.get('description') or ''
    plain = strip_markup(desc)
    # Also build a version that preserves line breaks for line-wise label parsing
    def strip_markup_keep_newlines(text: str) -> str:
        t = re.sub(r"```[\s\S]*?```", " ", text)
        t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
        t = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1 \2", t)
        t = re.sub(r"<[^>]+>", " ", t)
        t = unescape(t)
        # Collapse spaces but keep newlines
        # Normalize Windows/Mac newlines first
        t = t.replace('\r\n', '\n').replace('\r', '\n')
        # Collapse spaces per line
        t = "\n".join(re.sub(r"\s+", " ", line).strip() for line in t.split("\n"))
        # Remove consecutive blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()
    lines_text = strip_markup_keep_newlines(desc)

    result = {
        'version': None,
        'modalities': None,
        'domain': None,
        'architecture': None,
        'modelSize': None,
        'dataset': None,
        'trainingType': None,
    }

    # Prepare RL method names to avoid misclassification as architectures and enrich training type
    def _load_rl_methods_local() -> List[str]:
        patterns = load_architecture_patterns()
        rl = set([m.strip() for m in patterns.get('rl_models', [])])
        rl.update({
            'RLHF', 'RLAIF', 'DPO', 'GRPO', 'GSPO', 'PPO', 'TRPO', 'A3C', 'SAC', 'TD3',
            'DeepQ', 'V-MPO', 'VMPO', 'Muesli', 'Dreamer', 'DreamerV3', 'IMPALA', 'Rainbow', 'Ape-X', 'R2D2', 'GRPO'
        })
        return sorted(rl, key=len, reverse=True)

    rl_methods = set(_load_rl_methods_local())

    # Get name from original data only for pattern matching (not returned)
    name = data.get('name')
    keywords = data.get('keywords') or []

    # Version: derive from name or description
    version = None
    if isinstance(name, str):
        # Pattern 1: Version at end of name (e.g., "model-v1.2" or "model_1.5")
        m = re.search(r"(?:[-_/ ]v?)(\d+\.\d+(?:\.\d+)?)$", name, re.IGNORECASE)
        if m:
            version = m.group(1)
        # Pattern 2: Version within model name with underscore (e.g., "InternVL3_5-1B" -> "3.5")
        if not version:
            m = re.search(r'(\d+)_(\d+)[-_]', name)
            if m:
                version = f"{m.group(1)}.{m.group(2)}"
        # Pattern 3: Version within model name with dot (e.g., "Qwen2.5-0.5B" -> "2.5")
        if not version:
            m = re.search(r'\b[A-Za-z]+(\d+\.\d+)[-_]', name)
            if m:
                version = m.group(1)
    if not version:
        # Prefer explicit 'version' or 'ver' labels in text
        m = re.search(r"\b(?:version|ver)\s*(\d+\.\d+(?:\.\d+)?)\b", plain, re.IGNORECASE)
        if not m:
            # Accept standalone 'v' only when not attached to a preceding token (avoid 'MMBench v1.1')
            m = re.search(r"(?<![A-Za-z0-9_])v\s*(\d+\.\d+(?:\.\d+)?)\b", plain, re.IGNORECASE)
        if m:
            version = m.group(1)
    if version:
        result['version'] = version

    # Modalities - infer from mlTask or description
    ml_task = data.get('mlTask')
    if ml_task:
        # mlTask can be a list or a string
        if isinstance(ml_task, list):
            ml_task_str = ' '.join(ml_task)
        else:
            ml_task_str = str(ml_task)
        modalities = infer_modalities_from_task(ml_task_str)
        if modalities:
            result['modalities'] = modalities

    # Domain heuristics - check keywords from original data and description
    domain = None
    domain_candidates = load_modality_config().get("domain_candidates", [
        'anime', 'medical', 'code', 'general', 'art', 'biology', 'chemistry',
        'finance', 'legal', 'multimodal', 'research', 'academic', 'commercial',
        'roleplay', 'rp'  # Added roleplay domains
    ])
    # Check keywords from original data first
    if isinstance(keywords, list):
        for d in domain_candidates:
            for kw in keywords:
                if isinstance(kw, str) and d.lower() == kw.lower():
                    domain = d
                    break
            if domain:
                break
    # If not found in keywords, check description
    if not domain:
        for d in domain_candidates:
            if re.search(rf"\b{re.escape(d)}\b", plain, re.IGNORECASE):
                domain = d
                break
    # Normalize 'rp' to 'roleplay'
    if domain and domain.lower() == 'rp':
        domain = 'roleplay'
    if domain:
        result['domain'] = domain

    # Architecture/base model
    arch = None
    # 0) Try to detect known architecture families directly from the model name first
    try:
        arch_patterns = load_architecture_patterns()
        all_architectures_name = []
        for family, archs in arch_patterns.items():
            if family == 'rl_models':
                continue
            all_architectures_name.extend(archs)
        all_architectures_name.sort(key=len, reverse=True)
        if isinstance(name, str):
            name_pattern = r"\b(" + "|".join(re.escape(a) for a in all_architectures_name) + r")\b"
            m_name = re.search(name_pattern, name, re.IGNORECASE)
            if m_name:
                cand = m_name.group(1)
                if cand not in rl_methods:
                    arch = cand
    except Exception:
        pass

    # 1) Explicit "Architecture:" label in description (line-aware, tolerant to bullets and separators)
    m = re.search(
        r"^\s*(?:[-*•\d+\.)\s]*)?\s*Architecture\s*(?:[:\-—–]|:)\s*(.+)$",
        lines_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if m:
        arch_line = m.group(1).strip()
        # Extract just the architecture name (first word/phrase before additional details)
        # Split on common separators and take first part
        arch = re.split(r'\s+(vision|encoder|with|using|model|architecture)', arch_line, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        arch = arch[:100].strip()  # Keep it concise
    if not arch:
        m = re.search(r"\b(finetune|fine-tune|retrained|based on|built on|derived from)\s+of?\s*([A-Z][\w\- ]{2,})",
                      plain, re.IGNORECASE)
        if m:
            cand = m.group(2).strip()
            # Truncate at common connectors to avoid swallowing descriptive tails
            cand = re.split(r"\s+(with|using|for|on)\b", cand, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            # Remove trailing qualifiers
            cand = re.sub(r"\s+(model|xl|v\d[\w\.]*)$", "", cand, flags=re.IGNORECASE).strip()
            if cand and cand not in rl_methods:
                arch = cand
        else:
            # Load configurable architecture patterns
            arch_patterns = load_architecture_patterns()
            # Build comprehensive pattern from all architecture families
            all_architectures = []
            for family, archs in arch_patterns.items():
                if family == 'rl_models':
                    continue
                all_architectures.extend(archs)
            
            # Sort by length (longest first) to match more specific names first
            all_architectures.sort(key=len, reverse=True)
            
            # Create regex pattern (escape special characters)
            escaped_archs = [re.escape(arch) for arch in all_architectures]
            pattern = r"\b(" + "|".join(escaped_archs) + r")\b"
            
            m = re.search(pattern, plain, re.IGNORECASE)
            if m:
                cand = m.group(1)
                if cand not in rl_methods:
                    arch = cand
            
            # Additional pattern: Look for common architecture naming patterns
            # e.g., "NameV2", "Name-XL", "Name-Base", "NameForCausalLM"
            if not arch:
                m = re.search(
                    r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*(?:-?(?:V\d+|XL|Base|Large|Small|Tiny|Mini)))(?:\s|$|For)',
                    plain
                )
                if m:
                    arch = m.group(1)
    if not arch:
        # spaCy fallback
        arch = spacy_extract_architecture(lines_text)
    # Exclude RL methods mistakenly set as architecture
    if arch and arch in rl_methods:
        arch = None
    if arch:
        result['architecture'] = arch

    # Model size: prefer name-embedded tokens first, then labeled lines, then totals
    model_size = None
    # Derive from name if it includes a size token like -7B or -0.5B
    if not model_size and isinstance(name, str):
        m = re.search(r"[-_](\d+(?:\.\d+)?)\s*(B|M|K)\b", name, re.IGNORECASE)
        if m:
            model_size = f"{m.group(1)}{m.group(2).upper()}"
    # Explicit label (line-aware, tolerant to bullets, separators, and common typos)
    if not model_size:
        m = re.search(
            r"^\s*(?:[-*•\d+\.)\s]*)?\s*(Number of\s+Parameters|Number of\s+Paramaters|Params|Parameters|#Total\s+Param)\s*(?:[:\-—–]|:)\s*(.+)$",
            lines_text,
            re.IGNORECASE | re.MULTILINE,
        )
    if m and not model_size:
        model_size_line = m.group(2).strip()
        # Prefer compact token like 0.49B or 7B
        m2 = re.search(r"\b(\d+(?:\.\d+)?)\s*(B|M|K)\b", model_size_line, re.IGNORECASE)
        model_size = (m2.group(1) + m2.group(2).upper()) if m2 else model_size_line.split(' - ')[0][:100]
    
    # Pattern: "X.XB total" or similar variations
    if not model_size:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(B|M|K)\s+(?:total|params?|parameters?)", plain, re.IGNORECASE)
        if m:
            model_size = f"{m.group(1)}{m.group(2).upper()}"

    if model_size:
        result['modelSize'] = model_size

    # Dataset name only: prefer explicit labels; otherwise leave null
    dataset = None
    # Labeled lines: Dataset(s)/Name, Training Dataset/Data/Corpus/Set, Pretraining Data/Corpus,
    # Data Source(s), Training set, Pretrain set, etc.
    m = re.search(
        r"^\s*(?:[-*•\d+\.)\s]*)?\s*("
        r"Datasets?|Dataset\s*Name|Training\s*Dataset|Training\s*Data|Training\s*Corpus|Training\s*Set|"
        r"Pretraining\s*Data|Pretraining\s*Corpus|Pretrain(?:ing)?\s*Set|Data\s*Source(?:s)?|Data\s*Sources|"
        r"Corpus|Pretrain\s*Corpus"
        r")\s*(?:[:\-—–]|:)\s*(.+)$",
        lines_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if m:
        ds_line = m.group(2).strip()
        # Stop at common sentence delimiters (but keep hyphens in dataset names like "ImageNet-1K")
        ds_line = re.split(r'\s*[–—]\s*|\.|,|;', ds_line)[0].strip()
        
        # Reject if it starts with markdown header BEFORE removing it
        if ds_line.startswith('#'):
            ds_line = None
        else:
            # Validate: reject if it's just a number, version-like, or too short
            if 3 <= len(ds_line) <= 120:
                # Reject if it's just digits or version numbers (e.g., "3", "3.5.1")
                if not re.match(r'^[\d.]+$', ds_line):
                    # Reject if it looks like a framework version (contains multiple dots with digits)
                    if not re.match(r'^\d+\.\d+\.\d+', ds_line):
                        # Reject if it's a generic section header or common word
                        if not re.match(r'^(features|test|model|description|overview|details|my|this|introduction|usage|about)', ds_line, re.IGNORECASE):
                            dataset = ds_line
                        else:
                            ds_line = None
                    else:
                        ds_line = None
                else:
                    ds_line = None
            else:
                ds_line = None
    if not dataset:
        # spaCy fallback
        dataset = spacy_extract_dataset(lines_text)
        # Apply same validation to spaCy results
        if dataset:
            # Reject markdown headers, short strings, pure numbers, or version patterns
            if (len(dataset) < 3 or 
                dataset.startswith('#') or
                re.match(r'^[\d.]+$', dataset) or 
                re.match(r'^\d+\.\d+\.\d+', dataset) or
                re.match(r'^(features|test|model|description|overview|details|my|this|introduction)', dataset, re.IGNORECASE)):
                dataset = None
    if dataset:
        result['dataset'] = dataset

    # Training type (allow multiple tags joined by '+')
    training_signals: List[str] = []
    if re.search(r"\bfinetune(d)?\b|\bfine\-tune(d)?\b", plain, re.IGNORECASE):
        training_signals.append('finetune')
    if re.search(r"\bLoRA\b", plain, re.IGNORECASE):
        training_signals.append('LoRA')
    if re.search(r"\bmerged?\b", plain, re.IGNORECASE):
        training_signals.append('merged')
    if re.search(r"\bdistill(ed|ation)\b", plain, re.IGNORECASE):
        training_signals.append('distilled')
    if re.search(r"\bquantiz(ed|ation)\b", plain, re.IGNORECASE):
        training_signals.append('quantized')
    # RL family (uppercase tokens preserved)
    for rl in rl_methods:
        # Use word boundary to capture tokens like V-MPO/GSPO/GRPO appearing as-is
        if re.search(rf"\b{re.escape(rl)}\b", plain):
            if rl not in training_signals:
                training_signals.append(rl)
    if training_signals:
        result['trainingType'] = '+'.join(list(dict.fromkeys(training_signals)))

    # Omitted: language, license, sourceUrl, deploymentTarget, hardwareRequirements, performanceMetrics, citation

    return result


def run_extraction_optimized(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata ONLY from description, never from original data fields.
    
    Extracts these fields from description:
    - version: Model version (e.g., "1.2")
    - modalities: Inferred from mlTask (e.g., ["text", "image"])
    - domain: Domain category (e.g., "roleplay", "medical")
    - architecture: Model architecture (e.g., "LLaMA", "BERT")
    - modelSize: Parameter count (e.g., "24B", "7B")
    - dataset: Training dataset name from description
    - trainingType: Training method (e.g., "merged", "finetune", "LoRA")
    
    Original data fields (db_identifier, name, sharedBy, mlTask, etc.) 
    are never included in extracted metadata to maintain clear separation.
    """
    # Get the full extraction result from description
    full_result = run_extraction(data)
    
    # The extraction now only returns truly extracted fields
    return full_result

