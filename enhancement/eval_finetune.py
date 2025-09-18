import argparse
import json
import os
from typing import Dict, List, Optional
import re
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from vllm.lora.request import LoRARequest

def get_language_names():
    """Mapping of language codes to full language names using FLORES-200 format"""
    return {
        # Underrepresented
        'mlt_Latn': 'Maltese',
        'afr_Latn': 'Afrikaans',
        'isl_Latn': 'Icelandic',
        'cym_Latn': 'Welsh',
        'mkd_Cyrl': 'Macedonian',
        'lvs_Latn': 'Latvian',
        'lit_Latn': 'Lithuanian',
        'slv_Latn': 'Slovenian',
        'slk_Latn': 'Slovak',
        'ekk_Latn': 'Estonian',
        'kat_Geor': 'Georgian',
        'npi_Deva': 'Nepali',

        # High-resource
        'eng_Latn': 'English',
        'fra_Latn': 'French',
        'deu_Latn': 'German',
        'spa_Latn': 'Spanish',
        'ita_Latn': 'Italian',
        'rus_Cyrl': 'Russian',
        'zho_Hans': 'Chinese',
        'jpn_Jpan': 'Japanese',
        'arb_Arab': 'Arabic',

        # Aliases
        'eng': 'English',
    }

def get_opus_lang_mapping():
    """Mapping from FLORES codes to OPUS-100 codes"""
    return {
        # Underrepresented
        'mlt_Latn': 'mt',
        'afr_Latn': 'af',
        'isl_Latn': 'is',
        'cym_Latn': 'cy',
        'mkd_Cyrl': 'mk',
        'lvs_Latn': 'lv',
        'lit_Latn': 'lt',
        'slv_Latn': 'sl',
        'slk_Latn': 'sk',
        'ekk_Latn': 'et',
        'kat_Geor': 'ka',
        'npi_Deva': 'ne',
        'est_Latn': 'et',

        # High-resource
        'eng_Latn': 'en',
        'fra_Latn': 'fr',
        'deu_Latn': 'de',
        'spa_Latn': 'es',
        'ita_Latn': 'it',
        'rus_Cyrl': 'ru',
        'zho_Hans': 'zh',
        'jpn_Jpan': 'ja',
        'arb_Arab': 'ar',
    }


def load_flores_data(source_lang: str, target_lang: str, split: str = "devtest", max_samples: Optional[int] = None):
    """Load FLORES-200 translation data using 'all' configuration"""
    # Use the 'all' configuration to get all language pairs
    dataset = load_dataset("facebook/flores", name="all", split=split)
    data = list(dataset)
    
    if max_samples is not None:
        data = data[:max_samples]
        
    return data

def load_opus_data(source_lang: str, target_lang: str, max_samples: Optional[int] = None):
    """Load OPUS-100 translation data"""
    opus_mapping = get_opus_lang_mapping()
    src_code = opus_mapping.get(source_lang, source_lang.split('_')[0])
    tgt_code = opus_mapping.get(target_lang, target_lang.split('_')[0])
    
    # Try the original direction first
    try:
        dataset = load_dataset("Helsinki-NLP/opus-100", f"{src_code}-{tgt_code}", split="test")
        data = list(dataset)
        reverse_direction = False
    except ValueError:
        # If that fails, try the reverse direction
        try:
            dataset = load_dataset("Helsinki-NLP/opus-100", f"{tgt_code}-{src_code}", split="test")
            data = list(dataset)
            reverse_direction = True
        except ValueError:
            print("skipping opus")
            return None, None
    
    # If we used reverse direction, swap the source and target in each item
    if reverse_direction:
        for item in data:
            translation = item['translation']
            item['translation'] = {src_code: translation[tgt_code], tgt_code: translation[src_code]}
        
    return data, reverse_direction

def load_sib200_data(language: str, max_samples: Optional[int] = None):
    """Load SIB-200 classification data"""
    if language=="ekk_Latn":
        language="est_Latn"
    dataset = load_dataset("Davlan/sib200", language, split="test")
    data = list(dataset)
    
    if max_samples is not None:
        data = data[:max_samples]
        
    return data

def load_mmlu_data(max_samples: Optional[int] = None):
    """Load MMLU validation data"""
    dataset = load_dataset("cais/mmlu", "all", split="validation")
    data = list(dataset)
    
    if max_samples is not None:
        data = data[:max_samples]
        
    return data

def load_commonsense_data(dataset_name: str, max_samples: Optional[int] = None):
    """Load commonsense reasoning datasets"""
    if dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
    elif dataset_name == "piqa":
        dataset = load_dataset("piqa", split="validation")
    elif dataset_name == "winogrande":
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
    elif dataset_name == "arc_easy":
        dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation")
    elif dataset_name == "arc_challenge":
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    else:
        raise ValueError(f"Unknown commonsense dataset: {dataset_name}")
    
    data = list(dataset)
    
    if max_samples is not None:
        data = data[:max_samples]
        
    return data

def load_belebele_data(language: str, max_samples: Optional[int] = None):
    """Load BELEBELE reading comprehension data"""
    if language=="ekk_Latn":
        language="est_Latn"
    dataset = load_dataset("facebook/belebele", language, split="test")
    data = list(dataset)
    
    if max_samples is not None:
        data = data[:max_samples]
        
    return data

def format_translation_prompt(item: Dict, source_lang: str, target_lang: str, lang_name_mapping: Dict, dataset_type: str = "flores") -> str:
    """Format translation prompt for FLORES/OPUS evaluation"""
    source_name = lang_name_mapping.get(source_lang, source_lang)
    target_name = lang_name_mapping.get(target_lang, target_lang)
    
    if dataset_type == "flores":
        # Use the sentence_{lang_code} format with the 'all' configuration
        source_text = item[f'sentence_{source_lang}']
    else:  # opus
        opus_mapping = get_opus_lang_mapping()
        src_code = opus_mapping.get(source_lang, source_lang.split('_')[0])
        tgt_code = opus_mapping.get(target_lang, target_lang.split('_')[0])
        source_text = item['translation'][src_code.split('_')[0]]
    
    prompt = f"Translate this sentence into {target_name}: {source_text} Translation:" # {source_name} 
    return prompt

def format_sib200_prompt(item: Dict, language: str, lang_name_mapping: Dict) -> str:
    """Format SIB-200 topic classification prompt"""
    language_name = lang_name_mapping.get(language, language) #  {language_name}
    text = item['text']
    
    topics = ["Science/Technology", "Travel", "Politics", "Sports", "Health", "Entertainment", "Geography"]
    
    prompt = f"""Classify the following text into one of these topics:
A) Science/Technology
B) Travel  
C) Politics
D) Sports
E) Health
F) Entertainment
G) Geography

Text: {text}

Answer:"""
    
    return prompt


def format_mmlu_prompt(item: Dict) -> str:
    """Format MMLU prompt"""
    question = item['question']
    choices = item['choices']
    
    options = [f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)]
    
    prompt = f"""Answer the following multiple choice question:

{question}

{chr(10).join(options)}

Answer:"""
    
    return prompt

def format_commonsense_prompt(item: Dict, dataset_name: str) -> str:
    """Format commonsense reasoning prompts"""
    if dataset_name == "hellaswag":
        ctx = item['ctx']
        endings = item['endings']
        options = [f"{chr(65+i)}) {ending}" for i, ending in enumerate(endings)]
        
        prompt = f"""Complete the following scenario by choosing the most likely continuation:

{ctx}

{chr(10).join(options)}

Answer:"""
        
    elif dataset_name == "piqa":
        goal = item['goal']
        sol1 = item['sol1']
        sol2 = item['sol2']
        
        prompt = f"""Choose the most appropriate solution for the following goal:

Goal: {goal}

A) {sol1}
B) {sol2}

Answer:"""
        
    elif dataset_name == "winogrande":
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        
        prompt = f"""Complete the sentence by choosing the correct option:

{sentence}

A) {option1}
B) {option2}

Answer:"""
        
    elif dataset_name in ["arc_easy", "arc_challenge"]:
        question = item['question']
        choices = item['choices']['text']
        
        options = [f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)]
        
        prompt = f"""Answer the following question:

{question}

{chr(10).join(options)}

Answer:"""
    
    return prompt

def format_belebele_prompt(item: Dict, language: str, lang_name_mapping: Dict) -> str:
    """Format BELEBELE reading comprehension prompt"""
    language_name = lang_name_mapping.get(language, language)
    
    passage = item['flores_passage']
    question = item['question']
    
    # Format the multiple choice options
    options = [
        f"A) {item['mc_answer1']}",
        f"B) {item['mc_answer2']}", 
        f"C) {item['mc_answer3']}",
        f"D) {item['mc_answer4']}"
    ]
    
    prompt = f"""Read the following passage and pick the right answer.

Passage: {passage}

Question: {question}

{chr(10).join(options)}

Answer:"""
    
    return prompt

def compute_bleu_score(reference: str, candidate: str) -> float:
    """Compute BLEU score for translation evaluation"""
    from sacrebleu import sentence_bleu
    score = sentence_bleu(candidate, [reference])
    return score.score

def compute_chrf_score(reference: str, candidate: str) -> float:
    """Compute chrF score for translation evaluation"""
    from sacrebleu import sentence_chrf
    score = sentence_chrf(candidate, [reference])
    return score.score

def compute_rouge_score(reference: str, candidate: str) -> Dict[str, float]:
    """Compute ROUGE scores for summarization evaluation"""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def extract_answer_choice(response: str, num_choices: int = 4) -> Optional[str]:
    """Extract A/B/C/D choice from model response"""
    response = response.strip().upper()
    
    valid_choices = [chr(65 + i) for i in range(num_choices)]  # A, B, C, D... 
    
    patterns = [
        rf'^([{"".join(valid_choices)}])\)',
        rf'^([{"".join(valid_choices)}])\.',
        rf'^([{"".join(valid_choices)}])\s',
        rf'^([{"".join(valid_choices)}])$',
        rf'\b([{"".join(valid_choices)}])\)',
        rf'\b([{"".join(valid_choices)}])\.',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    return None

def compute_accuracy(references: List[str], predictions: List[Optional[str]]) -> float:
    """Compute accuracy for classification task"""
    if not references or not predictions:
        return 0.0
    
    correct = sum(1 for ref, pred in zip(references, predictions) if pred is not None and pred == ref)
    total = len(references)
    
    return correct / total if total > 0 else 0.0

def evaluate_translation(model, source_lang: str, target_lang: str, sampling_params, dataset_type: str = "flores", max_samples: Optional[int] = None) -> Dict:
    """Evaluate model on translation task (FLORES or OPUS)"""
    original_source_lang = source_lang
    original_target_lang = target_lang
    
    if source_lang=="ekk_Latn":
        source_lang="est_Latn"
    if target_lang=="ekk_Latn":
        target_lang="est_Latn"
    print(f"Evaluating {dataset_type} translation: {source_lang} -> {target_lang}")
    
    lang_name_mapping = get_language_names()
    
    if dataset_type == "flores":
        data = load_flores_data(source_lang, target_lang, split="devtest", max_samples=max_samples)
    else:  # opus
        data, was_reversed = load_opus_data(source_lang, target_lang, max_samples=max_samples)
        if data is None:
            return None
    
    # Format prompts
    prompts = [format_translation_prompt(item, source_lang, target_lang, lang_name_mapping, dataset_type) for item in data]
    
    # Generate responses
    print(f"Generating {len(prompts)} translations...")
    if use_lora:
        outputs = model.generate(prompts, sampling_params, lora_request=LoRARequest("adapter", 1, args.lora_adapter_path))
    else:
        outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    bleu_scores = []
    chrf_scores = []
    results = []
    
    for i, (item, response) in enumerate(zip(data, responses)):
        if dataset_type == "flores":
            reference = item[f'sentence_{target_lang}']
        else:  # opus
            opus_mapping = get_opus_lang_mapping()
            
            # Get the correct reference based on whether direction was reversed
            if was_reversed:
                # If direction was reversed in loading, we need to get the source language reference
                # because the data was swapped but we want the target language output
                src_code = opus_mapping.get(original_source_lang, original_source_lang.split('_')[0])
                reference = item['translation'][src_code]
            else:
                # Normal direction, get target language reference
                tgt_code = opus_mapping.get(original_target_lang, original_target_lang.split('_')[0])
                reference = item['translation'][tgt_code]
        
        candidate = response
        
        # Compute scores
        bleu_score = compute_bleu_score(reference, candidate)
        chrf_score = compute_chrf_score(reference, candidate)
        bleu_scores.append(bleu_score)
        chrf_scores.append(chrf_score)
        
        results.append({
            "id": i,
            "reference_translation": reference,
            "model_translation": candidate,
            "bleu_score": bleu_score,
            "chrf_score": chrf_score
        })
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
    
    return {
        "task": "translation",
        "dataset": dataset_type,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "bleu_score": avg_bleu,
        "chrf_score": avg_chrf,
        "total_samples": len(data),
        "results": results
    }

def evaluate_sib200(model, language: str, sampling_params, max_samples: Optional[int] = None) -> Dict:
    """Evaluate model on SIB-200 topic classification"""
    print(f"Evaluating SIB-200 classification: {language}")
    
    lang_name_mapping = get_language_names()
    
    data = load_sib200_data(language, max_samples=max_samples)
    
    # Format prompts
    prompts = [format_sib200_prompt(item, language, lang_name_mapping) for item in data]
    
    # Generate responses
    print(f"Generating {len(prompts)} classifications...")
    if use_lora:
        outputs = model.generate(prompts, sampling_params, lora_request=LoRARequest("adapter", 1, args.lora_adapter_path))
    else:
        outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Extract predictions and compute accuracy
    predictions = [extract_answer_choice(response, 7) for response in responses]  # 7 categories
    
    # Map label indices to letters
    category_to_choice = {
        "science/technology": "A",
        "travel": "B", 
        "politics": "C",
        "sports": "D",
        "health": "E",
        "entertainment": "F",
        "geography": "G"
    }
    references = [category_to_choice.get(item['category']) for item in data]
    
    accuracy = compute_accuracy(references, predictions)
    
    results = []
    for i, (item, response, prediction, reference) in enumerate(zip(data, responses, predictions, references)):
        correct = prediction is not None and prediction == reference
        
        results.append({
            "id": i,
            "text": item['text'],
            "correct_answer": reference,
            "model_response": response,
            "extracted_answer": prediction,
            "correct": correct
        })
    
    return {
        "task": "classification",
        "dataset": "sib200",
        "language": language,
        "accuracy": accuracy,
        "total_samples": len(data),
        "correct_count": sum(1 for r in results if r['correct']),
        "results": results
    }

def evaluate_mmlu(model, sampling_params, max_samples: Optional[int] = None) -> Dict:
    """Evaluate model on MMLU"""
    print("Evaluating MMLU")
    
    data = load_mmlu_data(max_samples=max_samples)
    
    # Format prompts
    prompts = [format_mmlu_prompt(item) for item in data]
    
    # Generate responses
    print(f"Generating {len(prompts)} MMLU answers...")
    if use_lora:
        outputs = model.generate(prompts, sampling_params, lora_request=LoRARequest("adapter", 1, args.lora_adapter_path))
    else:
        outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Extract predictions and compute accuracy
    predictions = [extract_answer_choice(response, 4) for response in responses]
    references = [chr(65 + item['answer']) for item in data]  # Convert 0,1,2,3 to A,B,C,D
    
    accuracy = compute_accuracy(references, predictions)
    
    # Compute per-subject accuracy
    subjects = {}
    for item, pred, ref in zip(data, predictions, references):
        subject = item['subject']
        if subject not in subjects:
            subjects[subject] = {'correct': 0, 'total': 0}
        subjects[subject]['total'] += 1
        if pred == ref:
            subjects[subject]['correct'] += 1
    
    subject_accuracies = {
        subject: stats['correct'] / stats['total'] 
        for subject, stats in subjects.items()
    }
    
    results = []
    for i, (item, response, prediction, reference) in enumerate(zip(data, responses, predictions, references)):
        correct = prediction is not None and prediction == reference
        
        results.append({
            "id": i,
            "subject": item['subject'],
            "question": item['question'],
            "choices": item['choices'],
            "correct_answer": reference,
            "model_response": response,
            "extracted_answer": prediction,
            "correct": correct
        })
    
    return {
        "task": "knowledge",
        "dataset": "mmlu",
        "accuracy": accuracy,
        "subject_accuracies": subject_accuracies,
        "total_samples": len(data),
        "correct_count": sum(1 for r in results if r['correct']),
        "results": results
    }

def evaluate_commonsense(model, dataset_name: str, sampling_params, max_samples: Optional[int] = None) -> Dict:
    """Evaluate model on commonsense reasoning"""
    print(f"Evaluating {dataset_name}")
    
    data = load_commonsense_data(dataset_name, max_samples=max_samples)
    
    # Format prompts
    prompts = [format_commonsense_prompt(item, dataset_name) for item in data]
    
    # Generate responses
    print(f"Generating {len(prompts)} {dataset_name} answers...")
    if use_lora:
        outputs = model.generate(prompts, sampling_params, lora_request=LoRARequest("adapter", 1, args.lora_adapter_path))
    else:
        outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Extract predictions and compute accuracy
    if dataset_name == "hellaswag":
        num_choices = 4
        predictions = [extract_answer_choice(response, num_choices) for response in responses]
        # Fix: Convert string labels to integers first, then to letters
        references = []
        for item in data:
            label = item['label']
            if isinstance(label, str):
                try:
                    label = int(label)
                except ValueError:
                    label = 0  # Default fallback
            references.append(chr(65 + label))
            
    elif dataset_name in ["piqa", "winogrande"]:
        num_choices = 2
        predictions = [extract_answer_choice(response, num_choices) for response in responses]
        if dataset_name == "piqa":
            # Fix: Convert string labels to integers first, then to letters
            references = []
            for item in data:
                label = item['label']
                if isinstance(label, str):
                    try:
                        label = int(label)
                    except ValueError:
                        label = 0  # Default fallback
                references.append(chr(65 + label))
        else:  # winogrande
            # Fix: Convert string answers to integers first, then to letters
            references = []
            for item in data:
                answer = item['answer']
                if isinstance(answer, str):
                    try:
                        answer = int(answer)
                    except ValueError:
                        answer = 1  # Default fallback
                references.append(chr(65 + answer - 1))  # 1,2 -> A,B
                
    elif dataset_name in ["arc_easy", "arc_challenge"]:
        num_choices = len(data[0]['choices']['text'])
        predictions = [extract_answer_choice(response, num_choices) for response in responses]
        # Find answer index
        references = []
        for item in data:
            answer_key = item['answerKey']
            if answer_key.isdigit():
                references.append(chr(65 + int(answer_key) - 1))  # 1,2,3,4 -> A,B,C,D
            else:
                references.append(answer_key.upper())
    
    accuracy = compute_accuracy(references, predictions)
    
    results = []
    for i, (item, response, prediction, reference) in enumerate(zip(data, responses, predictions, references)):
        correct = prediction is not None and prediction == reference
        
        results.append({
            "id": i,
            "item": item,
            "correct_answer": reference,
            "model_response": response,
            "extracted_answer": prediction,
            "correct": correct
        })
    
    return {
        "task": "commonsense",
        "dataset": dataset_name,
        "accuracy": accuracy,
        "total_samples": len(data),
        "correct_count": sum(1 for r in results if r['correct']),
        "results": results
    }

def evaluate_belebele(model, language: str, sampling_params, max_samples: Optional[int] = None) -> Dict:
    """Evaluate model on BELEBELE reading comprehension task"""
    print(f"Evaluating reading comprehension: {language}")
    
    lang_name_mapping = get_language_names()
    data = load_belebele_data(language, max_samples=max_samples)
    
    # Format prompts
    prompts = [format_belebele_prompt(item, language, lang_name_mapping) for item in data]
    
    # Generate responses
    print(f"Generating {len(prompts)} reading comprehension answers...")
    if use_lora:
        outputs = model.generate(prompts, sampling_params, lora_request=LoRARequest("adapter", 1, args.lora_adapter_path))
    else:
        outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Extract predictions and compute accuracy
    predictions = [extract_answer_choice(response) for response in responses]
    
    # Convert numeric references to letters
    num_to_choice = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    references = []
    for item in data:
        ref = item['correct_answer_num']
        if isinstance(ref, str) and ref.isdigit():
            references.append(num_to_choice.get(int(ref)))
        elif isinstance(ref, int):
            references.append(num_to_choice.get(ref))
        else:
            references.append(str(ref).upper() if ref else None)
    
    accuracy = compute_accuracy(references, predictions)
    
    results = []
    for i, (item, response, prediction, reference) in enumerate(zip(data, responses, predictions, references)):
        correct = prediction is not None and prediction == reference
        
        results.append({
            "id": i,
            "passage": item['flores_passage'],
            "question": item['question'],
            "options": [item['mc_answer1'], item['mc_answer2'], item['mc_answer3'], item['mc_answer4']],
            "correct_answer": reference,
            "model_response": response,
            "extracted_answer": prediction,
            "correct": correct
        })
    
    return {
        "task": "reading_comprehension",
        "dataset": "belebele",
        "language": language,
        "accuracy": accuracy,
        "total_samples": len(data),
        "correct_count": sum(1 for r in results if r['correct']),
        "results": results
    }

def run_full_evaluation(model, target_lang: str, sampling_params, max_samples: Optional[int] = None) -> Dict:
    """Run full evaluation suite for a given target language"""
    results = {}
    
    # General capability evaluations (English only)
    print("\n" + "="*40)
    print("GENERAL CAPABILITIES")
    print("="*40)
    
    results["mmlu"] = evaluate_mmlu(model, sampling_params, max_samples)
    
    # Commonsense reasoning tasks
    commonsense_datasets = ["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge"]
    for dataset in commonsense_datasets:
        try:
            results[f"commonsense_{dataset}"] = evaluate_commonsense(
                model, dataset, sampling_params, max_samples
            )
        except Exception as e:
            print(f"Failed to evaluate {dataset}: {e}")
    
    print("\n" + "="*40)
    print(f"MULTILINGUAL CAPABILITIES ({target_lang})")
    print("="*40)
    
    # Translation tasks - both directions for FLORES
    results[f"flores_eng_Latn_to_{target_lang}"] = evaluate_translation(
        model, "eng_Latn", target_lang, sampling_params, "flores", max_samples
    )
    results[f"flores_{target_lang}_to_eng_Latn"] = evaluate_translation(
        model, target_lang, "eng_Latn", sampling_params, "flores", max_samples
    )
    
    # Translation tasks - both directions for OPUS (if available)
    try:
        results[f"opus_eng_Latn_to_{target_lang}"] = evaluate_translation(
            model, "eng_Latn", target_lang, sampling_params, "opus", max_samples
        )
        results[f"opus_{target_lang}_to_eng_Latn"] = evaluate_translation(
            model, target_lang, "eng_Latn", sampling_params, "opus", max_samples
        )
    except Exception as e:
        print(f"OPUS not available for {target_lang}: {e}")
    
    # Reading comprehension
    try:
        results[f"belebele_{target_lang}"] = evaluate_belebele(
            model, target_lang, sampling_params, max_samples
        )
    except Exception as e:
        print(f"Belebele not available for {target_lang}: {e}")
    
    # Topic classification
    results[f"sib200_{target_lang}"] = evaluate_sib200(
        model, target_lang, sampling_params, max_samples
    )
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned vs base models on multilingual tasks")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--finetuned_model", type=str, required=True, 
                       help="Path to fine-tuned model (.pt weights file or model directory)")
    parser.add_argument("--use_lora_adapter", action="store_true",
                       help="Load LoRA adapter instead of full fine-tuned model")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                       help="Path to LoRA adapter folder (required if --use_lora_adapter is set)")
    parser.add_argument("--target_lang", type=str, required=True,
                       help="Target language code (e.g., 'mlt_Latn', 'spa_Latn', 'fra_Latn')")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples per task (None for all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                       help="Tensor parallel size (default: auto-detect GPU count)")
    
    global args
    args = parser.parse_args()

    global use_lora
    use_lora = False
    
    # Validate LoRA adapter arguments
    if args.use_lora_adapter and args.lora_adapter_path is None:
        raise ValueError("--lora_adapter_path must be specified when --use_lora_adapter is used")
    
    if args.use_lora_adapter and not os.path.exists(args.lora_adapter_path):
        raise ValueError(f"LoRA adapter path does not exist: {args.lora_adapter_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect tensor parallel size if not specified
    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    print(f"Using tensor parallel size: {tensor_parallel_size}")

    # Set up sampling parameters
    base_model = LLM(
        model=args.base_model,
        tensor_parallel_size=1,
        enforce_eager=True
    )

    eos_token_id = base_model.get_tokenizer().eos_token_id
    sampling_params = SamplingParams(
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        top_p=1.0,        # Disable nucleus sampling
        top_k=-1,         # Disable top-k sampling
        stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
        stop=[
            "\n", "\n\n",           # Basic newlines
            "Question:", "Q:",       # Prevent continuing to next question
            "Answer:", "A:",         # Prevent repeating answer prefix
            "Translation:", "The translation", "Translate:",  # Prevent repeating translation prompt
            "Summary:", "The summary", "Summarize:",      # Prevent repeating summary prompt
            "Passage:", "The passage", "Text:",     # Prevent including next passage
            "Options:", "The options", "The choices", "Choices:",  # Prevent listing options again
            "Context:", "The context", "The problem", "Problem:",  # General task delimiters
            "\n\nQuestion",         # Multi-line question start
            "\n\nAnswer",          # Multi-line answer start
            "</s>", "<|endoftext|>", # Common end tokens
            "Human:", "Assistant:", # Chat format delimiters
        ],
        skip_special_tokens=True
    )

    # Load and evaluate base model
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL")
    print("="*60)
    
    base_results = run_full_evaluation(
        base_model, args.target_lang, sampling_params, args.max_samples
    )
    
    # Clean up base model for memory efficiency
    del base_model
    torch.cuda.empty_cache()
    
    # Load and evaluate fine-tuned model or LoRA adapter
    print("\n" + "="*60)
    if args.use_lora_adapter:
        print("EVALUATING BASE MODEL WITH LORA ADAPTER")
        print(f"Base model: {args.base_model}")
        print(f"LoRA adapter: {args.lora_adapter_path}")
    else:
        print("EVALUATING FINE-TUNED MODEL")
        print(f"Fine-tuned model: {args.finetuned_model}")
    print("="*60)
    
    if args.use_lora_adapter:
        # Load base model with LoRA adapter
        use_lora = True
        finetuned_model = LLM(
            model=args.base_model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            enable_lora=True,
            max_lora_rank=32
        )
    else:
        # Load full fine-tuned model
        finetuned_model = LLM(
            model=args.finetuned_model,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True
        )
    
    finetuned_results = run_full_evaluation(
        finetuned_model, args.target_lang, sampling_params, args.max_samples
    )
    
    # Clean up fine-tuned model
    del finetuned_model
    torch.cuda.empty_cache()
    
    # Combine and save results
    evaluation_results = {
        "target_language": args.target_lang,
        "evaluation_config": {
            "max_samples": args.max_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "repetition_penalty": args.repetition_penalty,
            "use_lora_adapter": args.use_lora_adapter,
            "lora_adapter_path": args.lora_adapter_path if args.use_lora_adapter else None
        },
        "base_model_results": base_results,
        "finetuned_model_results": finetuned_results
    }
    
    # Save detailed results
    model_suffix = "lora" if args.use_lora_adapter else "finetuned"
    results_file = os.path.join(args.output_dir, f"evaluation_results_{args.target_lang}_{model_suffix}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # General capabilities summary
    print("\nGENERAL CAPABILITIES:")
    for task_name, base_result in base_results.items():
        if task_name in ["mmlu"] or task_name.startswith("commonsense_"):
            finetuned_result = finetuned_results.get(task_name, {})
            base_acc = base_result.get("accuracy", 0)
            ft_acc = finetuned_result.get("accuracy", 0)
            
            print(f"  {task_name.upper()}:")
            model_type = "LoRA" if args.use_lora_adapter else "Fine-tuned"
            print(f"    Base: {base_acc:.3f}, {model_type}: {ft_acc:.3f}, Δ: {ft_acc-base_acc:+.3f}")
        
    print(f"\nMULTILINGUAL CAPABILITIES ({args.target_lang}):")
    for task_name, base_result in base_results.items():
        if task_name.startswith(("flores_", "opus_", "belebele_", "sib200_", "eurlex_")):
            finetuned_result = finetuned_results.get(task_name, {})
            
            if base_result.get("task") == "translation":
                base_bleu = base_result.get("bleu_score", 0)
                base_chrf = base_result.get("chrf_score", 0)
                ft_bleu = finetuned_result.get("bleu_score", 0)
                ft_chrf = finetuned_result.get("chrf_score", 0)
                
                print(f"  {task_name.upper()}:")
                print(f"    Base - BLEU: {base_bleu:.2f}, chrF: {base_chrf:.2f}")
                model_type = "LoRA" if args.use_lora_adapter else "Fine-tuned"
                print(f"    {model_type} - BLEU: {ft_bleu:.2f}, chrF: {ft_chrf:.2f}")
                print(f"    Improvement - BLEU: {ft_bleu-base_bleu:+.2f}, chrF: {ft_chrf-base_chrf:+.2f}")
                
            elif base_result.get("task") in ["reading_comprehension", "classification"]:
                base_acc = base_result.get("accuracy", 0)
                ft_acc = finetuned_result.get("accuracy", 0)
                
                print(f"  {task_name.upper()}:")
                model_type = "LoRA" if args.use_lora_adapter else "Fine-tuned"
                print(f"    Base: {base_acc:.3f}, {model_type}: {ft_acc:.3f}, Δ: {ft_acc-base_acc:+.3f}")
                
            elif base_result.get("task") == "summarization":
                base_rouge = base_result.get("rouge_scores", {})
                ft_rouge = finetuned_result.get("rouge_scores", {})
                
                print(f"  {task_name.upper()}:")
                print(f"    Base - R1: {base_rouge.get('rouge1', 0):.3f}, R2: {base_rouge.get('rouge2', 0):.3f}, RL: {base_rouge.get('rougeL', 0):.3f}")
                model_type = "LoRA" if args.use_lora_adapter else "Fine-tuned"
                print(f"    {model_type} - R1: {ft_rouge.get('rouge1', 0):.3f}, R2: {ft_rouge.get('rouge2', 0):.3f}, RL: {ft_rouge.get('rougeL', 0):.3f}")
                print(f"    Improvement - R1: {ft_rouge.get('rouge1', 0)-base_rouge.get('rouge1', 0):+.3f}, R2: {ft_rouge.get('rouge2', 0)-base_rouge.get('rouge2', 0):+.3f}, RL: {ft_rouge.get('rougeL', 0)-base_rouge.get('rougeL', 0):+.3f}")

    print(f"\nDetailed results saved to: {results_file}")
    
if __name__ == "__main__":
    main()