from __future__ import annotations
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Mapping, Sequence

VISIBLE_FIELDS = ("title", "subtitle", "authors", "year", "subjects", "description")
IDENTITY_FIELDS = ("record_id", "work_key", "split", "source_group")


def metadata_json(row: Mapping[str, Any]) -> str:
    payload = {field: str(row.get(field) or "") for field in VISIBLE_FIELDS}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def prompt_delimiters(metadata: str) -> tuple[str, str]:
    """Return deterministic delimiter strings that cannot occur in this metadata."""
    counter = 0
    while True:
        digest = hashlib.sha256(f"{counter}\0{metadata}".encode("utf-8")).hexdigest()
        begin = f"BEGIN_UNTRUSTED_METADATA_{digest}"
        end = f"END_UNTRUSTED_METADATA_{digest}"
        if begin not in metadata and end not in metadata:
            return begin, end
        counter += 1


def render_prompt(
    template: str,
    row: Mapping[str, Any],
    *,
    label: str | None = None,
) -> str:
    metadata_marker = "{{METADATA_JSON}}"
    label_marker = "{{LABEL}}"
    begin_marker = "{{BEGIN_METADATA_BOUNDARY}}"
    end_marker = "{{END_METADATA_BOUNDARY}}"
    if template.count(metadata_marker) != 1:
        raise RuntimeError("Prompt template must contain one metadata marker")
    if template.count(begin_marker) != 1 or template.count(end_marker) != 1:
        raise RuntimeError("Prompt template must contain one boundary-marker pair")
    label_markers = template.count(label_marker)
    if label is None and label_markers:
        raise RuntimeError("Prompt template unexpectedly requires a label")
    if label is not None and label_markers != 1:
        raise RuntimeError("Evidence prompt must contain one label marker")
    template_without_known_markers = template
    for marker in (metadata_marker, label_marker, begin_marker, end_marker):
        template_without_known_markers = template_without_known_markers.replace(
            marker, ""
        )
    if "{{" in template_without_known_markers or "}}" in template_without_known_markers:
        raise RuntimeError("Unresolved prompt-template marker")
    rendered_template = (
        template.replace(label_marker, str(label)) if label is not None else template
    )
    metadata = metadata_json(row)
    begin, end = prompt_delimiters(metadata)
    rendered_template = rendered_template.replace(begin_marker, begin)
    rendered_template = rendered_template.replace(end_marker, end)
    # Metadata is substituted last and is never reinterpreted as template syntax or
    # able to close its record-specific delimiter.
    return rendered_template.replace(metadata_marker, metadata)


def chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def code_token_ids(tokenizer: Any, codes: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    reverse: Dict[int, str] = {}
    for code in codes:
        token_ids = tokenizer.encode(code, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f"Code {code!r} is not one token: {token_ids}")
        token_id = int(token_ids[0])
        if tokenizer.decode([token_id], skip_special_tokens=False) != code:
            raise RuntimeError(f"Code token does not decode exactly: {code!r}")
        if token_id in reverse:
            raise RuntimeError(f"Codes share a token: {code!r}, {reverse[token_id]!r}")
        result[code] = token_id
        reverse[token_id] = code
    return result


def stable_softmax(
    values: Mapping[str, float], codes: Sequence[str]
) -> Dict[str, float]:
    maximum = max(float(values[code]) for code in codes)
    weights = {code: math.exp(float(values[code]) - maximum) for code in codes}
    denominator = sum(weights.values())
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Invalid constrained-choice probability mass")
    return {code: weights[code] / denominator for code in codes}


def preflight_prompt_tokens(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    output_tokens: int,
    maximum_model_tokens: int,
    task: str,
) -> List[int]:
    counts: List[int] = []
    for index, prompt in enumerate(prompts):
        count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if count + output_tokens > maximum_model_tokens:
            raise RuntimeError(
                f"{task} prompt {index} requires {count}+{output_tokens} tokens; "
                f"model cap is {maximum_model_tokens}"
            )
        counts.append(count)
    return counts


def logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def parse_choice_output(
    request_output: Any,
    codes: Sequence[str],
    token_ids: Mapping[str, int],
) -> Dict[str, Any]:
    if len(request_output.outputs) != 1:
        raise RuntimeError("Expected exactly one completion")
    completion = request_output.outputs[0]
    generated_ids = list(completion.token_ids)
    if len(generated_ids) != 1:
        raise RuntimeError(f"Expected one generated token; found {generated_ids}")
    by_token = {token: code for code, token in token_ids.items()}
    generated_token = int(generated_ids[0])
    if generated_token not in by_token:
        raise RuntimeError(
            f"Generated token is outside the frozen choices: {generated_token}"
        )
    if not completion.logprobs or len(completion.logprobs) != 1:
        raise RuntimeError("Missing one-token log probabilities")
    position = completion.logprobs[0]
    raw_logprobs: Dict[str, float] = {}
    for code in codes:
        token_id = token_ids[code]
        entry = position.get(token_id)
        if entry is None:
            entry = position.get(str(token_id))
        if entry is None:
            raise RuntimeError(f"Missing log probability for code {code!r}")
        raw_logprobs[code] = logprob_value(entry)
    scores = stable_softmax(raw_logprobs, codes)
    maximum = max(raw_logprobs.values())
    if raw_logprobs[by_token[generated_token]] < maximum - 1e-8:
        raise RuntimeError("Generated choice is not a maximum-logprob allowed token")
    return {
        "code": by_token[generated_token],
        "raw_text": str(completion.text),
        "logprobs": raw_logprobs,
        "scores": scores,
        "prompt_tokens": len(request_output.prompt_token_ids or []),
    }


def run_choices(
    llm: Any,
    tokenizer: Any,
    rendered_prompts: Sequence[str],
    codes: Sequence[str],
    seed: int,
    temperature: float,
    maximum_tokens: int,
) -> List[Dict[str, Any]]:
    from vllm import SamplingParams

    token_ids = code_token_ids(tokenizer, codes)
    if temperature != 0.0 or maximum_tokens != 1:
        raise RuntimeError("Frozen classification requires temperature 0 and one token")
    params = SamplingParams(
        temperature=temperature,
        max_tokens=maximum_tokens,
        min_tokens=maximum_tokens,
        seed=seed,
        allowed_token_ids=list(token_ids.values()),
        logprobs=len(token_ids),
    )
    outputs = llm.generate(list(rendered_prompts), params, use_tqdm=False)
    if len(outputs) != len(rendered_prompts):
        raise RuntimeError("vLLM returned a different number of choice outputs")
    return [parse_choice_output(output, codes, token_ids) for output in outputs]


def normalise_evidence(raw: str) -> tuple[str, bool]:
    value = raw.strip()
    changed = value != raw
    matching_quotes = (
        ('"', '"'),
        ("'", "'"),
        ("\u201c", "\u201d"),
        ("\u2018", "\u2019"),
    )
    for opening, closing in matching_quotes:
        if len(value) >= 2 and value.startswith(opening) and value.endswith(closing):
            value = value[len(opening) : -len(closing)].strip()
            changed = True
            break
    return value, changed


def ground_evidence(
    row: Mapping[str, Any],
    raw: str,
    no_evidence_token: str,
) -> Dict[str, Any]:
    value, normalised = normalise_evidence(raw)
    if not value or value == no_evidence_token:
        return {
            "evidence_raw": raw,
            "evidence_span": "",
            "evidence_field": "",
            "evidence_status": "missing",
            "evidence_normalised": normalised,
        }
    if len(value.split()) > 20:
        return {
            "evidence_raw": raw,
            "evidence_span": "",
            "evidence_field": "",
            "evidence_status": "invalid_too_long",
            "evidence_normalised": normalised,
        }
    fields = [field for field in VISIBLE_FIELDS if value in str(row.get(field) or "")]
    if not fields:
        return {
            "evidence_raw": raw,
            "evidence_span": "",
            "evidence_field": "",
            "evidence_status": "invalid_not_found",
            "evidence_normalised": normalised,
        }
    return {
        "evidence_raw": raw,
        "evidence_span": value,
        "evidence_field": fields[0] if len(fields) == 1 else "multiple",
        "evidence_status": "grounded",
        "evidence_normalised": normalised,
    }


def run_evidence(
    llm: Any,
    prompts: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    temperature: float,
    maximum_tokens: int,
    no_evidence_token: str,
) -> List[Dict[str, Any]]:
    from vllm import SamplingParams

    if not prompts:
        return []
    params = SamplingParams(
        temperature=temperature,
        max_tokens=maximum_tokens,
        seed=seed,
        stop=["\n"],
    )
    outputs = llm.generate(list(prompts), params, use_tqdm=False)
    if len(outputs) != len(prompts):
        raise RuntimeError("vLLM returned a different number of evidence outputs")
    results: List[Dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        if len(output.outputs) != 1:
            raise RuntimeError("Expected one evidence completion")
        grounded = ground_evidence(row, str(output.outputs[0].text), no_evidence_token)
        grounded["prompt_tokens_evidence"] = len(output.prompt_token_ids or [])
        results.append(grounded)
    return results


def derived_status(fiction_code: str, relevance_code: str) -> str:
    if fiction_code in {"N", "O"}:
        return "NEGATIVE"
    if fiction_code == "U" or relevance_code == "3":
        return "UNRESOLVED"
    if fiction_code == "F" and relevance_code in {"1", "2"}:
        return "POSITIVE"
    if fiction_code == "F" and relevance_code == "0":
        return "NEGATIVE"
    raise RuntimeError(
        f"Unsupported model-label combination: {fiction_code!r}, {relevance_code!r}"
    )


def result_rows(
    source_rows: Sequence[Mapping[str, Any]],
    fiction_results: Sequence[Mapping[str, Any]],
    relevance_results: Sequence[Mapping[str, Any]],
    evidence_by_index: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    start_index: int,
) -> List[Dict[str, Any]]:
    fiction_map = config["classification"]["fiction_codes"]
    relevance_map = config["classification"]["relevance_codes"]
    positive_codes = set(config["classification"]["positive_relevance_codes"])
    output: List[Dict[str, Any]] = []
    for local_index, (source, fiction, relevance) in enumerate(
        zip(source_rows, fiction_results, relevance_results)
    ):
        fiction_code = str(fiction["code"])
        relevance_code = str(relevance["code"])
        model_positive = fiction_code == "F" and relevance_code in positive_codes
        retained_relevance = (
            relevance_map[relevance_code] if fiction_code in {"F", "U"} else ""
        )
        score = float(fiction["scores"]["F"]) * (
            float(relevance["scores"]["1"]) + float(relevance["scores"]["2"])
        )
        evidence = evidence_by_index.get(
            local_index,
            {
                "evidence_raw": "",
                "evidence_span": "",
                "evidence_field": "",
                "evidence_status": "not_requested",
                "evidence_normalised": False,
                "prompt_tokens_evidence": 0,
            },
        )
        row: Dict[str, Any] = {
            "input_index": start_index + local_index,
            **{field: source.get(field, "") for field in IDENTITY_FIELDS},
            "fiction_code": fiction_code,
            "fiction_status": fiction_map[fiction_code],
            "fiction_raw_text": fiction["raw_text"],
            "relevance_code_raw": relevance_code,
            "relevance_label_raw": relevance_map[relevance_code],
            "environment_climate_relevance": retained_relevance,
            "relevance_raw_text": relevance["raw_text"],
            "derived_status": derived_status(fiction_code, relevance_code),
            "model_positive": model_positive,
            "target_rank_score": score,
            "prompt_tokens_fiction": int(fiction["prompt_tokens"]),
            "prompt_tokens_relevance": int(relevance["prompt_tokens"]),
        }
        for code in fiction_map:
            row[f"fiction_logprob_{code}"] = float(fiction["logprobs"][code])
            row[f"fiction_score_{code}"] = float(fiction["scores"][code])
        for code in relevance_map:
            row[f"relevance_logprob_{code}"] = float(relevance["logprobs"][code])
            row[f"relevance_score_{code}"] = float(relevance["scores"][code])
        row.update(evidence)
        output.append(row)
    return output
