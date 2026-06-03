
from app.characters import characters_info
from app.characters import prompts


def load_prompt(prompt_type=None, prompt_key=None):
    """Retrieve a prompt from the prompts dictionary based on a given key."""
    if prompt_type == "system":
        return prompts.system_prompts.get(prompt_key)
    return prompts.user_prompts.get(prompt_key)


def format_history_chunks(chunks=None) -> str:
    """Render retrieved history chunks into the <history> block injected into the
    system prompt. Accepts a list of HistoryChunk objects (or plain strings).
    Returns a clear placeholder when nothing was retrieved so the model doesn't
    see an empty block and hallucinate."""
    if not chunks:
        return "(No specific records were retrieved for this question.)"
    parts = []
    for i, c in enumerate(chunks, 1):
        content = (getattr(c, "content", None) or str(c)).strip()
        source = getattr(c, "source_doc", None)
        header = f"[{i}]" + (f" (source: {source})" if source else "")
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


def fill_character_fields(prompt: str, character_id: str) -> str:
    """Substitute all {persona-field} placeholders in a template for one character."""
    prompt = prompt.replace("{first_name}", characters_info.first_name.get(character_id, ""))
    prompt = prompt.replace("{middle_name}", characters_info.middle_name.get(character_id, ""))
    prompt = prompt.replace("{last_name}", characters_info.last_name.get(character_id, ""))
    prompt = prompt.replace("{department}", characters_info.department.get(character_id, ""))
    prompt = prompt.replace("{location}", characters_info.location.get(character_id, ""))
    prompt = prompt.replace("{gender}", characters_info.gender.get(character_id, ""))
    prompt = prompt.replace("{operation_year}", characters_info.operation_year.get(character_id, ""))
    prompt = prompt.replace("{financial_status}", characters_info.financial_status.get(character_id, ""))
    prompt = prompt.replace("{personal_items}", ", ".join(characters_info.personal_items.get(character_id, [])))
    prompt = prompt.replace("{influences}", ", ".join(characters_info.influences.get(character_id, [])))
    prompt = prompt.replace("{significant_info}", ", ".join(characters_info.significant_info.get(character_id, [])))
    prompt = prompt.replace("{academic_rank}", characters_info.academic_rank.get(character_id, ""))
    prompt = prompt.replace("{courses}", ", ".join(characters_info.courses.get(character_id, [])))
    prompt = prompt.replace("{graduation_year}", characters_info.graduation_year.get(character_id, ""))
    prompt = prompt.replace("{tools_used}", ", ".join(characters_info.tools_used.get(character_id, [])))
    prompt = prompt.replace("{good_traits}", ", ".join(characters_info.good_traits.get(character_id, [])))
    prompt = prompt.replace("{bad_traits}", ", ".join(characters_info.bad_traits.get(character_id, [])))
    prompt = prompt.replace("{internal_conflicts}", ", ".join(characters_info.internal_conflicts.get(character_id, [])))
    prompt = prompt.replace("{hobbies}", ", ".join(characters_info.hobbies.get(character_id, [])))
    return prompt


def generate_prompt(prompt_type=None, prompt_key=None, character_id=None, question=None, answer=None, retrieved_chunks=None):
    """Replace placeholders in the prompt with actual values."""

    if prompt_key is None:
        print("Prompt key is None")
        return None

    if prompt_type is None:
        print(f"Prompt type is None for key: {prompt_key}")
        return None

    if prompt_type == "user" and character_id is None:
        print(f"Character ID is None for user prompt key: {prompt_key}")
        return None

    prompt = load_prompt(prompt_type, prompt_key=prompt_key)

    if prompt is None:
        print(f"Prompt not found for type: {prompt_type}, key: {prompt_key}")
        return None

    if prompt_type == "system":
        # Fill the RAG grounding block when the prompt declares the placeholder.
        if retrieved_chunks is not None and "{retrieved_chunks}" in prompt:
            prompt = prompt.replace("{retrieved_chunks}", retrieved_chunks)
        return prompt

    if question is None:
        print(f"Question is None for user prompt key: {prompt_key} and character ID: {character_id}")
        return None

    prompt = fill_character_fields(prompt, character_id)
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{answer}", answer or "")

    return prompt


def build_narrator_prompts(character_id, question, prompt_key, retrieved_chunks=None):
    """Build (user_prompt, system_prompt) for the narrator.

    Persona, rules, and the output format all live in the SYSTEM prompt (sent once).
    The user message is just the raw question — identical in shape to the stored
    conversation-history turns — so multi-turn follow-ups ("tell me more", "when was
    it?") read as genuine continuations instead of being re-anchored by a full
    per-turn template.
    """
    persona = prompts.persona_blocks.get(prompt_key) or prompts.persona_blocks["student"]
    persona = fill_character_fields(persona, character_id).strip()

    operation_year = characters_info.operation_year.get(character_id, "")
    system_prompt = prompts.system_prompts["historical-narrator"]
    system_prompt = system_prompt.replace("{persona}", persona)
    system_prompt = system_prompt.replace("{operation_year}", operation_year)
    system_prompt = system_prompt.replace("{retrieved_chunks}", format_history_chunks(retrieved_chunks))

    user_prompt = question
    return user_prompt, system_prompt


def build_verifier_prompts(character_id, question, answer):
    user_prompt = generate_prompt(
        prompt_type="user",
        prompt_key="user-verifier",
        character_id=character_id,
        question=question,
        answer=answer,
    )

    system_prompt = generate_prompt(
        prompt_type="system",
        prompt_key="verifier",
    )
    operation_year = characters_info.operation_year.get(character_id, "")
    system_prompt = system_prompt.replace("{operation_year}", operation_year)

    return user_prompt, system_prompt