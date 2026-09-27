from __future__ import annotations

END_OF_TEXT = "<|endoftext|>"
PAD = "<|pad|>"
QUESTION, QUESTION_END = "<|question|>", "<|/question|>"
PLAN, PLAN_END = "<|plan|>", "<|/plan|>"
SEARCH, SEARCH_END = "<|search|>", "<|/search|>"
PASSAGE, PASSAGE_END = "<|passage|>", "<|/passage|>"
NOTE, NOTE_END = "<|note|>", "<|/note|>"
ANSWER, ANSWER_END = "<|answer|>", "<|/answer|>"
DECLINE = "<|decline|>"

NAMED = (
    END_OF_TEXT,
    PAD,
    QUESTION,
    QUESTION_END,
    PLAN,
    PLAN_END,
    SEARCH,
    SEARCH_END,
    PASSAGE,
    PASSAGE_END,
    NOTE,
    NOTE_END,
    ANSWER,
    ANSWER_END,
    DECLINE,
)


def special_tokens(count: int) -> list[str]:
    if count < len(NAMED):
        raise ValueError(f"{count} reserved slots cannot hold the {len(NAMED)} named tokens")
    return list(NAMED) + [f"<|reserved_{index}|>" for index in range(len(NAMED), count)]
