## agent/fallback.py — map honest exit reasons to graceful responses.
## Reads the exit_reason the Ch3 loop already records. Nothing here is new
## detection; it's deciding what the USER experiences for each known stop.
FALLBACK_RESPONSES = {
    "deadlock":          "I wasn't able to make progress on this. Here's what I "
                         "found before getting stuck: {partial}",
    "no_progress":       "I couldn't fully resolve this. Here's my best partial "
                         "answer: {partial}",
    "recursion_limit":   "This turned out to be more involved than I can complete "
                         "in one go. Here's where I got: {partial}",
    "retries_exhausted": "A service I depend on is having trouble right now. "
                         "Please try again shortly.",
}

def graceful_response(result: dict) -> dict:
    reason = result.get("exit_reason")
    if reason == "goal_achieved":
        return result                                  # normal path, untouched
    template = FALLBACK_RESPONSES.get(reason,
        "I ran into a problem and couldn't complete this request.")
    partial = result.get("final_answer") or "nothing conclusive"
    return {**result,
            "final_answer": template.format(partial=partial),
            "degraded": True}                          # flag for monitoring
