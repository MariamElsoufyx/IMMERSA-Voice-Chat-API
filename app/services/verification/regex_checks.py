
import re
import time

import app.core.config as config
from app.characters import characters_info
from app.services.verification.base import CheckResult
from app.utils.log import log



_PROFANITY_WORDS = [
    # English shteema
    "fuck", "fucking", "fucked", "fucker",
    "shit", "shitting", "shitty",
    "ass", "asshole", "asses",
    "bitch", "bitches",
    "bastard", "bastards",
    "damn", "damned",
    "crap",
    "dick", "dicks",
    "cock", "cocks",
    "pussy", "pussies",
    "whore", "whores",
    "slut", "sluts",
    "nigger", "niggers",
    "faggot", "faggots",
    "cunt", "cunts",
    "piss", "pissed",
    "hell",
    # Arabic shteema
    "kos", "kuss", "kes", "cos", "kous",
    "metnak", "metnaka", "metnakeen", "neek",
    "teez", "tizz", "tiz",
    "khara",
    "maniak", "manyak", "manayek", "manyaka",
    "sharmout", "sharmouta", "sharmoot", "sharmoota",
    "kalb", "kalba", "homar", "homara",
    "zeb", "zib", "zobr", "azbar",
    "ahbal",
]

_PROFANITY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _PROFANITY_WORDS) + r")\b",
    re.IGNORECASE,
)




_MODERN_TERMS = [
    # Computing / internet
    "internet", "wifi", "wi-fi", "website", "web site", "online", "offline",
    "download", "downloads", "upload", "uploads", "email", "e-mail", "url",
    "computer", "laptop", "desktop", "tablet", "ipad", "smartphone", "iphone",
    "android", "app", "apps", "software", "hardware", "operating system",
    "windows", "macos", "linux", "browser", "chrome", "firefox", "safari",
    "edge", "javascript", "python", "html", "css", "api", "server", "cloud",
    "database", "blockchain", "crypto", "cryptocurrency", "bitcoin", "ethereum",
    "nft", "nfts",
    # AI
    "ai", "a.i.", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "neural networks", "chatgpt", "openai", "anthropic",
    "claude", "gemini", "llm", "llms", "gpt",
    # Mobile / comms
    "smartphone", "smartphones", "mobile phone", "mobile phones", "cell phone",
    "cellphone", "cell phones", "text message", "text messages", "sms",
    "whatsapp", "telegram", "signal", "zoom", "skype", "facetime", "video call",
    "video chat", "livestream", "live stream", "streaming",
    # Social / brands
    "facebook", "instagram", "twitter", "tiktok", "snapchat", "youtube",
    "google", "microsoft", "apple inc", "amazon", "netflix", "spotify",
    "uber", "airbnb", "tesla", "spacex", "discord", "twitch", "reddit",
    "linkedin", "github",
    #  tech 
    "gps", "bluetooth", "usb", "ethernet", "wireless", "satellite",
    "drone", "drones", "robot", "robots", "robotics", "vr",
    "virtual reality", "ar", "augmented reality", "selfie", "selfies",
    "hashtag", "emoji", "emojis", "podcast", "podcasts",
    "electric car", "hybrid car", "hybrid vehicle", "ev", "evs",
    # hagat random w masayeb
    "covid", "covid-19", "coronavirus", "pandemic 2020", "world war ii",
    "world war 2", "world war two", "wwii", "ww2", "cold war", "nuclear bomb",
    "atomic bomb", "moon landing",
]



_MODERN_TERMS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _MODERN_TERMS) + r")\b",
    re.IGNORECASE,
)

# 4-digit year between 1000 and 2999.
_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")

# URL: http(s)://... or bare www.something.tld
_URL_PATTERN = re.compile(r"\b(?:https?://\S+|www\.\S+\.\S+)", re.IGNORECASE)

# Simple email pattern — covers the common case
_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")






def check_profanity(text: str, *, name: str) -> CheckResult:
    
    label = name.split(".")[-1] 
    t0 = time.perf_counter()
    flagged = sorted({m.group().lower() for m in _PROFANITY_PATTERN.finditer(text or "")})
    latency = time.perf_counter() - t0
    if flagged:
        log.fail("REGEX", f"{label} — flagged {flagged} ({latency*1000:.1f}ms)")
    else:
        log.ok("REGEX", f"{label} clean ({latency*1000:.1f}ms)")
    return CheckResult(
        name=name,
        passed=not flagged,
        reasons=[f"profanity: {w}" for w in flagged],
        details={"flagged": flagged},
        latency_s=latency,
    )


def check_anachronism(text: str, character_id: str | None) -> CheckResult:
    #check fe mosta2abl walla La 
    t0 = time.perf_counter()
    reasons: list[str] = []

  
    cutoff = config.ANACHRONISM_DEFAULT_LATEST_YEAR
    if character_id:
        year = characters_info.operation_year.get(character_id)
        if year:
            try:
                cutoff = int(year)
            except ValueError:
                pass
    future_years = sorted({m.group() for m in _YEAR_PATTERN.finditer(text or "") if int(m.group()) > cutoff})
    for y in future_years:
        reasons.append(f"future year: {y} (cutoff {cutoff})")

    modern_hits = sorted({m.group().lower() for m in _MODERN_TERMS_PATTERN.finditer(text or "")})
    for term in modern_hits:
        reasons.append(f"modern term: {term}")

    for url in {m.group() for m in _URL_PATTERN.finditer(text or "")}:
        reasons.append(f"url: {url}")

    for email in {m.group() for m in _EMAIL_PATTERN.finditer(text or "")}:
        reasons.append(f"email: {email}")

    latency = time.perf_counter() - t0
    if reasons:
        log.fail("REGEX", f"anachronism (character={character_id}) — {len(reasons)} hit(s) ({latency*1000:.1f}ms)")
        for r in reasons:
            log.detail(r)
    else:
        log.ok("REGEX", f"anachronism clean (character={character_id}, cutoff={cutoff}) ({latency*1000:.1f}ms)")

    return CheckResult(
        name="regex.anachronism",
        passed=not reasons,
        reasons=reasons,
        details={
            "cutoff": cutoff,
            "future_years": future_years,
            "modern_terms": modern_hits,
        },
        latency_s=latency,
    )
