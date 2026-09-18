"""Styling for the chat UI.

Streamlit's native st.chat_message / st.chat_input carry the structure; this only
tones them down into something closer to a modern chat app: a narrow centred
reading column, no heavy message cards, and a soft tint on the user's turn only.
"""

CHAT_MAX_WIDTH = "46rem"

css = f"""
<style>
/* ---- Reading column ------------------------------------------------ */
[data-testid="stMainBlockContainer"] {{
    max-width: {CHAT_MAX_WIDTH};
    padding-top: 3rem;
    padding-bottom: 2rem;
}}
/* Keep the pinned composer aligned with the conversation above it. */
[data-testid="stBottomBlockContainer"] {{
    max-width: {CHAT_MAX_WIDTH};
    padding-bottom: 1rem;
}}

/* ---- Messages ------------------------------------------------------ */
/* Strip Streamlit's default card so turns read as a transcript, not boxes. */
[data-testid="stChatMessage"] {{
    background: transparent;
    border: none;
    padding: 0.4rem 0;
    gap: 0.85rem;
}}
[data-testid="stChatMessage"] img {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    object-fit: cover;
}}
/* Only the user's turn gets a fill, the way ChatGPT/Claude do it.
   Keyed off the st-key- class emitted by the wrapper container in app.turn():
   a custom image avatar renders the same testid for both roles, so the avatar
   itself is no help here. */
[class*="st-key-userturn-"] [data-testid="stChatMessageContent"] {{
    background: rgba(128, 132, 149, 0.13);
    border-radius: 1.05rem;
    padding: 0.55rem 0.95rem;
    /* Hug the text instead of stretching the full column, which reads as a
       banner rather than a message. Streamlit sets flex:1 1 auto on this
       element, so the grow factor has to be cancelled for width to matter. */
    flex: 0 1 auto;
    width: fit-content;
    max-width: 100%;
    /* Streamlit puts margin:auto on this element, which centres a fit-content
       box and leaves it floating away from the avatar. */
    margin-left: 0;
    margin-right: auto;
    margin-bottom: 0.4rem;
}}

/* Streamlit offsets its paragraph margin with margin-bottom:-1rem on
   stMarkdownContainer. That is invisible in normal flow, but it makes the
   container under-report its height by 1rem, so the last line of text spills
   out the bottom of any box we paint a background on. Zero it and let the
   paragraph rules above own the spacing. */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
    margin-bottom: 0 !important;
}}
/* Tighten the runaway vertical rhythm inside message bodies. */
[data-testid="stChatMessage"] p {{
    line-height: 1.65;
    margin-bottom: 0.6rem;
}}
[data-testid="stChatMessage"] p:last-child {{
    margin-bottom: 0;
}}
[data-testid="stChatMessage"] pre {{
    border-radius: 0.6rem;
    font-size: 0.86rem;
}}

/* ---- Empty state --------------------------------------------------- */
/* The block is a Streamlit vertical flex column with align-items:start, so the
   logo's wrapper shrinks to the image and pins left — centring needs the flex
   axis, not just text-align. Streamlit also sets text-align:left explicitly on
   headings and captions, so that has to be overridden on descendants rather
   than inherited. */
[class*="st-key-welcome"] {{
    align-items: center !important;
    margin-top: 10vh;
}}
[class*="st-key-welcome"],
[class*="st-key-welcome"] * {{
    text-align: center !important;
}}
[class*="st-key-welcome"] img {{
    border-radius: 20px;
}}
[class*="st-key-welcome"] h4 {{
    padding-top: 0.9rem;
    padding-bottom: 0.2rem;
}}

/* ---- Composer ------------------------------------------------------ */
/* st.chat_input is already pinned to the bottom and grows with its content;
   this just rounds it off and calms the focus ring. */
[data-testid="stChatInput"] {{
    border-radius: 1.4rem;
    border: 1px solid rgba(128, 132, 149, 0.28);
    box-shadow: 0 2px 14px rgba(0, 0, 0, 0.10);
}}
/* Streamlit draws its own focus ring in the theme's primary colour on an inner
   element; neutralise it so the rounded outer border is the only focus cue. */
[data-testid="stChatInput"]:focus-within {{
    border-color: rgba(128, 132, 149, 0.55);
}}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {{
    border-color: transparent !important;
    box-shadow: none !important;
}}
/* The filled grey box is this inner div, and it carries its own 8px radius —
   so without this it sits square on top of the rounded outer border, leaving
   the corners showing underneath. */
[data-testid="stChatInput"] > div {{
    border-radius: inherit !important;
}}
[data-testid="stChatInput"] textarea {{
    font-size: 0.97rem;
    line-height: 1.55;
}}
/* Streamlit's default primary is a bright red; tone the send button down to
   match the rest. Done in CSS rather than a [theme] block in config.toml,
   because defining a theme there pins the app to light and kills the automatic
   light/dark switching. */
[data-testid="stChatInputSubmitButton"] {{
    background: rgba(128, 132, 149, 0.20) !important;
    color: inherit !important;
    border-radius: 50% !important;
}}
[data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {{
    background: rgba(128, 132, 149, 0.34) !important;
}}
[data-testid="stChatInputSubmitButton"] svg {{
    fill: currentColor !important;
}}

/* ---- Sources disclosure -------------------------------------------- */
[data-testid="stChatMessage"] [data-testid="stExpander"] details {{
    border: none;
    background: transparent;
}}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary {{
    font-size: 0.8rem;
    opacity: 0.65;
    padding-left: 0;
}}

/* ---- Waiting indicator --------------------------------------------- */
.typing {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 1.6rem;
}}
.typing .dot {{
    width: 7px;
    height: 7px;
    background: currentColor;
    opacity: 0.55;
    border-radius: 50%;
    animation: typing-bounce 1.2s infinite ease-in-out;
}}
.typing .dot:nth-child(2) {{ animation-delay: 0.15s; }}
.typing .dot:nth-child(3) {{ animation-delay: 0.30s; }}

@keyframes typing-bounce {{
    0%, 80%, 100% {{ transform: translateY(0); opacity: 0.4; }}
    40%           {{ transform: translateY(-5px); opacity: 1; }}
}}

/* Hide the "Press Enter to submit" hint that clutters the composer. */
[data-testid="InputInstructions"] {{ display: none; }}
</style>
"""

TYPING_INDICATOR = (
    '<span class="typing">'
    '<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
    "</span>"
)
