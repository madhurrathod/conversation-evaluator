"""Generate a PDF of the full Claude Code conversation session."""

import json
import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos


JSONL = Path("/Users/madhurrathod/.claude/projects/-Users-madhurrathod-Ahoum/7dd51af3-b5d4-46ba-9b12-b272ae87d62e.jsonl")
OUT   = Path(__file__).parent / "conversation_log.pdf"


# ── Parse ────────────────────────────────────────────────────────────────────

def extract_messages(path: Path) -> list[dict]:
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg  = obj.get("message", {})
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(parts)

            # Strip XML/system noise
            content = re.sub(r"<system-reminder>.*?</system-reminder>", "", content, flags=re.DOTALL)
            content = re.sub(r"<local-command-caveat>.*?</local-command-caveat>", "", content, flags=re.DOTALL)
            content = re.sub(r"<[^>]+>", "", content)
            content = content.strip()

            if not content:
                continue

            messages.append({"role": role, "content": content})
    return messages


# ── PDF ──────────────────────────────────────────────────────────────────────

class ConversationPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "Ahoum AI & ML Assignment - Claude Code Session Log", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def safe_text(text: str) -> str:
    """Replace smart quotes and unicode chars with ASCII equivalents."""
    replacements = {
        "—": "--", "–": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "…": "...", "•": "*",
        "→": "->", "←": "<-", "×": "x", "÷": "/",
        "≠": "!=", "≤": "<=", "≥": ">=", "°": "deg",
        "α": "alpha", "β": "beta", "✓": "[ok]",
        "✔": "[ok]", "✕": "[x]", "❌": "[x]",
        "█": "#", "►": ">",
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(messages: list[dict]) -> None:
    pdf = ConversationPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title block
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Conversation Log", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(messages)} messages",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8)

    USER_BG      = (230, 244, 255)   # light blue
    ASSISTANT_BG = (242, 242, 242)   # light grey
    USER_HDR     = (0, 102, 204)
    ASST_HDR     = (80, 80, 80)

    for i, msg in enumerate(messages, 1):
        role    = msg["role"]
        content = safe_text(msg["content"])

        is_user = role == "user"
        bg      = USER_BG if is_user else ASSISTANT_BG
        hdr_col = USER_HDR if is_user else ASST_HDR
        label   = "You" if is_user else "Claude"

        # Header label
        pdf.set_fill_color(*bg)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*hdr_col)
        pdf.cell(0, 6, f"  {label}  (#{i})", fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Body
        pdf.set_font("Courier", "", 8)
        pdf.set_text_color(30, 30, 30)
        pdf.set_fill_color(*bg)

        lines = content.split("\n")
        for line in lines:
            # Wrap long lines manually
            max_w = pdf.w - pdf.l_margin - pdf.r_margin - 4
            while pdf.get_string_width(line) > max_w and len(line) > 1:
                # Find split point
                split = len(line)
                while pdf.get_string_width(line[:split]) > max_w and split > 1:
                    split -= 1
                pdf.cell(0, 4, "  " + line[:split], fill=True,
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                line = line[split:]
            pdf.cell(0, 4, "  " + line, fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(3)

    pdf.output(str(OUT))
    print(f"PDF saved → {OUT}  ({OUT.stat().st_size // 1024} KB,  {pdf.page} pages)")


if __name__ == "__main__":
    msgs = extract_messages(JSONL)
    print(f"Messages extracted: {len(msgs)}")
    build_pdf(msgs)
