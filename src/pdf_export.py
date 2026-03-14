"""
Export chat history to PDF.
"""

from fpdf import FPDF
from datetime import datetime
import os


def export_chat_to_pdf(session_title: str, messages: list, output_path: str = None) -> str:
    """Export a chat session to a PDF file."""
    if output_path is None:
        os.makedirs("data/exports", exist_ok=True)
        safe_title = "".join(c for c in session_title if c.isalnum() or c in " _-")[:50]
        output_path = f"data/exports/{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Jarvis Mk.X - {session_title}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Messages
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 100, 200)
            pdf.cell(0, 8, "You:", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(40, 160, 80)
            pdf.cell(0, 8, "Jarvis:", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)

        # Handle encoding
        safe_content = content.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_content)
        pdf.ln(3)

    pdf.output(output_path)
    return output_path
