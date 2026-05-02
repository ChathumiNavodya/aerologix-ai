import io
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def clean_text(text):
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = text.replace("**", "")
    text = text.replace("*", "•")
    return text


def generate_pdf_report(report_md):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AeroLogix AI - Airport Logistics Report", styles["Title"]))
    story.append(Spacer(1, 20))

    for line in report_md.split("\n"):
        line = clean_text(line.strip())

        if line:
            story.append(Paragraph(line, styles["BodyText"]))
            story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer