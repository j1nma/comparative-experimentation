from markdown import markdown
import pdfkit

input_filename = 'report.md'
output_filename = 'report.pdf'

with open(input_filename, 'r', encoding="utf-8") as f:
    fr = f.read()
    html_text = markdown(fr, output_format='html4', encoding="utf8")

options = {
    'encoding': 'utf-8',
}

pdfkit.from_string(html_text, output_filename, options=options)
