import os
import re
import sys
import markdown
from fpdf import FPDF
from fpdf.fonts import FontFace

class StyledPDF(FPDF):
    def header(self):
        # We don't print header on the cover page (page 1)
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(100, 110, 120)
            self.cell(0, 8, "Research Progress Report: Robust AV Insertion in Mixed-Autonomy Roundabouts", align="L")
            self.ln(6)
            self.set_draw_color(220, 225, 230)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(100, 110, 120)
        self.cell(0, 10, f"Page {self.page_no()} / {{nb}}", align="C")

def clean_latex_math(text):
    # Replaces LaTeX math format with clean text equivalents for PDF output
    text = text.replace(r"$\to$", " -> ")
    text = text.replace(r"$\le 30\text{ m}$", " <= 30 m")
    text = text.replace(r"$> 30\text{ m}$", " > 30 m")
    text = text.replace(r"$\le 30\text{m}$", " <= 30 m")
    text = text.replace(r"$> 30\text{m}$", " > 30 m")
    text = text.replace(r"$\Delta d$", "d_distance")
    text = text.replace(r"$+0.1 \times \Delta d$", "+0.1 * d_distance")
    text = text.replace(r"$80\text{ m}$", "80 m")
    text = text.replace(r"$80\text{m}$", "80 m")
    text = text.replace(r"$13.3\text{ s}$", "13.3 s")
    text = text.replace(r"$13.3\text{s}$", "13.3 s")
    text = text.replace(r"$13.5\text{ s}$", "13.5 s")
    text = text.replace(r"$13.5\text{s}$", "13.5 s")
    text = text.replace(r"$80\%$", "80%")
    text = text.replace(r"$85\%$", "85%")
    text = text.replace(r"$100\%$", "100%")
    text = text.replace(r"$0\%$", "0%")
    text = text.replace(r"$\Delta$", "Delta")
    text = text.replace(r"\text", "")
    text = text.replace("$", "")
    return text

def convert_md_to_pdf(input_md_path, output_pdf_path, artifacts_dir):
    print(f"Reading markdown from {input_md_path}...")
    with open(input_md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Clean up LaTeX math formatting
    cleaned_md = clean_latex_math(md_content)

    # Convert local image paths to absolute file paths that FPDF can read
    def replace_image_path(match):
        alt_text = match.group(1)
        url = match.group(2)
        # Convert file:/// absolute path to standard Windows absolute path
        if url.startswith("file:///"):
            local_path = url.replace("file:///", "")
            # Double check if file exists
            if os.path.exists(local_path):
                return f"![{alt_text}]({local_path})"
        
        # Fallback: check if the image is in the artifacts directory
        filename = os.path.basename(url)
        alt_path = os.path.join(artifacts_dir, filename)
        if os.path.exists(alt_path):
            return f"![{alt_text}]({alt_path})"
            
        return match.group(0)

    cleaned_md = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_image_path, cleaned_md)

    # Convert Markdown to HTML
    print("Converting Markdown to HTML...")
    html_content = markdown.markdown(
        cleaned_md, 
        extensions=["tables", "fenced_code", "codehilite"]
    )

    # Wrap in simple HTML tags if needed, and apply some styling properties
    # Let's clean up any self-closing tags to avoid warnings
    html_content = html_content.replace("<hr />", "<hr>")
    html_content = html_content.replace("<br />", "<br>")

    # Style table headers and alternating table cell background colors via HTML attribute injection
    html_content = html_content.replace("<th>", '<th bgcolor="#e2e8f0" align="center">')
    
    # Alternating row coloring for table cells (td)
    parts = html_content.split("<tr>")
    new_parts = [parts[0]]
    for idx, part in enumerate(parts[1:], start=1):
        if idx % 2 == 1:
            part = part.replace("<td>", '<td bgcolor="#f8fafc">')
        else:
            part = part.replace("<td>", '<td>')
        new_parts.append(part)
    html_content = "<tr>".join(new_parts)

    # Initialize PDF
    pdf = StyledPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    
    # Page setup
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    
    # ------------------ COVER PAGE / HEADER BLOCK ------------------
    pdf.set_fill_color(30, 58, 138)  # Deep Navy Blue
    pdf.rect(0, 0, 210, 48, "F")
    
    pdf.set_xy(18, 12)
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "RESEARCH PROGRESS REPORT")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(191, 219, 254)  # Light Blue
    pdf.cell(0, 6, "Robust AV Insertion in Mixed-Autonomy Roundabouts")
    pdf.ln(8)
    
    pdf.set_y(54)
    pdf.set_text_color(55, 65, 81)  # Charcoal
    pdf.set_font("helvetica", "", 9.5)
    
    # Metadata Block
    pdf.set_fill_color(243, 244, 246)  # Very Light Gray
    pdf.rect(18, 54, 174, 22, "F")
    
    pdf.set_xy(22, 56)
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(30, 5, "Project Framework:")
    pdf.set_font("helvetica", "", 9)
    pdf.cell(100, 5, "Curriculum Reinforcement Learning (CRL)")
    pdf.ln(5)
    
    pdf.set_x(22)
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(30, 5, "Report Date:")
    pdf.set_font("helvetica", "", 9)
    pdf.cell(100, 5, "June 2026")
    pdf.ln(5)

    pdf.set_x(22)
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(30, 5, "Status:")
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(16, 124, 65)  # Safe Green
    pdf.cell(100, 5, "Completed & Evaluated (100% Success Rate)")
    pdf.ln(5)
    
    pdf.set_text_color(55, 65, 81)  # Reset Charcoal
    pdf.set_y(82)

    # ------------------ DEFINE TAG STYLES ------------------
    # Header sizes: h1 (14pt, Blue), h2 (11.5pt, Gray-Blue), h3 (10pt, Gray-Blue, Italic)
    # Body text: p (9.5pt)
    # fpdf2's write_html does not support style attributes or custom styling for table cells (th/td),
    # so those are styled in HTML via bgcolor and are removed here.
    tag_styles = {
        "h1": FontFace(family="helvetica", emphasis="B", size_pt=13, color="#1e3a8a"),
        "h2": FontFace(family="helvetica", emphasis="B", size_pt=11, color="#2c3e50"),
        "h3": FontFace(family="helvetica", emphasis="BI", size_pt=9.5, color="#34495e"),
        "p": FontFace(family="helvetica", size_pt=9.2, color="#374151"),
        "li": FontFace(family="helvetica", size_pt=9.2, color="#374151"),
        "strong": FontFace(family="helvetica", emphasis="B"),
        "em": FontFace(family="helvetica", emphasis="I"),
    }

    # Custom mapping of images to adjust sizes and alignments dynamically in write_html
    # write_html will parse <img> tags. Let's make sure fpdf2 can find them.
    def img_map(src):
        # Cleans and resolves paths
        clean_path = src.replace("file:///", "")
        if os.path.exists(clean_path):
            return clean_path
        # fallback to current workspace
        filename = os.path.basename(src)
        workspace_fallback = os.path.join("results", filename)
        if os.path.exists(workspace_fallback):
            return workspace_fallback
        return src

    print("Rendering HTML into PDF...")
    # write_html supports table_line_separators and tag_styles
    pdf.write_html(
        html_content,
        image_map=img_map,
        table_line_separators=True,
        tag_styles=tag_styles,
        ul_bullet_char=chr(149),
        li_prefix_color="#1e3a8a"
    )

    # Save PDF
    print(f"Saving PDF to {output_pdf_path}...")
    pdf.output(output_pdf_path)
    print("PDF conversion completed successfully!")

if __name__ == "__main__":
    artifacts_dir = r"C:\Users\hrato\.gemini\antigravity\brain\e3737dd8-ccac-46f4-85b9-24b16449e020\artifacts"
    input_md = os.path.join(artifacts_dir, "roundabout_av_rl_progress_report.md")
    output_pdf = r"c:\Users\hrato\OneDrive\Desktop\Roundabout_RL\results\roundabout_av_rl_progress_report.pdf"
    
    if not os.path.exists(input_md):
        print(f"Error: Input markdown not found at {input_md}")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    convert_md_to_pdf(input_md, output_pdf, artifacts_dir)
