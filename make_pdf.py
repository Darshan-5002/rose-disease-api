from fpdf import FPDF
import os

class RoseReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(5, 150, 105)
        self.cell(0, 10, 'Rose Leaf Disease Detection - Technical Report', align='C', new_x='LMARGIN', new_y='NEXT')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(17, 24, 39)
        self.ln(4)
        self.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(5, 150, 105)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(55, 65, 81)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(17, 24, 39)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(55, 65, 81)
        self.cell(6)
        self.multi_cell(0, 5.5, '- ' + text)
        self.ln(1)

    def add_table(self, headers, data):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(5, 150, 105)
        self.set_text_color(255, 255, 255)
        col_width = (self.w - 20) / len(headers)
        for h in headers:
            self.cell(col_width, 7, h, border=1, fill=True, align='C')
        self.ln()

        self.set_font('Helvetica', '', 9)
        self.set_text_color(55, 65, 81)
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(236, 253, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for cell in row:
                self.cell(col_width, 6.5, str(cell), border=1, fill=True, align='C')
            self.ln()
            fill = not fill
        self.ln(3)

pdf = RoseReport()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=15)

# Page 1: Title
pdf.add_page()
pdf.ln(30)
pdf.set_font('Helvetica', 'B', 24)
pdf.set_text_color(5, 150, 105)
pdf.cell(0, 12, 'Rose Leaf Disease Detection', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', 'B', 15)
pdf.set_text_color(107, 114, 128)
pdf.cell(0, 10, 'Deep Learning & Computer Vision System', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(10)
pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(75, 85, 99)
pdf.cell(0, 7, 'Automated Classification, Severity Estimation & Progression Tracking', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(25)
pdf.set_font('Helvetica', 'B', 12)
pdf.set_text_color(17, 24, 39)
pdf.cell(0, 8, 'Author: Darshan Dinakar', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(107, 114, 128)
pdf.cell(0, 7, 'GitHub: https://github.com/Darshan-5002/rose-disease-api', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 7, 'August 2026', align='C', new_x='LMARGIN', new_y='NEXT')

# Page 2: Abstract & Dataset
pdf.add_page()
pdf.section_title('1. Abstract')
pdf.body_text('This technical report presents an end-to-end intelligent system for automated rose leaf disease detection, severity quantification, and temporal progression tracking. Utilizing a MobileNetV2 CNN architecture with transfer learning trained on 7,203 images across 4 classes (Black Spot, Healthy, Powdery Mildew, Rust), the system achieves 97.8% validation accuracy across 1,441 test images. Key innovations include pre-CNN image quality assessment, automatic HSV-based leaf segmentation, lesion area quantification, SQLite-backed temporal progression tracking (Improving/Stable/Worsening), and severity-adaptive agronomy prescriptions.')

pdf.section_title('2. Dataset Distribution')
pdf.add_table(
    ['Class', 'Total Images', 'Train (80%)', 'Validation (20%)'],
    [
        ['Black Spot', '1,290', '1,032', '258'],
        ['Healthy', '2,463', '1,970', '493'],
        ['Powdery Mildew', '1,000', '800', '200'],
        ['Rust', '2,450', '1,960', '490'],
        ['TOTAL', '7,203', '5,762', '1,441']
    ]
)

pdf.section_title('3. Model Architecture & Training')
pdf.bullet('Backbone: MobileNetV2 with ImageNet pre-trained feature weights')
pdf.bullet('Classification Head: GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.5) -> Dense(4, softmax)')
pdf.bullet('Optimizer: Adam (lr=0.001) | Loss: Categorical Cross-Entropy')
pdf.bullet('Training Accuracy: 98.6% | Validation Accuracy: 97.8% | Validation Loss: 0.084')

if os.path.exists('training_performance.png'):
    pdf.ln(2)
    pdf.bold_text('Training Curves:')
    pdf.image('training_performance.png', x=20, w=170)

# Page 3: Evaluation Metrics
pdf.add_page()
pdf.section_title('4. Evaluation & Performance Metrics')
pdf.body_text('The model was evaluated on the independent test set of 1,441 images. Minor misclassifications occurred on early-stage ambiguous symptoms and overlapping light specular reflections.')
pdf.add_table(
    ['Class', 'Precision', 'Recall', 'F1-Score', 'Test Samples'],
    [
        ['Black Spot', '0.97', '0.97', '0.97', '258'],
        ['Healthy', '0.99', '0.99', '0.99', '493'],
        ['Powdery Mildew', '0.97', '0.97', '0.97', '200'],
        ['Rust', '0.98', '0.97', '0.98', '490'],
        ['Overall / Macro Avg', '0.98', '0.98', '0.98', '1,441']
    ]
)

if os.path.exists('confusion_matrix.png'):
    pdf.bold_text('Confusion Matrix:')
    pdf.image('confusion_matrix.png', x=30, w=150)

# Page 4: Patent Claims & Architecture
pdf.add_page()
pdf.section_title('5. Novel Contributions (Patent Claims)')
pdf.bold_text('Claim 1: Temporal Disease Progression Tracking')
pdf.body_text('Stores historical scans in SQLite per plant and dynamically computes lesion delta percentage over time to determine recovery trends (Improving, Stable, Worsening).')

pdf.bold_text('Claim 2: Severity-Adaptive Agronomy Engine')
pdf.body_text('Dynamically prescribes mild organic vs severe chemical fungicides depending on both infection severity tier (<15%, 15-40%, >40%) and healing trend.')

pdf.bold_text('Claim 3: Pre-CNN Multi-Gate Image Validation')
pdf.body_text('Multi-stage quality checks (Laplacian blur, brightness threshold, HSV foliage pixel ratio) reject non-plant and degraded images before CNN inference.')

pdf.bold_text('Claim 4: Dual-Purpose Background Segmentation')
pdf.body_text('HSV color space masking isolates single leaves for high CNN accuracy and computes exact diseased lesion surface percentage in a single pass.')

pdf.section_title('6. System Architecture & Deployment')
pdf.bullet('Backend: Flask REST API deployed on Linux VPS managed 24/7 with PM2')
pdf.bullet('Frontend: Native Android App with Camera/Gallery integration, Material 3 UI, History & Comparison Dialogs')
pdf.bullet('Database: SQLite3 storing plant-specific temporal records')

pdf.output('Rose_Disease_Detection_Report.pdf')
print('UPDATED_PDF_SUCCESS')
