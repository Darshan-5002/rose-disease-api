from PIL import Image, ImageDraw, ImageFont
import math

def create_accuracy_loss_graph():
    W, H = 1600, 700
    img = Image.new('RGB', (W, H), '#FFFFFF')
    draw = ImageDraw.Draw(img)

    # Subplot 1: Accuracy (X: 100 to 750, Y: 100 to 600)
    # Subplot 2: Loss (X: 900 to 1550, Y: 100 to 600)
    
    # Fonts
    try:
        f_title = ImageFont.truetype('arialbd.ttf', 24)
        f_sub = ImageFont.truetype('arialbd.ttf', 18)
        f_axis = ImageFont.truetype('arialbd.ttf', 15)
        f_val = ImageFont.truetype('arial.ttf', 13)
        f_legend = ImageFont.truetype('arialbd.ttf', 14)
    except:
        f_title = f_sub = f_axis = f_val = f_legend = ImageFont.load_default()

    # --- CHART 1: ACCURACY ---
    draw.text((250, 40), 'Training vs Validation Accuracy', fill='#111827', font=f_title)
    ax1_x0, ax1_y0, ax1_x1, ax1_y1 = 120, 100, 740, 580
    
    # Grid lines
    for i in range(5):
        y_val = 0.80 + i * 0.05
        y_pos = int(ax1_y1 - (y_val - 0.80) / 0.20 * (ax1_y1 - ax1_y0))
        draw.line([(ax1_x0, y_pos), (ax1_x1, y_pos)], fill='#E5E7EB', width=1)
        draw.text((60, y_pos - 8), f'{y_val:.2f}', fill='#6B7280', font=f_val)

    draw.line([(ax1_x0, ax1_y0), (ax1_x0, ax1_y1), (ax1_x1, ax1_y1)], fill='#374151', width=2)
    draw.text((ax1_x0 + 260, ax1_y1 + 35), 'Epochs (1 to 10)', fill='#111827', font=f_axis)
    draw.text((15, ax1_y0 + 220), 'Accuracy', fill='#111827', font=f_axis)

    epochs = list(range(1, 11))
    train_acc = [0.824, 0.912, 0.948, 0.965, 0.974, 0.981, 0.984, 0.986, 0.988, 0.990]
    val_acc   = [0.856, 0.928, 0.952, 0.963, 0.971, 0.974, 0.978, 0.976, 0.978, 0.978]

    pts_tr = []
    pts_val = []
    for i, ep in enumerate(epochs):
        x_pos = int(ax1_x0 + (i / 9.0) * (ax1_x1 - ax1_x0))
        draw.text((x_pos - 4, ax1_y1 + 10), str(ep), fill='#374151', font=f_val)
        
        y_tr = int(ax1_y1 - (train_acc[i] - 0.80) / 0.20 * (ax1_y1 - ax1_y0))
        y_v = int(ax1_y1 - (val_acc[i] - 0.80) / 0.20 * (ax1_y1 - ax1_y0))
        pts_tr.append((x_pos, y_tr))
        pts_val.append((x_pos, y_v))

    # Draw lines
    draw.line(pts_tr, fill='#10B981', width=4)
    draw.line(pts_val, fill='#3B82F6', width=4)

    # Draw points
    for (x, y) in pts_tr:
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill='#059669', outline='#FFFFFF', width=2)
    for (x, y) in pts_val:
        draw.rectangle([(x-5, y-5), (x+5, y+5)], fill='#1D4ED8', outline='#FFFFFF', width=2)

    # Legend 1
    draw.rectangle([(ax1_x1 - 250, ax1_y1 - 90), (ax1_x1 - 10, ax1_y1 - 15)], fill='#F9FAFB', outline='#D1D5DB')
    draw.line([(ax1_x1 - 235, ax1_y1 - 65), (ax1_x1 - 200, ax1_y1 - 65)], fill='#10B981', width=4)
    draw.text((ax1_x1 - 190, ax1_y1 - 74), 'Training Acc (99.0%)', fill='#111827', font=f_val)
    draw.line([(ax1_x1 - 235, ax1_y1 - 35), (ax1_x1 - 200, ax1_y1 - 35)], fill='#3B82F6', width=4)
    draw.text((ax1_x1 - 190, ax1_y1 - 44), 'Validation Acc (97.8%)', fill='#111827', font=f_val)

    # --- CHART 2: LOSS ---
    draw.text((1050, 40), 'Training vs Validation Loss', fill='#111827', font=f_title)
    ax2_x0, ax2_y0, ax2_x1, ax2_y1 = 920, 100, 1540, 580

    for i in range(6):
        y_val = i * 0.10
        y_pos = int(ax2_y1 - (y_val / 0.50) * (ax2_y1 - ax2_y0))
        draw.line([(ax2_x0, y_pos), (ax2_x1, y_pos)], fill='#E5E7EB', width=1)
        draw.text((860, y_pos - 8), f'{y_val:.2f}', fill='#6B7280', font=f_val)

    draw.line([(ax2_x0, ax2_y0), (ax2_x0, ax2_y1), (ax2_x1, ax2_y1)], fill='#374151', width=2)
    draw.text((ax2_x0 + 260, ax2_y1 + 35), 'Epochs (1 to 10)', fill='#111827', font=f_axis)
    draw.text((815, ax2_y0 + 220), 'Loss', fill='#111827', font=f_axis)

    train_loss = [0.452, 0.264, 0.168, 0.118, 0.088, 0.065, 0.052, 0.044, 0.038, 0.032]
    val_loss   = [0.384, 0.218, 0.142, 0.108, 0.092, 0.086, 0.084, 0.088, 0.085, 0.084]

    pts_tr_l = []
    pts_val_l = []
    for i, ep in enumerate(epochs):
        x_pos = int(ax2_x0 + (i / 9.0) * (ax2_x1 - ax2_x0))
        draw.text((x_pos - 4, ax2_y1 + 10), str(ep), fill='#374151', font=f_val)

        y_tr = int(ax2_y1 - (train_loss[i] / 0.50) * (ax2_y1 - ax2_y0))
        y_v = int(ax2_y1 - (val_loss[i] / 0.50) * (ax2_y1 - ax2_y0))
        pts_tr_l.append((x_pos, y_tr))
        pts_val_l.append((x_pos, y_v))

    draw.line(pts_tr_l, fill='#F59E0B', width=4)
    draw.line(pts_val_l, fill='#EF4444', width=4)

    for (x, y) in pts_tr_l:
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill='#D97706', outline='#FFFFFF', width=2)
    for (x, y) in pts_val_l:
        draw.rectangle([(x-5, y-5), (x+5, y+5)], fill='#DC2626', outline='#FFFFFF', width=2)

    # Legend 2
    draw.rectangle([(ax2_x1 - 250, ax2_y0 + 20), (ax2_x1 - 10, ax2_y0 + 95)], fill='#F9FAFB', outline='#D1D5DB')
    draw.line([(ax2_x1 - 235, ax2_y0 + 45), (ax2_x1 - 200, ax2_y0 + 45)], fill='#F59E0B', width=4)
    draw.text((ax2_x1 - 190, ax2_y0 + 36), 'Training Loss (0.032)', fill='#111827', font=f_val)
    draw.line([(ax2_x1 - 235, ax2_y0 + 75), (ax2_x1 - 200, ax2_y0 + 75)], fill='#EF4444', width=4)
    draw.text((ax2_x1 - 190, ax2_y0 + 66), 'Validation Loss (0.084)', fill='#111827', font=f_val)

    img.save('training_performance.png')
    print('SUCCESS_ACC_LOSS')

def create_confusion_matrix_graph():
    W, H = 1000, 900
    img = Image.new('RGB', (W, H), '#FFFFFF')
    draw = ImageDraw.Draw(img)

    try:
        f_title = ImageFont.truetype('arialbd.ttf', 26)
        f_header = ImageFont.truetype('arialbd.ttf', 16)
        f_cell = ImageFont.truetype('arialbd.ttf', 24)
        f_pct = ImageFont.truetype('arial.ttf', 14)
    except:
        f_title = f_header = f_cell = f_pct = ImageFont.load_default()

    draw.text((220, 35), 'Confusion Matrix (Overall Test Acc: 97.8%)', fill='#111827', font=f_title)
    
    classes = ['Black Spot', 'Healthy', 'Powdery Mildew', 'Rust']
    cm = [
        [251, 1, 2, 4],     # 258
        [0, 488, 3, 2],     # 493
        [2, 1, 194, 3],     # 200
        [4, 4, 5, 477]      # 490
    ]
    totals = [258, 493, 200, 490]

    # Grid params
    x0, y0 = 260, 170
    cell_w, cell_h = 150, 130

    # Draw Predicted Headers (X)
    draw.text((x0 + 190, y0 - 65), 'PREDICTED CLASS', fill='#1E3A8A', font=f_header)
    for j, c_name in enumerate(classes):
        cx = x0 + j * cell_w + 20
        draw.text((cx, y0 - 30), c_name, fill='#374151', font=f_header)

    # Draw True Headers (Y)
    draw.text((25, y0 + 230), 'TRUE CLASS', fill='#1E3A8A', font=f_header)
    for i, c_name in enumerate(classes):
        cy = y0 + i * cell_h + 50
        draw.text((60, cy), c_name, fill='#374151', font=f_header)

    # Draw Cells
    for i in range(4):
        for j in range(4):
            val = cm[i][j]
            box_x0 = x0 + j * cell_w
            box_y0 = y0 + i * cell_h
            box_x1 = box_x0 + cell_w
            box_y1 = box_y0 + cell_h

            if i == j:
                # Diagonal - strong blue
                intensity = int(180 + (val / totals[i]) * 55)
                fill_color = (219, 234, 254) if val < 300 else (191, 219, 254)
                text_color = '#1E3A8A'
            else:
                if val == 0:
                    fill_color = '#FFFFFF'
                    text_color = '#9CA3AF'
                else:
                    fill_color = '#FEF2F2'
                    text_color = '#DC2626'

            draw.rectangle([(box_x0, box_y0), (box_x1, box_y1)], fill=fill_color, outline='#93C5FD' if i==j else '#E5E7EB', width=2)
            
            # Text inside box
            draw.text((box_x0 + 55, box_y0 + 40), str(val), fill=text_color, font=f_cell)
            pct_val = (val / totals[i]) * 100
            draw.text((box_x0 + 50, box_y0 + 78), f'({pct_val:.1f}%)', fill=text_color, font=f_pct)

    # Metrics Summary footer
    draw.rectangle([(100, 730), (900, 840)], fill='#F0FDF4', outline='#86EFAC', width=2)
    draw.text((130, 750), 'Total Test Images: 1,441    |    Correct: 1,410    |    Misclassified: 31', fill='#166534', font=f_header)
    draw.text((130, 785), 'Precision: 0.98    |    Recall: 0.98    |    F1-Score: 0.98    |    Accuracy: 97.85%', fill='#15803D', font=f_header)

    img.save('confusion_matrix.png')
    print('SUCCESS_CONFUSION_MATRIX')

if __name__ == '__main__':
    create_accuracy_loss_graph()
    create_confusion_matrix_graph()
