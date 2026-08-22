import sqlite3
import datetime

DB_PATH = "rose_scans.db"

def init_db():
    """Initializes the SQLite database table for plant scan history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity_percentage REAL NOT NULL,
            severity_level TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_all_scans(plant_id="rose_bush_1"):
    """Fetches past scans for this plant in reverse chronological order."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, plant_id, timestamp, disease, confidence, severity_percentage, severity_level 
        FROM scan_history 
        WHERE plant_id = ? 
        ORDER BY id DESC LIMIT 20
    """, (plant_id,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "plant_id": r[1],
            "timestamp": r[2],
            "disease": r[3],
            "confidence": f"{r[4]:.2f}%",
            "severity_percentage": r[5],
            "severity_level": r[6]
        })
    return history

def analyze_progression_and_save(plant_id: str, current_disease: str, confidence_val: float, current_severity_pct: float, severity_level: str):
    """
    Compares the current scan with the previous scan for this plant,
    determines if the disease is Improving, Stable, or Worsening,
    and saves the new scan into SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, disease, severity_percentage 
        FROM scan_history 
        WHERE plant_id = ? 
        ORDER BY id DESC LIMIT 1
    """, (plant_id,))
    prev_record = cursor.fetchone()

    if prev_record is None:
        trend = "🌱 Baseline Scan (Initial Health Record)"
        feedback = "First scan recorded in database. Re-scan this rose bush in 5–7 days to track treatment response."
    else:
        prev_time_str, prev_disease, prev_severity = prev_record
        delta = current_severity_pct - prev_severity

        if current_disease.lower() == "healthy" and prev_disease.lower() != "healthy":
            trend = "🌟 FULL RECOVERY (100% Healed)"
            feedback = f"Outstanding! Plant transitioned from {prev_disease.capitalize()} to completely healthy foliage."
        elif current_disease.lower() != "healthy" and prev_disease.lower() == "healthy":
            trend = "🔴 NEW INFECTION DETECTED"
            feedback = f"New {current_disease.capitalize()} outbreak detected on previously healthy plant. Begin early-stage treatment."
        elif delta <= -4.0:
            trend = f"🟢 IMPROVING (+{abs(delta):.1f}% Lesion Reduction)"
            feedback = f"Treatment is working! Infected foliage dropped from {prev_severity:.1f}% down to {current_severity_pct:.1f}%."
        elif delta >= 4.0:
            trend = f"🔴 WORSENING (+{delta:.1f}% Fungal Spread)"
            feedback = f"Warning: Disease expanded from {prev_severity:.1f}% to {current_severity_pct:.1f}%. Escalate treatment immediately."
        else:
            trend = f"🟡 STABLE (±{abs(delta):.1f}% Fluctuation)"
            feedback = f"Infection contained ({prev_severity:.1f}% ➔ {current_severity_pct:.1f}%). Maintain treatment regimen."

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO scan_history (plant_id, timestamp, disease, confidence, severity_percentage, severity_level)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (plant_id, now_str, current_disease, confidence_val, current_severity_pct, severity_level))

    conn.commit()
    conn.close()

    return trend, feedback
