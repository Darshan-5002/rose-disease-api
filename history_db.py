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

def analyze_progression_and_save(plant_id: str, current_disease: str, confidence_val: float, current_severity_pct: float, severity_level: str):
    """
    Compares the current scan with the previous scan for this plant,
    determines if the disease is Improving, Stable, or Worsening,
    and saves the new scan into SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Fetch the most recent previous scan for this plant
    cursor.execute("""
        SELECT timestamp, disease, severity_percentage 
        FROM scan_history 
        WHERE plant_id = ? 
        ORDER BY id DESC LIMIT 1
    """, (plant_id,))
    prev_record = cursor.fetchone()

    # 2. Determine Progression Trend
    if prev_record is None:
        trend = "🌱 Baseline Scan (Initial Health Record)"
        feedback = "First scan recorded in database. Re-scan this rose bush in 5–7 days to track treatment response."
        delta_pct = 0.0
    else:
        prev_time_str, prev_disease, prev_severity = prev_record
        delta = current_severity_pct - prev_severity

        # If plant was previously diseased and is now healthy
        if current_disease.lower() == "healthy" and prev_disease.lower() != "healthy":
            trend = "🌟 FULL RECOVERY (100% Healed)"
            feedback = f"Outstanding! Plant transitioned from {prev_disease.capitalize()} to completely healthy foliage."
        elif current_disease.lower() != "healthy" and prev_disease.lower() == "healthy":
            trend = "🔴 NEW INFECTION DETECTED"
            feedback = f"New {current_disease.capitalize()} outbreak detected on previously healthy plant. Begin early-stage treatment."
        elif delta <= -4.0:
            # Lesion area reduced by at least 4%
            trend = f"🟢 IMPROVING (+{abs(delta):.1f}% Lesion Reduction)"
            feedback = f"Treatment is working! Infected foliage area dropped from {prev_severity:.1f}% down to {current_severity_pct:.1f}%. Continue current treatment regimen."
        elif delta >= 4.0:
            # Lesion area grew by at least 4%
            trend = f"🔴 WORSENING (+{delta:.1f}% Fungal Spread)"
            feedback = f"Warning: Disease expanded from {prev_severity:.1f}% to {current_severity_pct:.1f}%. Current treatment is ineffective; escalate to stronger fungicide immediately."
        else:
            trend = f"🟡 STABLE (±{abs(delta):.1f}% Fluctuation)"
            feedback = f"Infection is contained with no major spread ({prev_severity:.1f}% ➔ {current_severity_pct:.1f}%). Maintain treatment and re-evaluate in 3 days."

    # 3. Save current scan
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO scan_history (plant_id, timestamp, disease, confidence, severity_percentage, severity_level)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (plant_id, now_str, current_disease, confidence_val, current_severity_pct, severity_level))

    conn.commit()
    conn.close()

    return trend, feedback
