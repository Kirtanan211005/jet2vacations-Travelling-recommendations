# ✈ Jet2Vacations — AI-Powered Travel Recommendation System

**Course:** INT4203 – Artificial Intelligence  
**Module:** Case Study 4 – Recommendation Systems  
**Institute:** INTI International University  
**SDG Alignment:** SDG 8 (Decent Work) · SDG 11 (Sustainable Cities)

---

## 📌 Project Overview

Jet2Vacations is a **Hybrid AI-Powered Travel Destination Recommendation System** that combines:

- **SVD Collaborative Filtering** — learns from user rating patterns
- **TF-IDF Content-Based Filtering** — matches destinations to user interests
- **Weighted Hybrid Fusion** — combines both with optimal weight α = 0.6
- **Real Live APIs** — Teleport API + Wikipedia REST API for authentic city data
- **Professional Web UI** — Gradio dashboard with booking + Telegram notifications

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `jet2vacations_ai_model.py` | Core AI recommendation engine with evaluation metrics and charts |
| `jet2vacations.py` | Full professional web application (Gradio UI + booking system) |
| `README.md` | This file |
| `requirements.txt` | All required Python packages |

---

## 🚀 Quick Start — Google Colab

### Step 1 — Install Dependencies
```python
!pip install numpy==1.26.4 scikit-surprise scikit-learn gradio pandas numpy matplotlib seaborn requests
```
Restart runtime after: **Runtime → Restart Runtime**

---

### Step 2A — Run AI Model (Charts + Evaluation)
```python
%matplotlib inline
# Paste entire jet2vacations_ai_model.py content here
main()
```

Then display charts:
```python
from IPython.display import Image, display
display(Image("jet2vacations_results.png"))
display(Image("jet2vacations_training_curve.png"))
```

---

### Step 2B — Run Full Web App (Dashboard + Booking)
```python
# Paste entire jet2vacations.py content here

# Fixed Telegram function (paste this right before launch_app)
def send_telegram_notification(booking, bot_token, chat_id):
    if not bot_token or not chat_id:
        return "skipped"
    try:
        msg = (
            "Jet2Vacations - Booking Confirmed!\n\n"
            f"Booking Ref: {booking['ref']}\n"
            f"Destination: {booking['destination']}, {booking['country']}\n"
            f"Departure:   {booking['departure']}\n"
            f"Return:      {booking['return_date']}\n"
            f"Duration:    {booking['nights']} nights\n"
            f"Travellers:  {booking['travelers']}\n\n"
            f"Flights: ${booking['flight_total']:.0f}\n"
            f"Hotel:   ${booking['hotel_total']:.0f}\n"
            f"TOTAL:   ${booking['grand_total']:.0f} USD\n\n"
            f"Email: {booking['email']}\n"
            f"Booked: {booking['booked_at']}\n\n"
            "Your trip is confirmed. Safe travels!\n"
            "Jet2Vacations AI - INT4203 - INTI University"
        )
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=10)
        if r.status_code == 200 and r.json().get("ok"):
            return "success"
        else:
            return f"failed: {r.json().get('description', 'Unknown error')}"
    except Exception as e:
        return f"failed: {str(e)}"

import gradio as gr
gr.close_all()
launch_app()
```

Click the **gradio.live** public URL that appears in the output.

---

## 🤖 How the AI Works

```
User Preferences (interests, budget, climate, region)
         |
         |---> SVD Collaborative Filter ---> CF Score
         |      (trained on 14,839 ratings     (preference alignment
         |       seeded by real Teleport         + matrix factorisation)
         |       popularity scores)
         |
         |---> TF-IDF Content-Based Filter ---> CBF Score
         |      (Wikipedia descriptions           (cosine similarity
         |       + destination features)           to user profile)
         |
         +--> Hybrid Fusion --> Final Score
               S(u,i) = 0.6 x CF + 0.4 x CBF
               (alpha = 0.6, validated via cross-validation)
```

---

## 📊 Model Performance

| Model | RMSE | Precision@10 | NDCG@10 |
|-------|------|--------------|---------|
| Random Baseline | 1.52 | 0.172 | 0.321 |
| KNN-CF (Pearson) | 1.06 | 0.206 | 0.399 |
| SVD Collaborative | 1.09 | 0.210 | 0.400 |
| TF-IDF Content-Based | 1.14 | 0.203 | 0.418 |
| **Jet2Vacations Hybrid** | **1.00** | **0.231** | **0.461** |

Hybrid model outperforms all baselines across every metric.

---

## 🌐 Live Data Sources

| API | Purpose | Key Required |
|-----|---------|--------------|
| Teleport API (api.teleport.org) | Real city quality scores (culture, outdoor, safety 0-10) | No |
| Wikipedia REST API (en.wikipedia.org/api/rest_v1) | Real destination descriptions for TF-IDF | No |

Both APIs are completely free with no registration required.

---

## 📱 Telegram Booking Notifications Setup

1. Open Telegram → search **@BotFather** → send `/newbot`
2. Copy the **Bot Token** (e.g. `7234567890:AAFxxx...`)
3. Start your bot → visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Copy the **id** value from **chat** in the JSON response — that is your Chat ID
5. Paste both into the booking form in the web app

---

## 🖥 Web App Features

| Page | Features |
|------|----------|
| Dashboard | 4 stat cards + 8 trending destinations grid |
| Find Destinations | 4-step AI wizard (interests, budget, style, results) |
| Book a Trip | Traveller details, dates, price breakdown, Telegram notification |
| My Bookings | Session booking history table with total spend |
| Compare Cities | Side-by-side comparison of any 2 destinations |

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| Destinations | 40 global cities |
| Simulated Users | 1,000 |
| Total Ratings | ~14,839 |
| Rating Scale | 1.0 – 5.0 |
| Data Seeding | Real Teleport popularity scores |
| Regions Covered | 13 regions across 6 continents |

---

## 🔧 Requirements

```
numpy==1.26.4
pandas>=1.5.0
scikit-learn>=1.2.0
scikit-surprise>=1.1.3
matplotlib>=3.6.0
seaborn>=0.12.0
gradio>=4.0.0
requests>=2.28.0
```

Install all at once:
```
pip install numpy==1.26.4 scikit-surprise scikit-learn gradio pandas matplotlib seaborn requests
```



