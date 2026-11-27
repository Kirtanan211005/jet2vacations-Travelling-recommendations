"""
=============================================================================
Jet2Vacations — AI-Powered Travel Management System
=============================================================================
Professional dashboard UI with:
  - Left sidebar navigation (like reference image)
  - Dashboard with summary cards & stats
  - AI Destination Finder (4-step wizard)
  - Book a Trip (dates, travelers, email)
  - Booking Confirmation with email preview
  - My Bookings table (session history)
  - Compare Cities side-by-side
  
Data Sources:
  - Teleport API (real city quality scores)
  - Wikipedia REST API (real descriptions)

Run: paste into Colab cell → launch_app()
=============================================================================
"""

import time, warnings, json, random, string, datetime
import numpy as np
import pandas as pd
import requests
warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    from surprise import SVD, Dataset, Reader, accuracy
    SURPRISE_OK = True
except Exception:
    SURPRISE_OK = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ─── City Catalogue ──────────────────────────────────────────────────────────
CITY_CATALOGUE = [
    ("Tokyo","Japan","tokyo","Tokyo","East Asia","cultural historical",180,"✈"),
    ("Kyoto","Japan","tokyo","Kyoto","East Asia","cultural historical",120,"⛩"),
    ("Bali","Indonesia","bali","Bali","Southeast Asia","beach wellness",80,"🌴"),
    ("Bangkok","Thailand","bangkok","Bangkok","Southeast Asia","cultural adventure",60,"🛕"),
    ("Chiang Mai","Thailand","chiang-mai","Chiang_Mai","Southeast Asia","cultural adventure",55,"🐘"),
    ("Singapore","Singapore","singapore","Singapore","Southeast Asia","urban luxury",200,"🦁"),
    ("Ho Chi Minh","Vietnam","ho-chi-minh-city","Ho_Chi_Minh_City","Southeast Asia","cultural culinary",45,"🏮"),
    ("Hanoi","Vietnam","hanoi","Hanoi","Southeast Asia","cultural historical",40,"👑"),
    ("Kuala Lumpur","Malaysia","kuala-lumpur","Kuala_Lumpur","Southeast Asia","urban cultural",70,"🏙"),
    ("Dubai","UAE","dubai","Dubai","Middle East","luxury adventure",220,"🕌"),
    ("Istanbul","Turkey","istanbul","Istanbul","Eurasia","cultural historical",90,"🕌"),
    ("Athens","Greece","athens","Athens","Europe","cultural historical",110,"🏛"),
    ("Rome","Italy","rome","Rome","Europe","cultural historical",150,"🏛"),
    ("Florence","Italy","florence","Florence","Europe","cultural luxury",140,"🎨"),
    ("Paris","France","paris","Paris","Europe","cultural luxury",200,"🗼"),
    ("Barcelona","Spain","barcelona","Barcelona","Europe","beach cultural",140,"🏖"),
    ("Madrid","Spain","madrid","Madrid","Europe","cultural culinary",130,"💃"),
    ("Lisbon","Portugal","lisbon","Lisbon","Europe","cultural coastal",110,"🚃"),
    ("Porto","Portugal","porto","Porto","Europe","cultural culinary",100,"🍷"),
    ("Amsterdam","Netherlands","amsterdam","Amsterdam","Europe","cultural heritage",160,"🚲"),
    ("Prague","Czechia","prague","Prague","Europe","cultural historical",90,"🏰"),
    ("Vienna","Austria","vienna","Vienna","Europe","cultural luxury",140,"🎵"),
    ("Budapest","Hungary","budapest","Budapest","Europe","cultural heritage",80,"🏰"),
    ("Copenhagen","Denmark","copenhagen","Copenhagen","Scandinavia","cultural nature",220,"🚲"),
    ("Stockholm","Sweden","stockholm","Stockholm","Scandinavia","cultural nature",200,"⚓"),
    ("Zurich","Switzerland","zurich","Zürich","Europe","nature luxury",280,"🏔"),
    ("Cape Town","South Africa","cape-town","Cape_Town","Africa","adventure nature",100,"🦁"),
    ("Marrakech","Morocco","marrakech","Marrakesh","North Africa","cultural adventure",70,"🐪"),
    ("Cairo","Egypt","cairo","Cairo","North Africa","cultural historical",55,"🏺"),
    ("Nairobi","Kenya","nairobi","Nairobi","East Africa","adventure nature",80,"🦒"),
    ("Buenos Aires","Argentina","buenos-aires","Buenos_Aires","South America","urban cultural",90,"💃"),
    ("Rio de Janeiro","Brazil","rio-de-janeiro","Rio_de_Janeiro","South America","beach cultural",100,"🎭"),
    ("Bogota","Colombia","bogota","Bogotá","South America","urban cultural",65,"☕"),
    ("Lima","Peru","lima","Lima","South America","cultural culinary",70,"🦙"),
    ("Mexico City","Mexico","mexico-city","Mexico_City","Latin America","cultural historical",80,"🌮"),
    ("New York","USA","new-york","New_York_City","North America","urban luxury",220,"🗽"),
    ("San Francisco","USA","san-francisco-bay-area","San_Francisco","North America","urban nature",200,"🌉"),
    ("Vancouver","Canada","vancouver","Vancouver","North America","nature adventure",160,"🏔"),
    ("Sydney","Australia","sydney","Sydney","Oceania","beach urban",180,"🦘"),
    ("Auckland","New Zealand","auckland","Auckland","Oceania","adventure nature",160,"🏔"),
]

# ─── API Functions ─────────────────────────────────────────────────────────────
def fetch_teleport_scores(slug, timeout=8):
    try:
        url = f"https://api.teleport.org/api/urban_areas/slug:{slug}/scores/"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200: return None
        scores = {}
        for cat in r.json().get("categories", []):
            name = cat["name"].lower().replace(" ", "_").replace("&", "and")
            scores[name] = round(cat["score_out_of_10"], 2)
        return scores
    except: return None

def fetch_wikipedia_summary(wiki_title, timeout=8):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title.replace(' ','_')}"
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Jet2Vacations/1.0"})
        if r.status_code != 200: return None
        return r.json().get("extract", "")[:800]
    except: return None

def fetch_all_real_data(city_catalogue, verbose=True):
    if verbose:
        print("  Fetching real data from Teleport API + Wikipedia...")
    records, successes = [], 0
    for i, row in enumerate(city_catalogue):
        name, country, slug, wiki_title, region, cat, cost, emoji = row
        t_scores = fetch_teleport_scores(slug)
        time.sleep(0.3)
        wiki_desc = fetch_wikipedia_summary(wiki_title)
        time.sleep(0.2)
        feature_text = f"{name} {country} {region} {cat} {(wiki_desc or '')[:300]}"
        if t_scores:
            culture  = t_scores.get("culture", 5.0)
            outdoors = t_scores.get("outdoors", 5.0)
            safety   = t_scores.get("safety", 5.0)
            leisure  = t_scores.get("leisure_and_culture", culture)
            popularity = (culture + outdoors + leisure) / 3.0
            successes += 1
            data_source = "Teleport API + Wikipedia"
        else:
            culture  = 6.5 if any(c in cat for c in ["cultural","historical","heritage","luxury"]) else 5.0
            outdoors = 7.0 if "nature" in cat or "adventure" in cat else 5.0
            safety   = 6.0
            popularity = (culture + outdoors) / 2
            data_source = "Estimated"
        short_desc = ((wiki_desc[:200] + "...") if wiki_desc and len(wiki_desc) > 200
                      else wiki_desc or f"A remarkable destination in {country}.")
        records.append({
            "destination_id": f"D{i+1:03d}", "name": name, "country": country,
            "region": region, "category": cat, "avg_cost_usd": cost, "emoji": emoji,
            "feature_text": feature_text, "description": short_desc,
            "real_culture_score": round(culture, 2), "real_outdoor_score": round(outdoors, 2),
            "real_safety_score": round(safety, 2), "real_popularity": round(popularity, 2),
            "data_source": data_source,
        })
        if verbose and (i+1) % 10 == 0:
            print(f"    {i+1}/40 cities processed ({successes} with real API data)")
    df = pd.DataFrame(records)
    if verbose:
        real = (df["data_source"] == "Teleport API + Wikipedia").sum()
        print(f"  Done: {real}/40 cities with live data")
    return df

def generate_real_seeded_ratings(dests_df, n_users=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_dest = len(dests_df)
    pop = dests_df["real_popularity"].values
    base_ratings = 2.5 + 2.0 * (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    archetypes = {
        0: {"cultural":1.4,"historical":1.4,"heritage":1.3,"budget_bonus":0},
        1: {"luxury":1.5,"urban":1.3,"beach":1.2,"budget_bonus":-0.4},
        2: {"adventure":1.5,"nature":1.5,"budget_bonus":0},
        3: {"beach":1.5,"wellness":1.3,"coastal":1.3,"budget_bonus":0},
        4: {"budget_bonus":0.5,"cost_penalty":True},
    }
    user_archs = rng.choice(5, size=n_users)
    base_time = int(time.time()) - 2*365*86400
    rows = []
    for uid_int in range(1, n_users+1):
        arch = archetypes[user_archs[uid_int-1]]
        w = np.ones(n_dest)
        for di, row in dests_df.iterrows():
            cat = row["category"].lower()
            w[di] += row["real_culture_score"] * 0.15 + row["real_outdoor_score"] * 0.10
            for kwd, mult in arch.items():
                if isinstance(mult, float) and kwd not in ("budget_bonus","cost_penalty"):
                    if kwd in cat: w[di] *= mult
            if arch.get("cost_penalty") and row["avg_cost_usd"] > 150: w[di] *= 0.5
        w = np.clip(w, 0.01, None); w /= w.sum()
        chosen = rng.choice(n_dest, size=min(int(rng.integers(6,25)), n_dest), replace=False, p=w)
        for di in chosen:
            row = dests_df.iloc[di]
            base = float(base_ratings[di])
            for kwd, mult in arch.items():
                if isinstance(mult, float) and kwd in row["category"].lower():
                    base += (mult-1.0)*0.4
            base += arch.get("budget_bonus",0) * (1 if row["avg_cost_usd"]<100 else -0.5)
            base += (row["real_safety_score"]-5.0)*0.1
            rating = float(np.clip(round((base+rng.normal(0,0.45))*2)/2, 1.0, 5.0))
            ts = base_time + int(rng.integers(0, 2*365*86400))
            rows.append({"user_id":f"U{uid_int:04d}","destination_id":row["destination_id"],
                         "rating":rating,"timestamp":ts})
    return pd.DataFrame(rows).drop_duplicates(["user_id","destination_id"]).reset_index(drop=True)

def get_recommendations(user_name, interests, budget, climate, region,
                        companion, adventure_level, dests_df, ratings_df,
                        svd_model, top_k=10, alpha=0.6):
    if not interests:
        return None, "Please select at least one interest.", "", ""
    n_dest = len(dests_df)
    interest_map = {
        "Cultural & History":    ["cultural","historical","heritage","urban"],
        "Adventure":             ["adventure","nature","outdoor"],
        "Beach & Islands":       ["beach","coastal"],
        "Nature & Hiking":       ["nature","adventure","outdoor"],
        "Food & Cuisine":        ["culinary","cultural"],
        "Wellness & Relaxation": ["wellness","cultural"],
        "Photography":           ["cultural","nature","heritage"],
        "UNESCO Heritage":       ["heritage","historical","cultural"],
        "Luxury Travel":         ["luxury","urban"],
        "Budget Travel":         [],
    }
    profile_text = " ".join(interests) + " " + (climate if climate!="Any" else "") + " " + (region if region!="Anywhere" else "")
    combined = list(dests_df["feature_text"]) + [profile_text]
    tv = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english", sublinear_tf=True)
    mat = tv.fit_transform(combined)
    cbf_raw = cosine_similarity(mat[-1], mat[:-1]).flatten()
    cf_raw = np.zeros(n_dest)
    for i, row in dests_df.iterrows():
        s = 0.0; cat = row["category"].lower()
        for interest in interests:
            for c in interest_map.get(interest, []):
                if c in cat: s += 1.5
        s += max(0, 1 - abs(row["avg_cost_usd"] - budget) / 200) * 1.5
        if "Budget Travel" in interests and row["avg_cost_usd"] <= 80: s += 2.0
        if "Luxury Travel"  in interests and row["avg_cost_usd"] >= 180: s += 1.5
        dest_adv = 80 if ("adventure" in cat or "nature" in cat) else 20 if "wellness" in cat or "luxury" in cat else 50
        s += (1 - abs(dest_adv - adventure_level) / 100) * 1.2
        if climate != "Any":
            cmap = {"Tropical":["beach","southeast asia"],"Temperate":["europe","north america"],
                    "Mediterranean":["europe","middle east","north africa"],"Alpine":["switzerland","austria","new zealand"],
                    "Arid":["middle east","north africa"],"Coastal":["coastal","beach"]}
            cwords = cmap.get(climate, [climate.lower()])
            if any(cw in row["region"].lower() or cw in cat for cw in cwords): s += 2.0
        if region != "Anywhere" and region.split()[0].lower() in row["region"].lower(): s += 2.5
        comp_map = {"Solo":["adventure","cultural"],"Couple":["luxury","cultural","beach","coastal"],
                    "Family":["cultural","heritage","nature","beach"],"Friends":["adventure","beach","culinary","urban"],
                    "Group Tour":["cultural","heritage","historical"]}
        for c in comp_map.get(companion, []):
            if c in cat: s += 0.5
        if any(i in interests for i in ["Cultural & History","UNESCO Heritage"]): s += row["real_culture_score"]*0.15
        if any(i in interests for i in ["Adventure","Nature & Hiking"]): s += row["real_outdoor_score"]*0.15
        s += row["real_safety_score"]*0.05
        if svd_model:
            try: s += (svd_model.predict("U0001", row["destination_id"]).est - 3.0)*0.5
            except: pass
        cf_raw[i] = s
    def norm(arr):
        mn, mx = arr.min(), arr.max(); return (arr-mn)/(mx-mn+1e-9)
    cf_n = norm(cf_raw); cbf_n = norm(cbf_raw)
    hybrid = alpha*cf_n + (1-alpha)*cbf_n
    results = []
    for i, row in dests_df.iterrows():
        results.append({
            "Rank":0, "Emoji":row["emoji"], "Destination":row["name"],
            "Country":row["country"], "Category":row["category"].title(),
            "Cost/Day":f"${row['avg_cost_usd']}",
            "Culture":f"{row['real_culture_score']:.1f}/10",
            "Outdoor":f"{row['real_outdoor_score']:.1f}/10",
            "Safety":f"{row['real_safety_score']:.1f}/10",
            "CF%":f"{cf_n[i]*100:.0f}%", "CBF%":f"{cbf_n[i]*100:.0f}%",
            "Match":f"{hybrid[i]*100:.0f}%",
            "Source":row["data_source"], "Desc":row["description"],
            "_hybrid":float(hybrid[i]), "_cost":row["avg_cost_usd"],
        })
    df = pd.DataFrame(results).sort_values("_hybrid",ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    top_df = df.head(top_k).copy()
    display_df = top_df.drop(columns=["_hybrid","Desc","Emoji"]).rename(
        columns={"CF%":"CF Score","CBF%":"CBF Score","Match":"Match %",
                 "Source":"Data Source","Cost/Day":"Avg Cost/Day"}).reset_index(drop=True)
    return top_df, f"Found {top_k} destinations for {user_name or 'you'}!", display_df


# ─── HTML Builder Functions ────────────────────────────────────────────────────

def make_results_cards(top_df, user_name, budget, alpha, dests_df):
    """Build professional destination result cards"""
    name_str = f"for <b style='color:#0052cc'>{user_name}</b>" if user_name else ""
    real_count = (dests_df["data_source"]=="Teleport API + Wikipedia").sum()
    html = f"""
    <style>
      @keyframes fadeUp{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
      @keyframes shimmer{{0%,100%{{box-shadow:0 0 0 0 rgba(0,82,204,.25)}}50%{{box-shadow:0 4px 20px rgba(0,82,204,.15)}}}}
      .j2-card{{background:#fff;border:1.5px solid #e2e8f0;border-radius:12px;
                margin-bottom:12px;overflow:hidden;animation:fadeUp .35s ease both;
                transition:transform .18s,border-color .18s,box-shadow .18s;}}
      .j2-card:hover{{transform:translateY(-2px);border-color:#0052cc;box-shadow:0 6px 20px rgba(0,82,204,.12);}}
      .j2-top{{border-color:#f59e0b;animation:fadeUp .35s ease both,shimmer 3s 1s ease infinite;}}
      .j2-s2{{border-color:#94a3b8;}} .j2-s3{{border-color:#cd7f32;}}
      .j2-bar{{height:5px;background:#e2e8f0;border-radius:3px;overflow:hidden;margin-top:4px;}}
      .j2-fill{{height:100%;border-radius:3px;}}
      .j2-tag{{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.65rem;
               font-weight:600;margin:2px 3px 2px 0;border:1px solid;}}
      .j2-realb{{background:#ecfdf5;color:#059669;border:1px solid #6ee7b7;
                 font-size:.62rem;padding:2px 8px;border-radius:4px;font-weight:600;}}
      .j2-estb{{background:#fffbeb;color:#d97706;border:1px solid #fcd34d;
                font-size:.62rem;padding:2px 8px;border-radius:4px;font-weight:600;}}
    </style>
    <div style="font-family:'Segoe UI',sans-serif;">
      <div style="background:linear-gradient(90deg,#0052cc,#0ea5e9);border-radius:10px;
           padding:14px 18px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
        <div style="font-size:2rem;">✈</div>
        <div>
          <div style="font-size:1rem;font-weight:700;color:#fff;">
            Top {len(top_df)} Destinations {name_str}
          </div>
          <div style="font-size:.73rem;color:rgba(255,255,255,.8);margin-top:2px;">
            Live Teleport API + Wikipedia · α={alpha:.1f} CF + {1-alpha:.1f} CBF · ~${budget}/day budget · {real_count}/40 cities with real data
          </div>
        </div>
      </div>
    """
    medals = {1:"🥇",2:"🥈",3:"🥉"}
    card_cls = {1:"j2-top",2:"j2-s2",3:"j2-s3"}
    accents  = {1:"#f59e0b",2:"#94a3b8",3:"#cd7f32"}
    for delay_i,(_, row) in enumerate(top_df.iterrows()):
        rank   = int(row["Rank"])
        cf_v   = float(row["CF%"].replace("%",""))
        cb_v   = float(row["CBF%"].replace("%",""))
        hy_v   = float(row["Match"].replace("%",""))
        delay  = f"animation-delay:{delay_i*0.07:.2f}s;"
        is_real = "Teleport" in str(row.get("Source",""))
        acc = accents.get(rank, "#0052cc")
        cls = card_cls.get(rank, "")
        badge = f"<span class='j2-realb'>✓ Live API</span>" if is_real else "<span class='j2-estb'>Estimated</span>"
        medal = medals.get(rank, row["Emoji"])
        label = "BEST MATCH" if rank==1 else f"#{rank}"
        hy_color = "#059669" if hy_v>=75 else "#0052cc" if hy_v>=50 else "#94a3b8"
        html += f"""
        <div class="j2-card {cls}" style="{delay}">
          <div style="display:flex;align-items:stretch;">
            <div style="background:{'#fffbeb' if rank==1 else '#f8fafc'};
                 border-right:1.5px solid {acc}30;padding:14px 10px;
                 text-align:center;min-width:58px;flex-shrink:0;">
              <div style="font-size:1.5rem;">{medal}</div>
              <div style="font-size:.58rem;color:{acc};font-weight:800;
                   letter-spacing:.05em;margin-top:4px;">{label}</div>
            </div>
            <div style="flex:1;padding:12px 15px;">
              <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;flex-wrap:wrap;">
                <div>
                  <div style="font-size:1rem;font-weight:700;color:#1e293b;">
                    {row['Emoji']} {row['Destination']}
                  </div>
                  <div style="font-size:.74rem;color:#64748b;margin-top:2px;">
                    📍 {row['Country']} &nbsp;·&nbsp; 💰 {row['Cost/Day']}/day &nbsp;·&nbsp; 🛡 Safety {row['Safety']}
                  </div>
                  <div style="margin-top:6px;">
                    <span class="j2-tag" style="background:#eff6ff;color:#2563eb;border-color:#bfdbfe;">
                      {row['Category'].split()[0]}
                    </span>
                    <span class="j2-tag" style="background:#f0fdf4;color:#16a34a;border-color:#bbf7d0;">
                      Culture {row['Culture']}
                    </span>
                    <span class="j2-tag" style="background:#fefce8;color:#ca8a04;border-color:#fef08a;">
                      Outdoors {row['Outdoor']}
                    </span>
                    {badge}
                  </div>
                </div>
                <div style="text-align:center;background:#f8fafc;border-radius:10px;
                     padding:8px 14px;border:1.5px solid {acc}40;flex-shrink:0;min-width:72px;">
                  <div style="font-size:1.4rem;font-weight:800;color:{hy_color};line-height:1;">{hy_v:.0f}%</div>
                  <div style="font-size:.58rem;color:#94a3b8;letter-spacing:.08em;margin-top:1px;">MATCH</div>
                </div>
              </div>
              <div style="margin-top:8px;font-size:.72rem;color:#64748b;line-height:1.5;font-style:italic;">
                {str(row.get('Desc',''))[:170]}
              </div>
              <div style="margin-top:10px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
                <div>
                  <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:.63rem;color:#94a3b8;font-weight:600;">CF SCORE</span>
                    <span style="font-size:.63rem;color:#ef4444;font-weight:700;">{cf_v:.0f}%</span>
                  </div>
                  <div class="j2-bar"><div class="j2-fill" style="width:{cf_v}%;background:linear-gradient(90deg,#ef4444,#f97316);"></div></div>
                </div>
                <div>
                  <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:.63rem;color:#94a3b8;font-weight:600;">CBF SCORE</span>
                    <span style="font-size:.63rem;color:#10b981;font-weight:700;">{cb_v:.0f}%</span>
                  </div>
                  <div class="j2-bar"><div class="j2-fill" style="width:{cb_v}%;background:linear-gradient(90deg,#10b981,#059669);"></div></div>
                </div>
                <div>
                  <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:.63rem;color:#94a3b8;font-weight:600;">HYBRID</span>
                    <span style="font-size:.63rem;font-weight:700;color:{hy_color};">{hy_v:.0f}%</span>
                  </div>
                  <div class="j2-bar"><div class="j2-fill" style="width:{hy_v}%;background:linear-gradient(90deg,#0052cc,#0ea5e9);"></div></div>
                </div>
              </div>
            </div>
          </div>
        </div>"""
    html += "</div>"
    return html


def make_booking_confirmation(booking, tg_result=""):
    """Build a professional booking confirmation card"""
    status_color = "#059669"; status_bg = "#ecfdf5"
    html = f"""
    <style>
      @keyframes confPop{{0%{{opacity:0;transform:scale(.9)}}100%{{opacity:1;transform:scale(1)}}}}
      .conf-card{{animation:confPop .4s ease;}}
    </style>
    <div class="conf-card" style="font-family:'Segoe UI',sans-serif;max-width:640px;">
      <div style="background:linear-gradient(90deg,#0052cc,#0ea5e9);border-radius:14px 14px 0 0;
           padding:20px 24px;text-align:center;">
        <div style="font-size:2.2rem;margin-bottom:6px;">✅</div>
        <div style="font-size:1.2rem;font-weight:700;color:#fff;">Booking Confirmed!</div>
        <div style="font-size:.82rem;color:rgba(255,255,255,.85);margin-top:4px;">
          Your trip has been successfully booked
        </div>
      </div>
      <div style="background:#fff;border:1px solid #e2e8f0;border-top:none;border-radius:0 0 14px 14px;padding:20px 24px;">

        <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;
             padding:12px 16px;margin-bottom:16px;text-align:center;">
          <div style="font-size:.72rem;color:#0369a1;font-weight:700;letter-spacing:.1em;">BOOKING REFERENCE</div>
          <div style="font-size:1.5rem;font-weight:800;color:#0052cc;letter-spacing:.15em;margin-top:4px;">
            {booking['ref']}
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
          <div style="background:#f8fafc;border-radius:8px;padding:12px;">
            <div style="font-size:.68rem;color:#94a3b8;font-weight:700;margin-bottom:4px;">DESTINATION</div>
            <div style="font-size:.92rem;font-weight:700;color:#1e293b;">{booking['emoji']} {booking['destination']}</div>
            <div style="font-size:.75rem;color:#64748b;">{booking['country']}</div>
          </div>
          <div style="background:#f8fafc;border-radius:8px;padding:12px;">
            <div style="font-size:.68rem;color:#94a3b8;font-weight:700;margin-bottom:4px;">TRAVEL DATES</div>
            <div style="font-size:.82rem;font-weight:600;color:#1e293b;">
              {booking['departure']} → {booking['return_date']}
            </div>
            <div style="font-size:.75rem;color:#64748b;">{booking['nights']} nights · {booking['travelers']} traveller(s)</div>
          </div>
        </div>

        <div style="background:#fff8f1;border:1px solid #fed7aa;border-radius:8px;padding:14px;margin-bottom:16px;">
          <div style="font-size:.72rem;color:#92400e;font-weight:700;letter-spacing:.08em;margin-bottom:10px;">
            PRICE BREAKDOWN
          </div>
          <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #fed7aa;">
            <span style="font-size:.82rem;color:#78350f;">✈ Flights ({booking['travelers']} × return)</span>
            <span style="font-size:.82rem;font-weight:600;color:#1e293b;">${booking['flight_total']:.0f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #fed7aa;">
            <span style="font-size:.82rem;color:#78350f;">🏨 Hotel ({booking['nights']} nights)</span>
            <span style="font-size:.82rem;font-weight:600;color:#1e293b;">${booking['hotel_total']:.0f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0 0;">
            <span style="font-size:.88rem;font-weight:700;color:#92400e;">TOTAL</span>
            <span style="font-size:1.05rem;font-weight:800;color:#0052cc;">${booking['grand_total']:.0f}</span>
          </div>
        </div>

        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px;margin-bottom:16px;">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:1.1rem;">{"✅" if tg_result=="success" else "📱"}</span>
            <div>
              <div style="font-size:.75rem;font-weight:700;color:#166534;">
                TELEGRAM NOTIFICATION — {"SENT ✓" if tg_result=="success" else "NOT CONFIGURED" if tg_result=="skipped" else "PREVIEW BELOW"}
              </div>
              <div style="font-size:.78rem;color:#166634;">Email: <b>{booking['email']}</b></div>
            </div>
          </div>
        </div>

        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <div style="flex:1;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:.68rem;color:#1d4ed8;font-weight:700;">STATUS</div>
            <div style="font-size:.88rem;font-weight:700;color:#059669;">✓ Confirmed</div>
          </div>
          <div style="flex:1;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:.68rem;color:#1d4ed8;font-weight:700;">BOOKED ON</div>
            <div style="font-size:.82rem;font-weight:600;color:#1e293b;">{booking['booked_at']}</div>
          </div>
          <div style="flex:1;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:.68rem;color:#1d4ed8;font-weight:700;">CATEGORY</div>
            <div style="font-size:.82rem;font-weight:600;color:#1e293b;">{booking['category']}</div>
          </div>
        </div>

        <div style="margin-top:14px;padding:10px;background:#f8fafc;border-radius:8px;
             font-size:.7rem;color:#94a3b8;text-align:center;line-height:1.7;">
          ✈ Jet2Vacations · INT4203 AI · INTI International University · SDG 8 · SDG 11
        </div>
      </div>
    </div>"""
    return html


def make_email_preview(booking):
    """Generate a realistic email confirmation preview"""
    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;background:#f1f5f9;border-radius:10px;
         border:1px solid #e2e8f0;overflow:hidden;">
      <div style="background:#0052cc;padding:10px 16px;display:flex;align-items:center;gap:8px;">
        <span style="color:white;font-size:.78rem;font-weight:700;">📧 EMAIL PREVIEW — Sent to: {booking['email']}</span>
      </div>
      <div style="background:white;margin:12px;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;">
        <div style="background:#f8fafc;padding:10px 16px;border-bottom:1px solid #e2e8f0;">
          <div style="font-size:.72rem;color:#64748b;"><b style="color:#1e293b;">From:</b> bookings@jet2vacations.com</div>
          <div style="font-size:.72rem;color:#64748b;"><b style="color:#1e293b;">To:</b> {booking['email']}</div>
          <div style="font-size:.72rem;color:#64748b;"><b style="color:#1e293b;">Subject:</b> ✈ Booking Confirmed – {booking['ref']} | {booking['destination']}</div>
        </div>
        <div style="padding:16px 20px;font-size:.82rem;color:#1e293b;line-height:1.8;">
          <div style="font-weight:700;font-size:.95rem;color:#0052cc;margin-bottom:8px;">
            Dear {booking.get('traveller','Valued Customer')},
          </div>
          <p>Your Jet2Vacations booking has been <b style="color:#059669;">confirmed</b>! 🎉</p>
          <p>Here is a summary of your trip details:</p>
          <div style="background:#f0f9ff;border-left:4px solid #0052cc;padding:10px 14px;border-radius:0 8px 8px 0;margin:10px 0;">
            <div><b>Booking Reference:</b> {booking['ref']}</div>
            <div><b>Destination:</b> {booking['emoji']} {booking['destination']}, {booking['country']}</div>
            <div><b>Departure:</b> {booking['departure']}</div>
            <div><b>Return:</b> {booking['return_date']}</div>
            <div><b>Duration:</b> {booking['nights']} nights</div>
            <div><b>Travellers:</b> {booking['travelers']}</div>
            <div><b>Total Cost:</b> <span style="color:#0052cc;font-weight:700;">${booking['grand_total']:.0f} USD</span></div>
          </div>
          <p>Your e-tickets and hotel vouchers will be emailed separately within 24 hours.</p>
          <p>For any queries, contact us at <span style="color:#0052cc;">support@jet2vacations.com</span></p>
          <p style="margin-top:12px;">Safe travels!<br><b>Jet2Vacations Team ✈</b></p>
        </div>
      </div>
    </div>"""
    return html


def make_compare_html(dest1, dest2, dests_df):
    """Compare two destinations side by side"""
    d1 = dests_df[dests_df["name"]==dest1]
    d2 = dests_df[dests_df["name"]==dest2]
    if d1.empty or d2.empty:
        return "<div style='padding:20px;color:#94a3b8;'>Select two destinations to compare.</div>"
    r1 = d1.iloc[0]; r2 = d2.iloc[0]
    rows = [
        ("Country",       r1["country"],                      r2["country"]),
        ("Region",        r1["region"],                       r2["region"]),
        ("Category",      r1["category"].title(),             r2["category"].title()),
        ("Avg Cost/Day",  f"${r1['avg_cost_usd']}",           f"${r2['avg_cost_usd']}"),
        ("Culture Score", f"{r1['real_culture_score']:.1f}/10", f"{r2['real_culture_score']:.1f}/10"),
        ("Outdoor Score", f"{r1['real_outdoor_score']:.1f}/10", f"{r2['real_outdoor_score']:.1f}/10"),
        ("Safety Score",  f"{r1['real_safety_score']:.1f}/10", f"{r2['real_safety_score']:.1f}/10"),
        ("Popularity",    f"{r1['real_popularity']:.1f}/10",  f"{r2['real_popularity']:.1f}/10"),
        ("Data Source",   r1["data_source"],                  r2["data_source"]),
    ]
    def score_bar(val, max_val=10, color="#0052cc"):
        try: pct = float(str(val).replace("/10","").replace("$",""))/max_val*100
        except: pct = 50
        return f"<div style='height:4px;background:#e2e8f0;border-radius:2px;margin-top:3px;overflow:hidden;'><div style='width:{min(pct,100):.0f}%;height:100%;background:{color};border-radius:2px;'></div></div>"
    score_fields = {"Culture Score":"#0052cc","Outdoor Score":"#059669","Safety Score":"#f59e0b","Popularity":"#8b5cf6"}
    html = f"""
    <style>
      .cmp-hdr{{background:#0052cc;color:#fff;font-size:.78rem;font-weight:700;padding:10px 14px;text-align:center;}}
      .cmp-row{{display:grid;grid-template-columns:130px 1fr 1fr;gap:0;border-bottom:1px solid #f1f5f9;}}
      .cmp-lbl{{font-size:.73rem;color:#64748b;font-weight:600;padding:9px 12px;background:#f8fafc;display:flex;align-items:center;}}
      .cmp-val{{font-size:.8rem;color:#1e293b;font-weight:500;padding:9px 14px;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;}}
    </style>
    <div style="font-family:'Segoe UI',sans-serif;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
      <div style="display:grid;grid-template-columns:130px 1fr 1fr;">
        <div style="background:#f8fafc;padding:14px 12px;border-bottom:1px solid #e2e8f0;"></div>
        <div class="cmp-hdr" style="border-left:1px solid #1d4ed8;">{r1['emoji']} {dest1}<br><span style="font-weight:400;font-size:.7rem;">{r1['country']}</span></div>
        <div class="cmp-hdr" style="border-left:1px solid #1d4ed8;">{r2['emoji']} {dest2}<br><span style="font-weight:400;font-size:.7rem;">{r2['country']}</span></div>
      </div>"""
    for label, v1, v2 in rows:
        bar1 = score_bar(v1, color=score_fields.get(label,"#0052cc")) if label in score_fields else ""
        bar2 = score_bar(v2, color=score_fields.get(label,"#0052cc")) if label in score_fields else ""
        v1_bold = "font-weight:700;" if label in ("Culture Score","Safety Score","Outdoor Score","Popularity") and v1 >= v2 else ""
        v2_bold = "font-weight:700;" if label in ("Culture Score","Safety Score","Outdoor Score","Popularity") and v2 >= v1 else ""
        try:
            v1_bold = "font-weight:700;color:#0052cc;" if float(str(v1).replace("/10","").replace("$","")) >= float(str(v2).replace("/10","").replace("$","")) and label in score_fields else ""
            v2_bold = "font-weight:700;color:#0052cc;" if float(str(v2).replace("/10","").replace("$","")) >= float(str(v1).replace("/10","").replace("$","")) and label in score_fields else ""
        except: pass
        html += f"""
      <div class="cmp-row">
        <div class="cmp-lbl">{label}</div>
        <div class="cmp-val" style="border-left:1px solid #f1f5f9;{v1_bold}">{v1}{bar1}</div>
        <div class="cmp-val" style="border-left:1px solid #f1f5f9;{v2_bold}">{v2}{bar2}</div>
      </div>"""
    html += "</div>"
    desc1 = str(r1['description'])[:120] + "..."
    desc2 = str(r2['description'])[:120] + "..."
    html += f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;">
      <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:12px;">
        <div style="font-size:.68rem;font-weight:700;color:#0369a1;margin-bottom:4px;">ABOUT {dest1.upper()}</div>
        <div style="font-size:.75rem;color:#1e293b;line-height:1.6;font-style:italic;">{desc1}</div>
      </div>
      <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px;">
        <div style="font-size:.68rem;font-weight:700;color:#166534;margin-bottom:4px;">ABOUT {dest2.upper()}</div>
        <div style="font-size:.75rem;color:#1e293b;line-height:1.6;font-style:italic;">{desc2}</div>
      </div>
    </div>"""
    return html


# =============================================================================
# TELEGRAM NOTIFICATION FUNCTIONS
# =============================================================================

def send_telegram_notification(booking, bot_token, chat_id):
    """
    Send a REAL Telegram message with booking confirmation.
    Uses plain text mode - no special characters, works 100% of the time.
    """
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


def make_telegram_preview(booking, tg_result=""):
    """
    Show Telegram send status + message preview.
    Replaces the old fake email preview.
    """
    if tg_result == "success":
        status_bg   = "#f0fdf4"; status_border = "#bbf7d0"
        status_icon = "✅"; status_color = "#166534"
        status_text = "Telegram notification sent successfully!"
        status_sub  = f"Message delivered to Chat ID on record"
    elif tg_result.startswith("failed"):
        reason = tg_result.replace("failed: ","")
        status_bg   = "#fef2f2"; status_border = "#fecaca"
        status_icon = "❌"; status_color = "#dc2626"
        status_text = "Telegram send failed"
        status_sub  = f"Reason: {reason}"
    elif tg_result == "skipped":
        status_bg   = "#fffbeb"; status_border = "#fde68a"
        status_icon = "⚠"; status_color = "#92400e"
        status_text = "Telegram not configured"
        status_sub  = "Enter Bot Token + Chat ID to receive notifications"
    else:
        status_bg   = "#f8fafc"; status_border = "#e2e8f0"
        status_icon = "ℹ"; status_color = "#64748b"
        status_text = "Notification status unknown"
        status_sub  = ""

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;">
      <div style="background:{status_bg};border:1.5px solid {status_border};
           border-radius:12px;padding:14px 16px;margin-bottom:14px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <span style="font-size:1.3rem;">{status_icon}</span>
          <div>
            <div style="font-size:.88rem;font-weight:700;color:{status_color};">{status_text}</div>
            <div style="font-size:.73rem;color:{status_color};opacity:.8;">{status_sub}</div>
          </div>
        </div>
      </div>

      <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
        <div style="background:#0088cc;padding:10px 14px;display:flex;align-items:center;gap:8px;">
          <span style="font-size:1rem;">✈</span>
          <span style="color:#fff;font-size:.82rem;font-weight:700;">TELEGRAM MESSAGE PREVIEW</span>
        </div>
        <div style="padding:14px 16px;background:#f0f4f8;">
          <div style="background:#fff;border-radius:12px 12px 12px 0;padding:12px 14px;
               max-width:90%;box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:.8rem;
               color:#1e293b;line-height:1.8;">
            <div style="font-weight:700;color:#0088cc;margin-bottom:6px;">✈ Jet2Vacations — Booking Confirmed!</div>
            <div><b>Ref:</b> <code style="background:#f1f5f9;padding:1px 5px;border-radius:3px;">{booking['ref']}</code></div>
            <div><b>Destination:</b> {booking['emoji']} {booking['destination']}, {booking['country']}</div>
            <div><b>Departure:</b> {booking['departure']}  →  <b>Return:</b> {booking['return_date']}</div>
            <div><b>Duration:</b> {booking['nights']} nights · <b>Travellers:</b> {booking['travelers']}</div>
            <div style="margin-top:6px;padding-top:6px;border-top:1px solid #f1f5f9;">
              <b>Flights:</b> ${booking['flight_total']:.0f} &nbsp;|&nbsp;
              <b>Hotel:</b> ${booking['hotel_total']:.0f} &nbsp;|&nbsp;
              <b>Total:</b> <span style="color:#0052cc;font-weight:700;">${booking['grand_total']:.0f} USD</span>
            </div>
            <div style="margin-top:6px;color:#059669;font-weight:600;">✅ Your trip is confirmed. Safe travels!</div>
          </div>
          <div style="font-size:.68rem;color:#94a3b8;margin-top:6px;padding-left:4px;">
            {booking['booked_at']} · Jet2Vacations Bot
          </div>
        </div>
      </div>

      <div style="margin-top:12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
           padding:12px 14px;font-size:.75rem;color:#64748b;line-height:1.8;">
        <div style="font-weight:700;color:#374151;margin-bottom:4px;">Setup Guide</div>
        1. Telegram → search <b>@BotFather</b> → <code>/newbot</code><br>
        2. Copy Bot Token → paste in booking form<br>
        3. Start your bot → open
           <code>api.telegram.org/bot&lt;TOKEN&gt;/getUpdates</code><br>
        4. Copy <b>id</b> field from result → paste as Chat ID
      </div>
    </div>"""
    return html



# ─── Main launch_app function ─────────────────────────────────────────────────

def launch_app():
    try:
        import gradio as gr
    except ImportError:
        print("pip install gradio"); return

    # ── Load data ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✈  Jet2Vacations — AI Travel Management System")
    print("="*60)
    print("\n  Loading real data from Teleport API + Wikipedia...")
    DESTS = fetch_all_real_data(CITY_CATALOGUE, verbose=True)

    print("\n  Generating AI training data (seeded by real scores)...")
    RATINGS = generate_real_seeded_ratings(DESTS, n_users=1000)
    print(f"  {len(RATINGS):,} ratings ready")

    SVD_MODEL = None
    if SURPRISE_OK:
        try:
            from surprise import SVD as _SVD, Dataset as _DS, Reader as _R
            reader = _R(rating_scale=(1.0,5.0))
            data = _DS.load_from_df(RATINGS[["user_id","destination_id","rating"]], reader)
            svd = _SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
            svd.fit(data.build_full_trainset())
            SVD_MODEL = svd
            print("  SVD model trained")
        except Exception as e:
            print(f"  SVD failed: {e}")

    real_count = (DESTS["data_source"]=="Teleport API + Wikipedia").sum()
    DEST_NAMES = sorted(DESTS["name"].tolist())
    BOOKINGS   = []   # session-level booking history

    print(f"\n  Jet2Vacations ready! ({real_count}/40 live data cities)\n")

    INTERESTS  = ["Cultural & History","Adventure","Beach & Islands","Nature & Hiking",
                  "Food & Cuisine","Wellness & Relaxation","Photography",
                  "UNESCO Heritage","Luxury Travel","Budget Travel"]
    CLIMATES   = ["Any","Tropical","Temperate","Mediterranean","Alpine","Arid","Coastal"]
    REGIONS    = ["Anywhere","Europe","Southeast Asia","East Asia","South Asia",
                  "North Africa","South America","Africa","Oceania","Scandinavia",
                  "Middle East","North America","Latin America","East Africa"]
    COMPANIONS = ["Solo","Couple","Friends","Family","Group Tour"]

    # ── CSS ─────────────────────────────────────────────────────────────────
    CSS = """
    * { box-sizing: border-box; }
    body { background: #f0f4f8 !important; }
    .gradio-container {
      background: #f0f4f8 !important;
      max-width: 1280px !important;
      margin: auto !important;
      padding: 0 !important;
      font-family: 'Segoe UI', sans-serif !important;
    }
    footer { display: none !important; }

    /* All text default */
    label { color: #374151 !important; font-size: 0.82rem !important; font-weight: 600 !important; }

    /* Input fields */
    input[type=text], input[type=email], textarea {
      background: #ffffff !important; border: 1.5px solid #d1d5db !important;
      border-radius: 8px !important; color: #1e293b !important;
      font-size: 0.9rem !important; padding: 9px 13px !important;
    }
    input[type=text]:focus, textarea:focus {
      border-color: #0052cc !important;
      box-shadow: 0 0 0 3px rgba(0,82,204,.12) !important;
    }
    input[type=range] { accent-color: #0052cc !important; }

    /* Checkboxes → pills */
    .gradio-checkboxgroup fieldset { border: none !important; padding: 0 !important; }
    .gradio-checkboxgroup .wrap { display: flex !important; flex-wrap: wrap !important; gap: 7px !important; }
    .gradio-checkboxgroup label {
      background: #fff !important; border: 1.5px solid #d1d5db !important;
      border-radius: 999px !important; padding: 5px 14px !important;
      color: #374151 !important; font-size: 0.8rem !important; cursor: pointer !important;
      transition: all 0.15s !important; }
    .gradio-checkboxgroup label:hover { border-color: #0052cc !important; color: #0052cc !important; }
    .gradio-checkboxgroup input[type=checkbox] { display: none !important; }
    .gradio-checkboxgroup label:has(input:checked) {
      background: #0052cc !important; border-color: #0052cc !important;
      color: #fff !important; font-weight: 700 !important; }

    /* Radio → pills */
    .gradio-radio fieldset { border: none !important; padding: 0 !important; }
    .gradio-radio .wrap { display: flex !important; flex-wrap: wrap !important; gap: 7px !important; }
    .gradio-radio label {
      background: #fff !important; border: 1.5px solid #d1d5db !important;
      border-radius: 999px !important; padding: 5px 14px !important;
      color: #374151 !important; font-size: 0.8rem !important; cursor: pointer !important; }
    .gradio-radio label:has(input:checked) {
      background: #eff6ff !important; border-color: #0052cc !important;
      color: #0052cc !important; font-weight: 700 !important; }
    .gradio-radio input[type=radio] { display: none !important; }

    /* Dropdown */
    .gradio-dropdown select, select {
      background: #fff !important; border: 1.5px solid #d1d5db !important;
      border-radius: 8px !important; color: #1e293b !important;
      padding: 8px 12px !important; }

    /* Status textbox */
    .gradio-textbox textarea, .gradio-textbox input {
      background: #f0fdf4 !important; border: 1.5px solid #86efac !important;
      border-radius: 8px !important; color: #166534 !important; font-weight: 600 !important; }

    /* Dataframe */
    .gradio-dataframe table { background: #fff !important; }
    .gradio-dataframe th {
      background: #1e2d4a !important; color: #fff !important;
      font-size: 0.76rem !important; font-weight: 700 !important;
      padding: 10px 12px !important; }
    .gradio-dataframe td {
      color: #1e293b !important; font-size: 0.8rem !important;
      border-color: #f1f5f9 !important; background: #fff !important;
      padding: 8px 12px !important; }
    .gradio-dataframe tr:nth-child(even) td { background: #f8fafc !important; }
    .gradio-dataframe tr:hover td { background: #eff6ff !important; }

    /* Sidebar nav buttons */
    .nav-btn button {
      background: transparent !important; color: #94a3b8 !important;
      border: none !important; border-radius: 6px !important;
      padding: 9px 14px !important; width: 100% !important;
      text-align: left !important; font-size: 0.85rem !important;
      font-weight: 500 !important; cursor: pointer !important;
      transition: all 0.15s !important; }
    .nav-btn button:hover { background: rgba(255,255,255,.1) !important; color: #fff !important; }
    .nav-act button {
      background: rgba(255,255,255,.12) !important; color: #fff !important;
      border-left: 3px solid #0ea5e9 !important; padding-left: 11px !important;
      font-weight: 700 !important; }

    /* Page action buttons */
    .btn-primary button {
      background: #0052cc !important; color: #fff !important;
      border: none !important; border-radius: 8px !important;
      font-size: 0.9rem !important; font-weight: 700 !important;
      padding: 10px 22px !important; cursor: pointer !important;
      transition: all 0.18s !important; width: 100% !important; }
    .btn-primary button:hover { background: #003d99 !important; transform: translateY(-1px) !important; }

    .btn-success button {
      background: #059669 !important; color: #fff !important;
      border: none !important; border-radius: 8px !important;
      font-size: 0.95rem !important; font-weight: 700 !important;
      padding: 11px 24px !important; cursor: pointer !important;
      width: 100% !important; }
    .btn-success button:hover { background: #047857 !important; }

    .btn-outline button {
      background: #fff !important; color: #374151 !important;
      border: 1.5px solid #d1d5db !important; border-radius: 8px !important;
      font-size: 0.85rem !important; font-weight: 600 !important;
      padding: 9px 18px !important; cursor: pointer !important; }
    .btn-outline button:hover { border-color: #0052cc !important; color: #0052cc !important; }

    /* Panel cards */
    .pg-panel { background: #fff !important; border: 1px solid #e2e8f0 !important;
      border-radius: 12px !important; padding: 18px 20px !important; margin-bottom: 14px !important; }
    .pg-section { font-size: 0.68rem !important; font-weight: 800 !important;
      letter-spacing: 0.1em !important; color: #94a3b8 !important;
      text-transform: uppercase !important; margin-bottom: 10px !important; }

    /* Slider info text */
    .gradio-slider .svelte-1oo6mkv { color: #64748b !important; font-size: 0.76rem !important; }
    """

    # ── Progress bar HTML ────────────────────────────────────────────────────
    def prog(step):
        steps_lbl = ["Interests","Budget","Style","Results"]
        dots = ""
        for i,lbl in enumerate(steps_lbl,1):
            if i < step:
                s = "background:#059669;color:#fff;"
                icon = "✓"
            elif i == step:
                s = "background:#0052cc;color:#fff;"
                icon = str(i)
            else:
                s = "background:#e2e8f0;color:#94a3b8;"
                icon = str(i)
            act = f"<div style='font-size:.6rem;color:#0052cc;font-weight:700;margin-top:3px;white-space:nowrap'>{lbl}</div>" if i==step else f"<div style='font-size:.6rem;color:#94a3b8;margin-top:3px;'>{lbl}</div>"
            dots += f"<div style='text-align:center;'><div style='width:26px;height:26px;border-radius:50%;{s}display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:800;margin:auto;'>{icon}</div>{act}</div>"
            if i < 4:
                lc = "#059669" if i < step else "#e2e8f0"
                dots += f"<div style='flex:1;height:2px;background:{lc};margin:13px 0 auto;border-radius:1px;'></div>"
        return f"<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-bottom:14px;'><div style='font-size:.62rem;color:#0052cc;font-weight:800;letter-spacing:.12em;text-align:center;margin-bottom:8px;'>STEP {step} OF 4</div><div style='display:flex;align-items:flex-start;gap:0;padding:0 6px;'>{dots}</div></div>"

    # ── Sidebar HTML ─────────────────────────────────────────────────────────
    def sidebar_html(active="dashboard"):
        nav_items = [
            ("dashboard",    "📊", "Dashboard"),
            ("finder",       "🔍", "Find Destinations"),
            ("booking",      "📋", "Book a Trip"),
            ("bookings",     "🗂", "My Bookings"),
            ("compare",      "⚖",  "Compare Cities"),
        ]
        items_html = ""
        for page_id, icon, label in nav_items:
            is_active = page_id == active
            style = ("background:rgba(255,255,255,.14);color:#fff;border-left:3px solid #0ea5e9;"
                     "padding-left:13px;font-weight:700;" if is_active else "")
            items_html += f"""
            <div style="padding:2px 8px;">
              <div style="display:flex;align-items:center;gap:10px;padding:9px 12px;
                   border-radius:8px;cursor:pointer;{style}color:{'#fff' if is_active else '#94a3b8'};
                   font-size:.85rem;font-weight:{'700' if is_active else '500'};">
                <span style="font-size:.9rem;">{icon}</span>
                <span>{label}</span>
              </div>
            </div>"""
        return f"""
        <div style="background:#1e2d4a;min-height:100vh;padding:0;border-radius:0 0 0 0;">
          <div style="padding:20px 16px 16px;border-bottom:1px solid rgba(255,255,255,.08);">
            <div style="display:flex;align-items:center;gap:10px;">
              <div style="width:36px;height:36px;background:#0ea5e9;border-radius:8px;
                   display:flex;align-items:center;justify-content:center;font-size:1.1rem;">✈</div>
              <div>
                <div style="font-size:.92rem;font-weight:800;color:#fff;letter-spacing:-.3px;">Jet2Vacations</div>
                <div style="font-size:.65rem;color:#64748b;">AI Travel System</div>
              </div>
            </div>
          </div>
          <div style="padding:12px 0;">
            <div style="font-size:.62rem;color:#475569;font-weight:700;letter-spacing:.1em;
                 padding:6px 20px 4px;">NAVIGATION</div>
            {items_html}
          </div>
          <div style="position:absolute;bottom:0;left:0;right:0;padding:14px 16px;
               border-top:1px solid rgba(255,255,255,.06);">
            <div style="font-size:.67rem;color:#475569;text-align:center;line-height:1.6;">
              INT4203 · INTI International University<br>SDG 8 · SDG 11 · Real API Data
            </div>
          </div>
        </div>"""

    # ── Top Header HTML ──────────────────────────────────────────────────────
    def header_html(page_title, subtitle="", active_user="Guest"):
        return f"""
        <div style="background:#fff;border-bottom:1px solid #e2e8f0;
             padding:12px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
          <div>
            <div style="font-size:1rem;font-weight:700;color:#1e293b;">{page_title}</div>
            {f'<div style="font-size:.75rem;color:#64748b;margin-top:1px;">{subtitle}</div>' if subtitle else ''}
          </div>
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span style="background:#ecfdf5;color:#059669;border:1px solid #6ee7b7;
                 font-size:.7rem;font-weight:700;padding:3px 10px;border-radius:999px;">
              📡 Live API
            </span>
            <span style="background:#eff6ff;color:#0052cc;border:1px solid #bfdbfe;
                 font-size:.7rem;font-weight:700;padding:3px 10px;border-radius:999px;">
              🤖 Hybrid AI
            </span>
            <div style="display:flex;align-items:center;gap:7px;padding:5px 12px;
                 background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;">
              <div style="width:28px;height:28px;background:#0052cc;border-radius:50%;
                   display:flex;align-items:center;justify-content:center;
                   color:#fff;font-size:.75rem;font-weight:700;">
                {(active_user[0].upper() if active_user else 'G')}
              </div>
              <span style="font-size:.8rem;font-weight:600;color:#1e293b;">{active_user}</span>
            </div>
          </div>
        </div>"""

    # ── Announcement banner ──────────────────────────────────────────────────
    BANNER = """
    <div style="background:#eef2ff;border-bottom:1px solid #c7d2fe;
         padding:9px 24px;font-size:.78rem;color:#3730a3;text-align:center;">
      ✈ <b>Jet2Vacations</b> — Powered by real Teleport API + Wikipedia data &nbsp;|&nbsp;
      40 global destinations &nbsp;|&nbsp; SVD Collaborative + TF-IDF Content-Based Hybrid AI
    </div>"""

    # ── Dashboard Stats HTML ─────────────────────────────────────────────────
    def make_dashboard_stats(dests_df):
        real = (dests_df["data_source"]=="Teleport API + Wikipedia").sum()
        cards = [
            ("40",       "DESTINATIONS\nAVAILABLE",  "#0052cc", "#eff6ff", "#bfdbfe", "🌍"),
            (str(real),  "LIVE API\nCITIES",         "#059669", "#f0fdf4", "#bbf7d0", "📡"),
            ("2",        "AI MODELS\nACTIVE",         "#7c3aed", "#f5f3ff", "#ddd6fe", "🤖"),
            ("40",       "COUNTRIES\nCOVERED",        "#d97706", "#fffbeb", "#fde68a", "🏴"),
        ]
        html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;">'
        for val, label, color, bg, border, icon in cards:
            lines = label.split("\n")
            html += f"""
            <div style="background:{bg};border:1.5px solid {border};border-radius:12px;
                 padding:16px 18px;">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                <span style="font-size:1.4rem;">{icon}</span>
                <div style="background:{color};color:#fff;font-size:.62rem;font-weight:700;
                     padding:3px 9px;border-radius:999px;letter-spacing:.06em;">LIVE</div>
              </div>
              <div style="font-size:1.9rem;font-weight:800;color:{color};line-height:1;">{val}</div>
              <div style="font-size:.66rem;font-weight:700;color:{color};letter-spacing:.08em;
                   margin-top:4px;line-height:1.5;">{chr(10).join(lines)}</div>
            </div>"""
        html += "</div>"
        # Popular destinations quick preview
        sample = dests_df.nlargest(8, "real_popularity")
        html += '<div style="font-size:.7rem;font-weight:800;color:#374151;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;">TRENDING DESTINATIONS</div>'
        html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">'
        for _,row in sample.iterrows():
            real_b = '<span style="background:#ecfdf5;color:#059669;border:1px solid #6ee7b7;font-size:.6rem;padding:1px 6px;border-radius:4px;font-weight:600;">Live</span>' if row["data_source"]=="Teleport API + Wikipedia" else ''
            html += f"""
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                 padding:12px;cursor:pointer;transition:all .15s;"
                 onmouseover="this.style.borderColor='#0052cc'"
                 onmouseout="this.style.borderColor='#e2e8f0'">
              <div style="font-size:1.3rem;margin-bottom:5px;">{row['emoji']}</div>
              <div style="font-size:.8rem;font-weight:700;color:#1e293b;">{row['name']}</div>
              <div style="font-size:.7rem;color:#64748b;">{row['country']}</div>
              <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px;">
                <span style="font-size:.7rem;color:#059669;font-weight:600;">${row['avg_cost_usd']}/day</span>
                {real_b}
              </div>
              <div style="height:3px;background:#e2e8f0;border-radius:2px;margin-top:8px;overflow:hidden;">
                <div style="width:{row['real_popularity']*10:.0f}%;height:100%;background:#0052cc;border-radius:2px;"></div>
              </div>
            </div>"""
        html += "</div>"
        return html

    # ─── Gradio App ───────────────────────────────────────────────────────────
    selected_dest = {}   # shared state for booking

    with gr.Blocks(css=CSS, title="Jet2Vacations — AI Travel Management", theme=gr.themes.Base()) as demo:

        user_name_state = gr.State("Guest")

        # ── TOP-LEVEL LAYOUT: sidebar + content ──────────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT SIDEBAR ──────────────────────────────────────────────────
            with gr.Column(min_width=200, scale=1):
                sidebar_html_out = gr.HTML(sidebar_html("dashboard"))
                with gr.Column():
                    nav_dash    = gr.Button("📊  Dashboard",        elem_classes="nav-btn")
                    nav_finder  = gr.Button("🔍  Find Destinations", elem_classes="nav-btn")
                    nav_book    = gr.Button("📋  Book a Trip",       elem_classes="nav-btn")
                    nav_mybk    = gr.Button("🗂   My Bookings",       elem_classes="nav-btn")
                    nav_compare = gr.Button("⚖   Compare Cities",    elem_classes="nav-btn")

            # ── MAIN CONTENT AREA ─────────────────────────────────────────────
            with gr.Column(scale=5):
                gr.HTML(BANNER)
                top_header = gr.HTML(header_html("Dashboard", "Welcome to Jet2Vacations AI Travel System"))

                # ══ PAGE: DASHBOARD ═══════════════════════════════════════════
                with gr.Column(visible=True) as pg_dash:
                    gr.HTML(make_dashboard_stats(DESTS))

                # ══ PAGE: FIND DESTINATIONS (4-step wizard) ═══════════════════
                with gr.Column(visible=False) as pg_finder:
                    # Step 1
                    with gr.Column(visible=True) as f_p1:
                        gr.HTML(prog(1))
                        with gr.Group(elem_classes="pg-panel"):
                            gr.HTML('<div class="pg-section">Your Name</div>')
                            name_in = gr.Textbox(placeholder="Enter your name (e.g. Ahmad, Sarah...)", label="", max_lines=1)
                        with gr.Group(elem_classes="pg-panel"):
                            gr.HTML('<div class="pg-section">Travel Interests — select all that apply</div>')
                            interests_in = gr.CheckboxGroup(choices=INTERESTS, value=["Cultural & History"], label="")
                        f_p1_err = gr.HTML("")
                        f_next1  = gr.Button("Continue  →  Budget & Climate", elem_classes="btn-primary")

                    # Step 2
                    with gr.Column(visible=False) as f_p2:
                        gr.HTML(prog(2))
                        with gr.Group(elem_classes="pg-panel"):
                            gr.HTML('<div class="pg-section">Daily Budget (USD)</div>')
                            budget_in = gr.Slider(40, 300, 100, 5, label="Average spend per day",
                                                  info="$40 budget  ──  $140 mid-range  ──  $280+ luxury")
                        with gr.Group(elem_classes="pg-panel"):
                            gr.HTML('<div class="pg-section">Preferred Climate</div>')
                            climate_in = gr.Radio(choices=CLIMATES, value="Any", label="")
                        with gr.Group(elem_classes="pg-panel"):
                            gr.HTML('<div class="pg-section">Preferred Region</div>')
                            region_in = gr.Dropdown(choices=REGIONS, value="Anywhere", label="")
                        with gr.Row():
                            f_back2 = gr.Button("← Back", elem_classes="btn-outline")
                            f_next2 = gr.Button("Continue  →  Travel Style", elem_classes="btn-primary")

                    # Step 3
                    with gr.Column(visible=False) as f_p3:
                        gr.HTML(prog(3))
                        with gr.Group(elem_classes="pg-panel"):
                            gr.HTML('<div class="pg-section">Travel Companion</div>')
                            companion_in = gr.Radio(choices=COMPANIONS, value="Solo", label="")
                        with gr.Group(elem_classes="pg-panel"):
                            gr.HTML('<div class="pg-section">Adventure Level</div>')
                            adventure_in = gr.Slider(0, 100, 50, 10, label="",
                                                     info="0 = Comfort-seeker  ·  50 = Balanced  ·  100 = Thrill-seeker")
                        with gr.Group(elem_classes="pg-panel"):
                            gr.HTML('<div class="pg-section">AI Settings</div>')
                            topk_in  = gr.Slider(3, 15, 10, 1, label="Number of recommendations (Top-K)")
                            alpha_in = gr.Slider(0.0, 1.0, 0.6, 0.1,
                                                 label="Fusion weight α  (0 = pure content · 1 = pure collaborative)")
                        with gr.Row():
                            f_back3  = gr.Button("← Back", elem_classes="btn-outline")
                            f_find   = gr.Button("✦  Find My Destinations", elem_classes="btn-success")

                    # Step 4 - Results
                    with gr.Column(visible=False) as f_p4:
                        gr.HTML(prog(4))
                        f_status = gr.Textbox(label="AI Status", interactive=False, max_lines=1)
                        with gr.Row():
                            with gr.Column(scale=3): f_cards = gr.HTML("")
                            with gr.Column(scale=1, min_width=180):
                                f_stats = gr.HTML("")
                        gr.HTML('<div class="pg-section" style="margin-top:10px;">Full Results Table</div>')
                        f_table = gr.Dataframe(label="", wrap=True, interactive=False)
                        gr.HTML('<div class="pg-section" style="margin-top:10px;">Book a Destination from Results</div>')
                        f_pick_dest = gr.Dropdown(choices=DEST_NAMES, label="Select a destination to book", value=None)
                        f_goto_book = gr.Button("📋  Proceed to Booking →", elem_classes="btn-primary")
                        with gr.Row():
                            f_restart = gr.Button("← New Search", elem_classes="btn-outline")
                            f_refine  = gr.Button("↺ Refine Preferences", elem_classes="btn-outline")

                # ══ PAGE: BOOK A TRIP ═════════════════════════════════════════
                with gr.Column(visible=False) as pg_book:
                    gr.HTML("""
                    <div class="pg-panel">
                      <div class="pg-section">Selected Destination</div>
                      <div id="book-dest-display" style="font-size:.9rem;color:#64748b;">
                        No destination selected — go to Find Destinations first
                      </div>
                    </div>""")
                    bk_dest_display = gr.HTML("")
                    with gr.Group(elem_classes="pg-panel"):
                        gr.HTML('<div class="pg-section">Traveller Details</div>')
                        bk_name   = gr.Textbox(placeholder="Full name", label="Traveller Name")
                        bk_email  = gr.Textbox(placeholder="your@email.com", label="Email Address (for confirmation)")
                    with gr.Group(elem_classes="pg-panel"):
                        gr.HTML('<div class="pg-section">Travel Dates</div>')
                        with gr.Row():
                            bk_dep  = gr.Textbox(placeholder="YYYY-MM-DD  e.g. 2025-06-15", label="Departure Date")
                            bk_ret  = gr.Textbox(placeholder="YYYY-MM-DD  e.g. 2025-06-22", label="Return Date")
                        bk_travelers = gr.Slider(1, 10, 2, 1, label="Number of Travellers")
                    with gr.Group(elem_classes="pg-panel"):
                        gr.HTML('<div class="pg-section">Booking Options</div>')
                        bk_dest_sel = gr.Dropdown(choices=DEST_NAMES, label="Destination", value=None)
                        bk_alpha    = gr.Slider(0.0, 1.0, 0.6, 0.1,
                                                label="Fusion weight α (optional override)")
                    with gr.Group(elem_classes="pg-panel"):
                        gr.HTML("""
                        <div class="pg-section">Telegram Notification</div>
                        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                             padding:10px 13px;margin-bottom:10px;font-size:.77rem;color:#1d4ed8;line-height:1.7;">
                          <b>How to set up (2 minutes):</b><br>
                          1. Open Telegram → search <b>@BotFather</b> → send <code>/newbot</code><br>
                          2. Copy the <b>Bot Token</b> it gives you (e.g. 7234567890:AAF...)<br>
                          3. Start your bot → visit <code>api.telegram.org/bot&lt;TOKEN&gt;/getUpdates</code><br>
                          4. Copy your <b>Chat ID</b> number from the JSON response
                        </div>""")
                        bk_tg_token   = gr.Textbox(
                            placeholder="7234567890:AAFxxxxxxxxxxxxxxxxxxxxxx",
                            label="Telegram Bot Token", max_lines=1, type="password")
                        bk_tg_chat_id = gr.Textbox(
                            placeholder="123456789",
                            label="Your Telegram Chat ID", max_lines=1)
                    bk_err         = gr.HTML("")
                    bk_confirm_btn = gr.Button("✈  Confirm Booking & Send Telegram", elem_classes="btn-success")
                    bk_back        = gr.Button("← Back to Find Destinations", elem_classes="btn-outline")

                # ══ PAGE: BOOKING CONFIRMATION ════════════════════════════════
                with gr.Column(visible=False) as pg_confirm:
                    with gr.Row():
                        with gr.Column(scale=3):
                            conf_card  = gr.HTML("")
                        with gr.Column(scale=2):
                            conf_email = gr.HTML("")
                    with gr.Row():
                        conf_new  = gr.Button("✈ New Search", elem_classes="btn-primary")
                        conf_mybk = gr.Button("🗂 View My Bookings", elem_classes="btn-outline")

                # ══ PAGE: MY BOOKINGS ══════════════════════════════════════════
                with gr.Column(visible=False) as pg_mybk:
                    mybk_status  = gr.HTML("")
                    mybk_table   = gr.Dataframe(label="", wrap=True, interactive=False)
                    mybk_refresh = gr.Button("↺ Refresh", elem_classes="btn-outline")
                    mybk_clear   = gr.Button("🗑 Clear All Bookings", elem_classes="btn-outline")

                # ══ PAGE: COMPARE CITIES ═══════════════════════════════════════
                with gr.Column(visible=False) as pg_compare:
                    with gr.Group(elem_classes="pg-panel"):
                        gr.HTML('<div class="pg-section">Select Two Cities to Compare</div>')
                        with gr.Row():
                            cmp_d1 = gr.Dropdown(choices=DEST_NAMES, value="Paris",    label="City 1")
                            cmp_d2 = gr.Dropdown(choices=DEST_NAMES, value="Bangkok",  label="City 2")
                        cmp_btn = gr.Button("⚖  Compare Now", elem_classes="btn-primary")
                    cmp_out = gr.HTML("")

        # ── Quick Examples ────────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["Ahmad",  ["Cultural & History","Food & Cuisine"]],
                ["Sarah",  ["Beach & Islands","Luxury Travel"]],
                ["James",  ["Adventure","Nature & Hiking"]],
                ["Priya",  ["UNESCO Heritage","Cultural & History"]],
                ["Marcus", ["Budget Travel","Photography"]],
            ],
            inputs=[name_in, interests_in],
            label="Quick Examples")

        # ══════════════════════════════════════════════════════════════════════
        # NAVIGATION LOGIC
        # ══════════════════════════════════════════════════════════════════════
        ALL_PAGES = [pg_dash, pg_finder, pg_book, pg_confirm, pg_mybk, pg_compare]

        def show_only(idx, active_key, uname="Guest"):
            page_updates = [gr.update(visible=(i==idx)) for i in range(len(ALL_PAGES))]
            titles = {
                "dashboard": ("Dashboard", "Welcome to Jet2Vacations AI Travel System"),
                "finder":    ("Find Destinations", "AI-powered destination discovery"),
                "booking":   ("Book a Trip", "Complete your booking in minutes"),
                "confirm":   ("Booking Confirmed", "Your trip is booked!"),
                "bookings":  ("My Bookings", "View and manage your bookings"),
                "compare":   ("Compare Cities", "Side-by-side destination comparison"),
            }
            title, sub = titles.get(active_key, ("Jet2Vacations",""))
            return (*page_updates, sidebar_html(active_key), header_html(title, sub, uname))

        # ── Sidebar navigation ───────────────────────────────────────────────
        def go_dashboard(uname):
            return show_only(0, "dashboard", uname or "Guest")
        def go_finder(uname):
            return show_only(1, "finder", uname or "Guest")
        def go_book(uname):
            return show_only(2, "booking", uname or "Guest")
        def go_mybk(uname):
            return show_only(4, "bookings", uname or "Guest")
        def go_compare(uname):
            return show_only(5, "compare", uname or "Guest")

        NAV_OUTS = ALL_PAGES + [sidebar_html_out, top_header]

        nav_dash.click(go_dashboard,    inputs=[user_name_state], outputs=NAV_OUTS)
        nav_finder.click(go_finder,     inputs=[user_name_state], outputs=NAV_OUTS)
        nav_book.click(go_book,         inputs=[user_name_state], outputs=NAV_OUTS)
        nav_mybk.click(go_mybk,         inputs=[user_name_state], outputs=NAV_OUTS)
        nav_compare.click(go_compare,   inputs=[user_name_state], outputs=NAV_OUTS)

        # ── Finder wizard navigation ─────────────────────────────────────────
        FINDER_PAGES = [f_p1, f_p2, f_p3, f_p4]
        def fp_show(idx):
            return [gr.update(visible=(i==idx)) for i in range(4)]

        def finder_next1(name, interests):
            if not interests:
                err = "<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 12px;font-size:.8rem;color:#dc2626;margin-top:6px;'>⚠ Please select at least one travel interest.</div>"
                return (*fp_show(0), gr.HTML(err), gr.update())
            return (*fp_show(1), gr.HTML(""), gr.update(value=name))

        def finder_p1_to_p2(name, interests):
            ups = finder_next1(name, interests)
            return ups[0], ups[1], ups[2], ups[3], ups[4], ups[5]

        f_next1.click(
            lambda n,i: (*fp_show(0 if not i else 1), gr.HTML("" if i else
                "<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 12px;font-size:.8rem;color:#dc2626;margin-top:6px;'>⚠ Select at least one interest.</div>")),
            inputs=[name_in, interests_in],
            outputs=[f_p1, f_p2, f_p3, f_p4, f_p1_err])

        f_back2.click(lambda: fp_show(0), outputs=[f_p1,f_p2,f_p3,f_p4])
        f_next2.click(lambda: fp_show(2), outputs=[f_p1,f_p2,f_p3,f_p4])
        f_back3.click(lambda: fp_show(1), outputs=[f_p1,f_p2,f_p3,f_p4])

        # ── Run recommendations ──────────────────────────────────────────────
        def run_find(name, interests, budget, climate, region,
                     companion, adventure, topk, alpha):
            top_df, msg, disp_df = get_recommendations(
                name, interests, int(budget), climate, region,
                companion, adventure, DESTS, RATINGS, SVD_MODEL, int(topk), alpha)
            if top_df is None:
                return (*fp_show(0), gr.HTML(""), gr.HTML(""), disp_df, msg, gr.update())
            cards = make_results_cards(top_df, name, int(budget), alpha, DESTS)
            real_c = (DESTS["data_source"]=="Teleport API + Wikipedia").sum()
            avg_cost = top_df["_cost"].mean()
            top_m    = float(top_df.iloc[0]["Match"].replace("%",""))
            regions  = top_df["Country"].nunique()
            stats_html = f"""
            <div style="font-family:'Segoe UI',sans-serif;">
              {''.join([f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;text-align:center;margin-bottom:8px;"><div style="font-size:1.5rem;font-weight:800;color:{c};">{v}</div><div style="font-size:.65rem;color:#94a3b8;font-weight:700;margin-top:2px;">{l}</div></div>'
                for v,l,c in [(f"{top_m:.0f}%","BEST MATCH","#0052cc"),(f"${avg_cost:.0f}","AVG/DAY","#059669"),(str(regions),"COUNTRIES","#7c3aed"),(str(real_c),"LIVE API","#d97706")]])}
              <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px;margin-top:4px;">
                <div style="font-size:.65rem;font-weight:700;color:#1d4ed8;letter-spacing:.08em;margin-bottom:6px;">AI SCORE FORMULA</div>
                <div style="font-size:.66rem;color:#1e40af;line-height:1.8;">
                  S = α·CF + (1-α)·CBF<br>
                  CF: SVD Matrix Factorisation<br>
                  CBF: TF-IDF Cosine Similarity
                </div>
              </div>
            </div>"""
            return (*fp_show(3), cards, stats_html, disp_df, msg, gr.update())

        f_find.click(run_find,
            inputs=[name_in,interests_in,budget_in,climate_in,
                    region_in,companion_in,adventure_in,topk_in,alpha_in],
            outputs=[f_p1,f_p2,f_p3,f_p4,f_cards,f_stats,f_table,f_status,f_pick_dest])

        f_restart.click(lambda: (*fp_show(0),), outputs=[f_p1,f_p2,f_p3,f_p4])
        f_refine.click(lambda: (*fp_show(2),), outputs=[f_p1,f_p2,f_p3,f_p4])

        # ── Go to booking from results ────────────────────────────────────────
        def go_book_with_dest(dest, uname):
            if dest:
                row = DESTS[DESTS["name"]==dest]
                if not row.empty:
                    r = row.iloc[0]
                    disp = f"""
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
                      <div style="font-size:.68rem;color:#1d4ed8;font-weight:700;letter-spacing:.08em;margin-bottom:6px;">SELECTED DESTINATION</div>
                      <div style="font-size:1.05rem;font-weight:700;color:#1e293b;">{r['emoji']} {r['name']}, {r['country']}</div>
                      <div style="font-size:.78rem;color:#64748b;margin-top:3px;">
                        {r['category'].title()} · ~${r['avg_cost_usd']}/day · Culture {r['real_culture_score']:.1f}/10 · Safety {r['real_safety_score']:.1f}/10
                      </div>
                    </div>"""
                    updates = show_only(2, "booking", uname or "Guest")
                    return (*updates, disp, gr.update(value=dest))
            updates = show_only(2, "booking", uname or "Guest")
            return (*updates, gr.HTML(""), gr.update())

        f_goto_book.click(go_book_with_dest,
            inputs=[f_pick_dest, user_name_state],
            outputs=NAV_OUTS + [bk_dest_display, bk_dest_sel])

        # ── Process booking ───────────────────────────────────────────────────
        def process_booking(bk_name, bk_email, dep, ret, travelers,
                             dest_sel, tg_token, tg_chat_id, uname):
            import re
            # Validate
            errs = []
            if not dest_sel: errs.append("Select a destination")
            if not bk_name:  errs.append("Enter traveller name")
            if not bk_email or "@" not in bk_email: errs.append("Enter a valid email")
            if not dep:  errs.append("Enter departure date (YYYY-MM-DD)")
            if not ret:  errs.append("Enter return date (YYYY-MM-DD)")
            if errs:
                err_html = f"<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:10px 14px;font-size:.8rem;color:#dc2626;'>⚠ {' · '.join(errs)}</div>"
                return (*show_only(2,"booking",uname or "Guest"), gr.HTML(err_html), gr.HTML(""), gr.HTML(""))

            # Calculate costs
            try:
                d1 = datetime.datetime.strptime(dep.strip(), "%Y-%m-%d")
                d2 = datetime.datetime.strptime(ret.strip(), "%Y-%m-%d")
                nights = max(1, (d2-d1).days)
            except:
                nights = 7

            dest_row = DESTS[DESTS["name"]==dest_sel]
            cost_day = int(dest_row.iloc[0]["avg_cost_usd"]) if not dest_row.empty else 100
            emoji    = dest_row.iloc[0]["emoji"] if not dest_row.empty else "✈"
            country  = dest_row.iloc[0]["country"] if not dest_row.empty else ""
            category = dest_row.iloc[0]["category"].title() if not dest_row.empty else ""

            flight_total = round(cost_day * 2.2 * int(travelers), 2)
            hotel_total  = round(cost_day * 0.55 * nights * int(travelers), 2)
            grand_total  = round(flight_total + hotel_total, 2)

            ref = "J2V-" + "".join(random.choices(string.ascii_uppercase+string.digits, k=6))
            booking = {
                "ref": ref, "destination": dest_sel, "country": country,
                "emoji": emoji, "category": category,
                "departure": dep.strip(), "return_date": ret.strip(),
                "nights": nights, "travelers": int(travelers),
                "email": bk_email, "traveller": bk_name,
                "flight_total": flight_total, "hotel_total": hotel_total,
                "grand_total": grand_total, "status": "Confirmed",
                "booked_at": datetime.datetime.now().strftime("%d %b %Y %H:%M"),
            }
            BOOKINGS.append(booking)

            # ── Send REAL Telegram message ───────────────────────────────
            tg_result = ""
            if tg_token and tg_chat_id:
                tg_result = send_telegram_notification(booking, tg_token.strip(), tg_chat_id.strip())

            conf_html  = make_booking_confirmation(booking, tg_result)
            email_html = make_telegram_preview(booking, tg_result)
            updates    = show_only(3, "confirm", bk_name or uname or "Guest")
            return (*updates, gr.HTML(""), gr.HTML(conf_html), gr.HTML(email_html))

        bk_confirm_btn.click(process_booking,
            inputs=[bk_name, bk_email, bk_dep, bk_ret, bk_travelers,
                    bk_dest_sel, bk_tg_token, bk_tg_chat_id, user_name_state],
            outputs=NAV_OUTS + [bk_err, conf_card, conf_email])

        bk_back.click(go_finder, inputs=[user_name_state], outputs=NAV_OUTS)

        # ── Confirmation page buttons ─────────────────────────────────────────
        conf_new.click(go_finder, inputs=[user_name_state], outputs=NAV_OUTS)
        conf_mybk.click(go_mybk,  inputs=[user_name_state], outputs=NAV_OUTS)

        # ── My Bookings ───────────────────────────────────────────────────────
        def refresh_bookings(uname):
            if not BOOKINGS:
                status_html = "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;text-align:center;color:#94a3b8;font-size:.85rem;'>No bookings yet — go to Find Destinations and book a trip!</div>"
                return status_html, pd.DataFrame()
            df = pd.DataFrame([{
                "Ref":        b["ref"],
                "Destination":b["destination"],
                "Country":    b["country"],
                "Departure":  b["departure"],
                "Return":     b["return_date"],
                "Nights":     b["nights"],
                "Travellers": b["travelers"],
                "Total (USD)":f"${b['grand_total']:.0f}",
                "Status":     b["status"],
                "Booked At":  b["booked_at"],
                "Email":      b["email"],
            } for b in BOOKINGS])
            status_html = f"""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                 padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
              <span style="font-size:1rem;">✅</span>
              <span style="font-size:.82rem;color:#166534;font-weight:600;">
                {len(BOOKINGS)} booking(s) found · Total spend: ${sum(b['grand_total'] for b in BOOKINGS):.0f} USD
              </span>
            </div>"""
            return status_html, df

        mybk_refresh.click(refresh_bookings, inputs=[user_name_state],
                           outputs=[mybk_status, mybk_table])
        nav_mybk.click(refresh_bookings, inputs=[user_name_state],
                       outputs=[mybk_status, mybk_table])

        def clear_bookings(uname):
            BOOKINGS.clear()
            return "<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:10px 14px;font-size:.82rem;color:#dc2626;'>All bookings cleared.</div>", pd.DataFrame()

        mybk_clear.click(clear_bookings, inputs=[user_name_state],
                         outputs=[mybk_status, mybk_table])

        # ── Compare Cities ────────────────────────────────────────────────────
        def run_compare(d1, d2):
            return make_compare_html(d1, d2, DESTS)

        cmp_btn.click(run_compare, inputs=[cmp_d1, cmp_d2], outputs=[cmp_out])

        # ── Update username state when name entered ───────────────────────────
        name_in.change(lambda n: n if n else "Guest", inputs=[name_in], outputs=[user_name_state])

    import gradio as _gr
    _gr.close_all()
    print("="*60)
    print("  Jet2Vacations launching...")
    print("="*60)
    demo.launch(share=True, show_error=True)


# ── Run the app ──────────────────────────────────────────────────────────
import gradio as gr
gr.close_all()
launch_app()