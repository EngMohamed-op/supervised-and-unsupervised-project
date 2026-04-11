"""
World Cup 2026 Prediction — FastAPI Deployment
NOTE: Does NOT modify the original notebook code.
"""
from __future__ import annotations
import json, random, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR     = Path(__file__).parent
MODEL_DIR    = BASE_DIR / "saved_models"
STATIC_DIR   = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

app = FastAPI(title="World Cup 2026 Prediction API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

def _safe_load(path):
    try: return joblib.load(path)
    except: return None

best_classifier = _safe_load(MODEL_DIR / "best_classifier.pkl")
upset_model     = _safe_load(MODEL_DIR / "upset_model.pkl")
home_goal_model = _safe_load(MODEL_DIR / "home_goal_model.pkl")
away_goal_model = _safe_load(MODEL_DIR / "away_goal_model.pkl")

FEATURES, GOAL_FEATURES, UPSET_FEATURES = [], [], []
_art = MODEL_DIR / "artifacts.json"
if _art.exists():
    d = json.loads(_art.read_text())
    FEATURES       = d.get("FEATURES", [])
    GOAL_FEATURES  = d.get("GOAL_FEATURES", [])
    UPSET_FEATURES = d.get("UPSET_FEATURES", [])

ELO_DICT: dict[str,float] = {}
_ELO_AVG = 1500.0
_ep = MODEL_DIR / "elo_snapshot_clustered.csv"
if not _ep.exists():
    _ep = BASE_DIR / "elo_snapshot_2026.csv"
if not _ep.exists():
    _ep = MODEL_DIR / "elo_snapshot_2026.csv"
if _ep.exists():
    _df = pd.read_csv(_ep)
    ELO_DICT = dict(zip(_df["team"], _df["elo_2026"]))
    _ELO_AVG = float(_df["elo_2026"].mean())

MC_RESULTS: list[dict] = []
_mp = MODEL_DIR / "wc2026_win_probabilities.csv"
if not _mp.exists():
    _mp = BASE_DIR / "wc2026_win_probabilities.csv"
if _mp.exists():
    MC_RESULTS = pd.read_csv(_mp).to_dict(orient="records")

FORM_DICT: dict[str,float] = {}
_fp = BASE_DIR / "saved_models" / "form_dict.json"
if not _fp.exists():
    _fp = BASE_DIR / "form_dict.json"
if _fp.exists():
    FORM_DICT = json.loads(_fp.read_text())

# Load training data for EDA
TRAINING_DATA = None
_td = BASE_DIR / "final_training_data_2026.csv"
if _td.exists():
    TRAINING_DATA = pd.read_csv(_td)

WC2026_GROUPS = {
    "A": ["Mexico","South Korea","South Africa","Czech Republic"],
    "B": ["Canada","Bosnia-Herzegovina","Qatar","Switzerland"],
    "C": ["Brazil","Morocco","Haiti","Scotland"],
    "D": ["USA","Paraguay","Australia","Turkey"],
    "E": ["Germany","Curaçao","Côte d'Ivoire","Ecuador"],
    "F": ["Netherlands","Japan","Sweden","Tunisia"],
    "G": ["Belgium","Egypt","Iran","New Zealand"],
    "H": ["Spain","Cabo Verde","Saudi Arabia","Uruguay"],
    "I": ["France","Senegal","Iraq","Norway"],
    "J": ["Argentina","Algeria","Austria","Jordan"],
    "K": ["Portugal","Congo DR","Uzbekistan","Colombia"],
    "L": ["England","Croatia","Ghana","Panama"],
}

FLAGS = {
    "Mexico":"🇲🇽","South Korea":"🇰🇷","South Africa":"🇿🇦","Czech Republic":"🇨🇿",
    "Canada":"🇨🇦","Bosnia-Herzegovina":"🇧🇦","Qatar":"🇶🇦","Switzerland":"🇨🇭",
    "Brazil":"🇧🇷","Morocco":"🇲🇦","Haiti":"🇭🇹","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "USA":"🇺🇸","Paraguay":"🇵🇾","Australia":"🇦🇺","Turkey":"🇹🇷",
    "Germany":"🇩🇪","Curaçao":"🇨🇼","Côte d'Ivoire":"🇨🇮","Ecuador":"🇪🇨",
    "Netherlands":"🇳🇱","Japan":"🇯🇵","Sweden":"🇸🇪","Tunisia":"🇹🇳",
    "Belgium":"🇧🇪","Egypt":"🇪🇬","Iran":"🇮🇷","New Zealand":"🇳🇿",
    "Spain":"🇪🇸","Cabo Verde":"🇨🇻","Saudi Arabia":"🇸🇦","Uruguay":"🇺🇾",
    "France":"🇫🇷","Senegal":"🇸🇳","Iraq":"🇮🇶","Norway":"🇳🇴",
    "Argentina":"🇦🇷","Algeria":"🇩🇿","Austria":"🇦🇹","Jordan":"🇯🇴",
    "Portugal":"🇵🇹","Congo DR":"🇨🇩","Uzbekistan":"🇺🇿","Colombia":"🇨🇴",
    "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Croatia":"🇭🇷","Ghana":"🇬🇭","Panama":"🇵🇦",
}

ALL_WC_TEAMS = sorted({t for ts in WC2026_GROUPS.values() for t in ts})

WORLD_TEAMS_BASE = sorted({
    "Afghanistan","Albania","Algeria","Andorra","Angola","Argentina","Armenia",
    "Australia","Austria","Azerbaijan","Bahrain","Bangladesh","Belarus","Belgium",
    "Benin","Bolivia","Bosnia-Herzegovina","Botswana","Brazil","Bulgaria",
    "Burkina Faso","Cameroon","Canada","Cabo Verde","Chile","China","Colombia",
    "Congo DR","Costa Rica","Croatia","Cuba","Czech Republic","Côte d'Ivoire",
    "Curaçao","Denmark","Ecuador","Egypt","England","Estonia","Ethiopia",
    "Finland","France","Gambia","Georgia","Germany","Ghana","Greece","Guinea",
    "Haiti","Honduras","Hungary","India","Indonesia","Iran","Iraq","Ireland",
    "Israel","Italy","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kosovo",
    "Kuwait","Lebanon","Libya","Lithuania","Luxembourg","Madagascar","Mali",
    "Malta","Mexico","Moldova","Montenegro","Morocco","Mozambique","Namibia",
    "Netherlands","New Zealand","Nigeria","North Korea","North Macedonia","Norway",
    "Oman","Pakistan","Palestine","Panama","Paraguay","Peru","Philippines",
    "Poland","Portugal","Qatar","Romania","Russia","Rwanda","Saudi Arabia",
    "Scotland","Senegal","Serbia","Sierra Leone","Slovakia","Slovenia",
    "South Africa","South Korea","Spain","Sudan","Sweden","Switzerland","Syria",
    "Tanzania","Thailand","Togo","Trinidad and Tobago","Tunisia","Turkey",
    "Uganda","Ukraine","United Arab Emirates","Uruguay","USA","Uzbekistan",
    "Venezuela","Vietnam","Wales","Yemen","Zambia","Zimbabwe",
} | set(ALL_WC_TEAMS))

def get_elo(t): return ELO_DICT.get(t, _ELO_AVG)
def get_form(t): return FORM_DICT.get(t, 0.5)
def get_flag(t): return FLAGS.get(t, "🏳️")

def _row(h, a, neutral=True):
    he,ae = get_elo(h),get_elo(a)
    hf,af = get_form(h),get_form(a)
    return {"home_elo_pre":he,"away_elo_pre":ae,"elo_diff":he-ae,
            "home_form":hf,"away_form":af,"form_diff":hf-af,
            "neutral":int(neutral),"tournament_weight":3.0,
            "neutral_num":int(neutral),"home_advantage":0 if neutral else 1}

def predict_outcome(h, a, neutral=True):
    row = _row(h, a, neutral)
    if best_classifier and FEATURES:
        try:
            X = pd.DataFrame([row])[FEATURES]
            probs = best_classifier.predict_proba(X)[0]
            lm = {0:"Away Win",1:"Draw",2:"Home Win"}
            pm = {lm.get(c,str(c)):float(p) for c,p in zip(best_classifier.classes_,probs)}
            return {**pm,"predicted_result":max(pm,key=pm.get),"source":"ML"}
        except: pass
    diff = row["elo_diff"] + (0 if neutral else 100)
    ph = 1/(1+10**(-diff/400))
    pd_=0.22; ph=ph*(1-pd_); pa=(1-pd_)-ph
    res="Home Win" if ph>pa else ("Draw" if pd_>pa else "Away Win")
    return {"Home Win":round(ph,4),"Draw":round(pd_,4),"Away Win":round(pa,4),
            "predicted_result":res,"source":"Elo fallback"}

def predict_goals(h, a, neutral=True):
    row = _row(h, a, neutral)
    if home_goal_model and away_goal_model and GOAL_FEATURES:
        try:
            X = pd.DataFrame([row])[GOAL_FEATURES]
            hg = float(home_goal_model.predict(X)[0])
            ag = float(away_goal_model.predict(X)[0])
            return {"home_goals":round(max(0,hg),2),"away_goals":round(max(0,ag),2),"source":"ML"}
        except: pass
    diff = row["elo_diff"]; avg=1.35
    return {"home_goals":round(max(0.3,avg*(1+diff/2000)),2),
            "away_goals":round(max(0.3,avg*(1-diff/2000)),2),"source":"Elo fallback"}

def predict_upset(h, a, neutral=True):
    row = _row(h, a, neutral); row["elo_gap"]=abs(row["elo_diff"])
    if upset_model and UPSET_FEATURES:
        try:
            X = pd.DataFrame([row])[UPSET_FEATURES]
            return {"upset_probability":round(float(upset_model.predict_proba(X)[0][1]),4),"source":"ML"}
        except: pass
    return {"upset_probability":round(max(0.05,0.45-row["elo_gap"]/1200),4),"source":"Elo fallback"}

def sim_winner(t1, t2):
    out = predict_outcome(t1, t2, neutral=True)
    gl  = predict_goals(t1, t2, neutral=True)
    r = random.random()
    hw = out.get("Home Win",0.34); dr = out.get("Draw",0.22)
    if r < hw:
        w,l = t1,t2
        g1 = int(max(0, round(gl["home_goals"]+random.uniform(0,0.5))))
        g2 = int(max(0, round(gl["away_goals"]-random.uniform(0,0.3))))
    elif r < hw+dr:
        e1,e2 = get_elo(t1),get_elo(t2)
        w = t1 if random.random()<(e1/(e1+e2)) else t2
        l = t2 if w==t1 else t1
        g1 = g2 = int(round(gl["home_goals"]))
    else:
        w,l = t2,t1
        g2 = int(max(0, round(gl["away_goals"]+random.uniform(0,0.5))))
        g1 = int(max(0, round(gl["home_goals"]-random.uniform(0,0.3))))
    return w, l, g1, g2

def simulate_group_stage():
    standings = {}
    for grp, teams in WC2026_GROUPS.items():
        pts = {t:0 for t in teams}; gd = {t:0 for t in teams}
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                t1,t2 = teams[i],teams[j]
                out = predict_outcome(t1,t2,neutral=True)
                gl  = predict_goals(t1,t2,neutral=True)
                g1 = int(max(0,round(gl["home_goals"]+random.uniform(-0.3,0.3))))
                g2 = int(max(0,round(gl["away_goals"]+random.uniform(-0.3,0.3))))
                r = random.random()
                hw = out.get("Home Win",0.33); dr = out.get("Draw",0.22)
                if r < hw:
                    pts[t1]+=3; gd[t1]+=g1-g2; gd[t2]-=g1-g2
                elif r < hw+dr:
                    pts[t1]+=1; pts[t2]+=1
                else:
                    pts[t2]+=3; gd[t2]+=g2-g1; gd[t1]-=g2-g1
        ranked = sorted(teams, key=lambda t:(pts[t],gd[t],get_elo(t)), reverse=True)
        standings[grp] = {"ranked":ranked, "points":pts, "gd":gd}
    return standings

def get_best_thirds(gs):
    """Return best 8 third-placed teams by points then GD."""
    thirds = [(g, gs[g]["ranked"][2]) for g in WC2026_GROUPS.keys()]
    thirds_sorted = sorted(thirds,
        key=lambda x: (gs[x[0]]["points"][x[1]], gs[x[0]]["gd"][x[1]], get_elo(x[1])),
        reverse=True)
    return [t for _,t in thirds_sorted[:8]]

def simulate_bracket():
    """
    Full WC 2026 bracket following official FIFA 2026 bracket structure.
    R32 matchups based on the official draw format from the image.
    Left half and right half structure preserved.
    """
    random.seed()
    gs = simulate_group_stage()

    # Group results
    def first(g):  return gs[g]["ranked"][0]
    def second(g): return gs[g]["ranked"][1]

    best3 = get_best_thirds(gs)
    # best3[0..7] = best 8 third-placed teams sorted by performance

    # ── Official R32 bracket (from FIFA image) ─────────────────────────
    # LEFT HALF (feeds Left Semi-Final)
    # Match 1: 1E vs 3(ABCDF)
    # Match 2: 1I vs 3(CDFGH)
    # Match 3: 2A vs 2B
    # Match 4: 1F vs 2C
    # Match 5: 2K vs 2L
    # Match 6: 1H vs 2J
    # Match 7: 1D vs 3(BEFIJ)
    # Match 8: 1G vs 3(AEHIJ)

    # RIGHT HALF (feeds Right Semi-Final)
    # Match 9:  1C vs 2F
    # Match 10: 2E vs 2I
    # Match 11: 1A vs 3(GEFH1)
    # Match 12: 1L vs 3(EHIJK)
    # Match 13: 1J vs 2H
    # Match 14: 2D vs 2G
    # Match 15: 1B vs 3(EFGIJ)
    # Match 16: 1K vs 3(EHIJK)

    left_r32 = [
        (first("E"),  best3[0]),   # 1E vs best3
        (first("I"),  best3[1]),   # 1I vs best3
        (second("A"), second("B")),# 2A vs 2B
        (first("F"),  second("C")),# 1F vs 2C
        (second("K"), second("L")),# 2K vs 2L
        (first("H"),  second("J")),# 1H vs 2J
        (first("D"),  best3[2]),   # 1D vs best3
        (first("G"),  best3[3]),   # 1G vs best3
    ]

    right_r32 = [
        (first("C"),  second("F")),# 1C vs 2F
        (second("E"), second("I")),# 2E vs 2I
        (first("A"),  best3[4]),   # 1A vs best3
        (first("L"),  best3[5]),   # 1L vs best3
        (first("J"),  second("H")),# 1J vs 2H
        (second("D"), second("G")),# 2D vs 2G
        (first("B"),  best3[6]),   # 1B vs best3
        (first("K"),  best3[7]),   # 1K vs best3
    ]

    def play_round(pairs, label):
        results = []
        winners = []
        for t1,t2 in pairs:
            w,l,g1,g2 = sim_winner(t1,t2)
            results.append({"t1":t1,"t2":t2,"winner":w,"score":f"{g1}–{g2}",
                            "f1":get_flag(t1),"f2":get_flag(t2),"fw":get_flag(w)})
            winners.append(w)
        return results, winners

    # R32
    l_r32_res, l_r16_in = play_round(left_r32,  "L-R32")
    r_r32_res, r_r16_in = play_round(right_r32, "R-R32")

    # R16 — pair winners sequentially within each half
    left_r16_pairs  = [(l_r16_in[i*2], l_r16_in[i*2+1]) for i in range(4)]
    right_r16_pairs = [(r_r16_in[i*2], r_r16_in[i*2+1]) for i in range(4)]
    l_r16_res, l_qf_in = play_round(left_r16_pairs,  "L-R16")
    r_r16_res, r_qf_in = play_round(right_r16_pairs, "R-R16")

    # QF
    left_qf_pairs  = [(l_qf_in[0], l_qf_in[1]), (l_qf_in[2], l_qf_in[3])]
    right_qf_pairs = [(r_qf_in[0], r_qf_in[1]), (r_qf_in[2], r_qf_in[3])]
    l_qf_res, l_sf_in = play_round(left_qf_pairs,  "L-QF")
    r_qf_res, r_sf_in = play_round(right_qf_pairs, "R-QF")

    # SF
    l_sf_res, l_final_in = play_round([(l_sf_in[0], l_sf_in[1])], "L-SF")
    r_sf_res, r_final_in = play_round([(r_sf_in[0], r_sf_in[1])], "R-SF")

    left_finalist  = l_final_in[0]
    right_finalist = r_final_in[0]
    left_loser     = l_sf_res[0]["t1"] if l_sf_res[0]["winner"] == l_sf_res[0]["t2"] else l_sf_res[0]["t2"]
    right_loser    = r_sf_res[0]["t1"] if r_sf_res[0]["winner"] == r_sf_res[0]["t2"] else r_sf_res[0]["t2"]

    # 3rd place
    w3,_,g1,g2 = sim_winner(left_loser, right_loser)
    third_res = {"t1":left_loser,"t2":right_loser,"winner":w3,"score":f"{g1}–{g2}",
                 "f1":get_flag(left_loser),"f2":get_flag(right_loser),"fw":get_flag(w3)}

    # Final
    champ,_,g1,g2 = sim_winner(left_finalist, right_finalist)
    runner = right_finalist if champ == left_finalist else left_finalist
    final_res = {"t1":left_finalist,"t2":right_finalist,"winner":champ,"score":f"{g1}–{g2}",
                 "f1":get_flag(left_finalist),"f2":get_flag(right_finalist),"fw":get_flag(champ)}

    group_out = {}
    for g in WC2026_GROUPS.keys():
        group_out[g] = [{"team":t,"flag":get_flag(t),"elo":round(get_elo(t),1),
                         "pts":gs[g]["points"][t],"gd":gs[g]["gd"][t]}
                        for t in gs[g]["ranked"]]

    return {
        "groups": group_out,
        "left": {
            "r32": l_r32_res, "r16": l_r16_res,
            "quarterfinals": l_qf_res, "semifinal": l_sf_res,
            "finalist": left_finalist, "finalist_flag": get_flag(left_finalist)
        },
        "right": {
            "r32": r_r32_res, "r16": r_r16_res,
            "quarterfinals": r_qf_res, "semifinal": r_sf_res,
            "finalist": right_finalist, "finalist_flag": get_flag(right_finalist)
        },
        "third_place": third_res,
        "final": final_res,
        "champion":  {"team":champ,  "flag":get_flag(champ)},
        "runner_up": {"team":runner, "flag":get_flag(runner)},
        "third":     {"team":w3,     "flag":get_flag(w3)},
    }

# ── EDA data endpoints ─────────────────────────────────────────────────────
@app.get("/api/eda/outcomes", response_class=JSONResponse)
async def eda_outcomes():
    if TRAINING_DATA is None:
        return {"home_win":48,"draw":22,"away_win":30}
    df = TRAINING_DATA
    total = len(df)
    hw = int((df["result"]==2).sum())
    dr = int((df["result"]==1).sum())
    aw = int((df["result"]==0).sum())
    return {"home_win":round(hw/total*100,1),"draw":round(dr/total*100,1),
            "away_win":round(aw/total*100,1),"total":total}

@app.get("/api/eda/winrate", response_class=JSONResponse)
async def eda_winrate():
    if TRAINING_DATA is None or not ELO_DICT:
        return {"teams":[]}
    df = TRAINING_DATA
    wins_h   = df[df["result"]==2].groupby("home_team").size()
    wins_a   = df[df["result"]==0].groupby("away_team").size()
    played_h = df.groupby("home_team").size()
    played_a = df.groupby("away_team").size()
    total_wins   = wins_h.add(wins_a, fill_value=0)
    total_played = played_h.add(played_a, fill_value=0)
    wr = (total_wins/total_played).sort_values(ascending=False)
    top10 = wr[total_played>=50].head(10).reset_index()
    top10.columns = ["team","win_rate"]
    return {"teams":[{"team":r["team"],"win_rate":round(r["win_rate"],3),
                      "flag":get_flag(r["team"])} for _,r in top10.iterrows()]}

@app.get("/api/eda/elo_dist", response_class=JSONResponse)
async def eda_elo_dist():
    if not ELO_DICT:
        return {"bins":[],"values":[]}
    elos = sorted(ELO_DICT.values())
    import numpy as np
    counts, edges = np.histogram(elos, bins=20)
    return {"bins":[round((edges[i]+edges[i+1])/2,0) for i in range(len(counts))],
            "values":counts.tolist(),
            "mean":round(float(np.mean(elos)),1),
            "median":round(float(np.median(elos)),1)}

@app.get("/api/eda/goals_dist", response_class=JSONResponse)
async def eda_goals_dist():
    if TRAINING_DATA is None:
        return {"home":[],"away":[]}
    df = TRAINING_DATA
    import numpy as np
    h_counts, h_edges = np.histogram(df["home_score"], bins=range(0,11))
    a_counts, a_edges = np.histogram(df["away_score"], bins=range(0,11))
    return {"home":h_counts.tolist(),"away":a_counts.tolist(),
            "labels":list(range(0,10)),
            "home_mean":round(float(df["home_score"].mean()),2),
            "away_mean":round(float(df["away_score"].mean()),2)}

@app.get("/api/eda/elo_vs_outcome", response_class=JSONResponse)
async def eda_elo_vs_outcome():
    if TRAINING_DATA is None:
        return {"data":[]}
    df = TRAINING_DATA.copy()
    if "elo_diff" not in df.columns:
        return {"data":[]}
    sample = df.sample(min(2000, len(df)), random_state=42)
    label_map = {2:"Home Win",1:"Draw",0:"Away Win"}
    data = [{"x":round(float(r["elo_diff"]),1),"y":label_map.get(int(r["result"]),"?")}
            for _,r in sample.iterrows()]
    return {"data":data}

@app.get("/api/eda/mc_top15", response_class=JSONResponse)
async def eda_mc_top15():
    if not MC_RESULTS:
        fb = [{"team":t,"probability":round(1/(1+10**(-(get_elo(t)-_ELO_AVG)/300)),4),"flag":get_flag(t)}
              for t in ALL_WC_TEAMS]
        tot = sum(r["probability"] for r in fb)
        for r in fb: r["probability"]=round(r["probability"]/tot,4)
        results = sorted(fb, key=lambda x:x["probability"], reverse=True)[:15]
    else:
        results = MC_RESULTS[:15]
        for r in results:
            if "flag" not in r:
                r["flag"] = get_flag(r["team"])
    return {"results":results}

# ── Standard routes ────────────────────────────────────────────────────────
class MatchRequest(BaseModel):
    home_team: str; away_team: str; neutral: bool = False

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/teams/wc", response_class=JSONResponse)
async def wc_teams():
    teams = [{"name":t,"elo":round(get_elo(t),1),"form":round(get_form(t),3),
              "flag":get_flag(t),"group":next((g for g,ts in WC2026_GROUPS.items() if t in ts),"?")}
             for t in ALL_WC_TEAMS]
    return {"teams": sorted(teams, key=lambda x:x["elo"], reverse=True)}

@app.get("/api/teams/world", response_class=JSONResponse)
async def world_teams():
    all_t = sorted(set(WORLD_TEAMS_BASE) | (set(ELO_DICT.keys()) if ELO_DICT else set()))
    teams = [{"name":t,"elo":round(get_elo(t),1),"flag":get_flag(t)} for t in all_t]
    return {"teams": sorted(teams, key=lambda x:x["elo"], reverse=True)}

@app.get("/api/groups", response_class=JSONResponse)
async def groups():
    return {"groups":{g:[{"name":t,"elo":round(get_elo(t),1),
                           "form":round(get_form(t),3),"flag":get_flag(t)}
                          for t in ts] for g,ts in WC2026_GROUPS.items()}}

@app.post("/api/predict/outcome", response_class=JSONResponse)
async def api_predict(body: MatchRequest):
    outcome = predict_outcome(body.home_team, body.away_team, body.neutral)
    goals   = predict_goals(body.home_team, body.away_team, body.neutral)
    upset   = predict_upset(body.home_team, body.away_team, body.neutral)
    return {"home_team":body.home_team,"away_team":body.away_team,"neutral":body.neutral,
            "home_elo":round(get_elo(body.home_team),1),"away_elo":round(get_elo(body.away_team),1),
            "elo_diff":round(get_elo(body.home_team)-get_elo(body.away_team),1),
            "home_flag":get_flag(body.home_team),"away_flag":get_flag(body.away_team),
            "outcome":outcome,"goals":goals,"upset":upset}

@app.get("/api/bracket/simulate", response_class=JSONResponse)
async def api_bracket():
    return simulate_bracket()

@app.get("/api/montecarlo", response_class=JSONResponse)
async def mc():
    if not MC_RESULTS:
        fb = [{"team":t,"probability":round(1/(1+10**(-(get_elo(t)-_ELO_AVG)/300)),4),"flag":get_flag(t)}
              for t in ALL_WC_TEAMS]
        tot = sum(r["probability"] for r in fb)
        for r in fb: r["probability"]=round(r["probability"]/tot,4)
        return {"results":sorted(fb,key=lambda x:x["probability"],reverse=True),"source":"Elo fallback"}
    return {"results":MC_RESULTS,"source":"Monte Carlo"}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
