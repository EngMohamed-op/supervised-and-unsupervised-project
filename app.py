import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import requests
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="World Cup 2026 Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BG_DARK      = "#0d1117"
BG_CARD      = "#161b22"
BORDER_COLOR = "#30363d"
C_PRIMARY    = "#2563EB"   # richer blue (matches old version)
C_NEUTRAL    = "#9CA3AF"
C_SECONDARY  = "#F59E0B"
C_ACCENT     = "#00d4aa"
C_DANGER     = "#EF4444"
GOLD         = "#D4AF37"
TEXT_WHITE   = "#FFFFFF"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_WHITE};
    }}

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    #MainMenu {{visibility: hidden;}}
    footer    {{visibility: hidden;}}
    header    {{visibility: hidden;}}

    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {{
        color: {TEXT_WHITE} !important;
    }}

    h1 {{
        border-bottom: 3px solid {C_PRIMARY};
        padding-bottom: 10px;
    }}

    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, {BG_CARD}, {BORDER_COLOR}, {BG_CARD});
        margin: 1.5rem 0;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(145deg, {C_PRIMARY} 0%, #1d4ed8 100%) !important;
        color: {TEXT_WHITE} !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        background: linear-gradient(145deg, #3b82f6 0%, {C_PRIMARY} 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BG_DARK} 0%, {BG_CARD} 100%);
        border-right: 1px solid {BORDER_COLOR};
    }}

    /* ── Selectbox ── */
    .stSelectbox > div > div {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
    }}

    /* ── Metric values ── */
    [data-testid="stMetricValue"] {{
        color: {C_PRIMARY} !important;
        font-size: 2rem !important;
    }}

    /* ── Reusable card classes ── */
    .dashboard-card {{
        background: linear-gradient(145deg, {BG_CARD} 0%, #1a2332 100%);
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .dashboard-card:hover {{
        border-color: {C_PRIMARY};
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(37,99,235,0.2);
    }}
    .card-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {TEXT_WHITE};
        margin-bottom: 0.4rem;
    }}
    .card-desc {{
        font-size: 0.82rem;
        color: #888;
    }}

    .result-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }}

    .winner-card {{
        background: linear-gradient(145deg, {BG_CARD} 0%, #1a2332 100%);
        border: 2px solid {C_PRIMARY};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(37,99,235,0.3);
    }}

    .stat-box {{
        background: {BG_CARD};
        border: 1px solid {C_PRIMARY};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }}
    .stat-number {{
        font-size: 2.5rem;
        font-weight: 800;
        color: {C_PRIMARY};
    }}
    .stat-label {{
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.3rem;
    }}

    .home-title {{
        font-size: 2.8rem;
        font-weight: 800;
        color: {TEXT_WHITE};
        text-align: center;
        margin: 1rem 0;
        letter-spacing: 3px;
    }}
    .home-subtitle {{
        font-size: 1.2rem;
        color: {C_PRIMARY};
        text-align: center;
        margin-bottom: 2rem;
    }}
</style>
""", unsafe_allow_html=True)

WC2026_GROUPS = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czech Republic'],
    'B': ['Canada', 'Bosnia-Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['USA', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', "Côte d'Ivoire", 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cabo Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Congo DR', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

@st.cache_resource
def load_data():
    try:
        elo_df = pd.read_csv("saved_models/elo_snapshot_clustered.csv")
        if elo_df.empty:
            raise ValueError("elo_snapshot_clustered.csv is empty")
        
        with open("saved_models/form_dict.json", 'r') as f:
            form_dict = json.load(f)
        
        classifier = joblib.load("saved_models/best_classifier.pkl")
        
        return elo_df, form_dict, classifier
    
    except FileNotFoundError as e:
        st.error(f"Error: File not found - {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

elo_df, form_dict, classifier = load_data()

team_elo_dict = dict(zip(elo_df['team'], elo_df['elo_2026']))
team_form_dict = form_dict

def get_flag_code(team):
    TEAM_FLAGS = {
        "Argentina": "ar",
        "Australia": "au",
        "Austria": "at",
        "Belgium": "be",
        "Brazil": "br",
        "Cameroon": "cm",
        "Canada": "ca",
        "Colombia": "co",
        "Croatia": "hr",
        "Denmark": "dk",
        "Ecuador": "ec",
        "Egypt": "eg",
        "England": "gb-eng",
        "France": "fr",
        "Germany": "de",
        "Ghana": "gh",
        "Iran": "ir",
        "Italy": "it",
        "Japan": "jp",
        "Mexico": "mx",
        "Morocco": "ma",
        "Netherlands": "nl",
        "New Zealand": "nz",
        "Nigeria": "ng",
        "Norway": "no",
        "Panama": "pa",
        "Paraguay": "py",
        "Poland": "pl",
        "Portugal": "pt",
        "Qatar": "qa",
        "Saudi Arabia": "sa",
        "Senegal": "sn",
        "Serbia": "rs",
        "South Korea": "kr",
        "Spain": "es",
        "Switzerland": "ch",
        "Tunisia": "tn",
        "Turkey": "tr",
        "Uruguay": "uy",
        "USA": "us",
        "Algeria": "dz",
        "Chile": "cl",
        "Scotland": "gb-sct",
        "Wales": "gb-wls",
        "Ukraine": "ua",
        "Sweden": "se",
        "Iraq": "iq",
        "Jordan": "jo",
        "South Africa": "za",
        "Czech Republic": "cz",
        "Bosnia-Herzegovina": "ba",
        "Haiti": "ht",
        "Côte d'Ivoire": "ci",
        "Cabo Verde": "cv",
        "Congo DR": "cd",
        "Uzbekistan": "uz",
        "Curaçao": "cw",
    }
    return TEAM_FLAGS.get(team, "xx")

def get_team_features(team_name):
    elo = team_elo_dict.get(team_name, 800)
    form = team_form_dict.get(team_name, 0.5)
    return {'elo': elo, 'form': form}

def predict_match(team1, team2):
    team1_features = get_team_features(team1)
    team2_features = get_team_features(team2)
    
    X = np.array([[
        team1_features['elo'],
        team2_features['elo'],
        team1_features['form'],
        team2_features['form']
    ]])
    
    try:
        proba = classifier.predict_proba(X)[0]
        proba = np.array(proba, dtype=np.float64)
        proba = proba / proba.sum()
        
        team1_prob = float(proba[0])
        team2_prob = float(proba[1])
        
        temperature = 0.8
        probs = np.array([team1_prob, team2_prob]) ** temperature
        probs = probs / probs.sum()
        
        winner = np.random.choice([team1, team2], p=probs)
        
        return winner, team1_prob, team2_prob
    
    except Exception as e:
        team1_prob = 0.55
        team2_prob = 0.45
        temperature = 0.8
        probs = np.array([team1_prob, team2_prob]) ** temperature
        probs = probs / probs.sum()
        winner = np.random.choice([team1, team2], p=probs)
        return winner, team1_prob, team2_prob

def simulate_group_stage():
    """Simulate group stage matches and return standings"""
    standings = {}
    
    for group_letter, teams in WC2026_GROUPS.items():
        team_points = {team: 0 for team in teams}
        
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team1 = teams[i]
                team2 = teams[j]
                
                winner, prob1, prob2 = predict_match(team1, team2)
                
                if winner == team1:
                    team_points[team1] += 3
                else:
                    team_points[team2] += 3
        
        sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
        
        standings[group_letter] = {
            'first': sorted_teams[0][0],
            'second': sorted_teams[1][0],
            'third': sorted_teams[2][0],
            'third_points': sorted_teams[2][1]
        }
    
    return standings

def get_round_of_32():
    """Get 32 teams for Round of 32"""
    group_standings = simulate_group_stage()
    
    all_first = []
    all_second = []
    all_third = []
    
    for group_letter in sorted(group_standings.keys()):
        standing = group_standings[group_letter]
        all_first.append(standing['first'])
        all_second.append(standing['second'])
        all_third.append((standing['third'], standing['third_points'], group_letter))
    
    all_third.sort(key=lambda x: x[1], reverse=True)
    best_8_third = [team[0] for team in all_third[:8]]
    
    qualified_teams = all_first + all_second + best_8_third
    
    left_side = qualified_teams[:16]
    right_side = qualified_teams[16:32]
    
    return {
        'left': left_side,
        'right': right_side,
        'all': qualified_teams,
        'group_standings': group_standings,
        'best_third': best_8_third,
        'all_third': all_third
    }

def simulate_full_bracket(r32_data):
    """Simulate the full tournament"""
    bracket = {
        'left_r32': [],
        'right_r32': [],
        'left_r16': [],
        'right_r16': [],
        'left_qf': [],
        'right_qf': [],
        'left_sf': [],
        'right_sf': [],
        'final': []
    }
    
    left_teams = r32_data['left']
    right_teams = r32_data['right']
    
    for i in range(0, len(left_teams), 2):
        if i + 1 < len(left_teams):
            team1 = left_teams[i]
            team2 = left_teams[i + 1]
            winner, prob1, prob2 = predict_match(team1, team2)
            
            bracket['left_r32'].append({
                'team1': team1,
                'team2': team2,
                'winner': winner,
                'prob1': prob1,
                'prob2': prob2
            })
    
    for i in range(0, len(right_teams), 2):
        if i + 1 < len(right_teams):
            team1 = right_teams[i]
            team2 = right_teams[i + 1]
            winner, prob1, prob2 = predict_match(team1, team2)
            
            bracket['right_r32'].append({
                'team1': team1,
                'team2': team2,
                'winner': winner,
                'prob1': prob1,
                'prob2': prob2
            })
    
    left_r16_teams = [match['winner'] for match in bracket['left_r32']]
    right_r16_teams = [match['winner'] for match in bracket['right_r32']]
    
    for i in range(0, len(left_r16_teams), 2):
        if i + 1 < len(left_r16_teams):
            team1 = left_r16_teams[i]
            team2 = left_r16_teams[i + 1]
            winner, prob1, prob2 = predict_match(team1, team2)
            
            bracket['left_r16'].append({
                'team1': team1,
                'team2': team2,
                'winner': winner,
                'prob1': prob1,
                'prob2': prob2
            })
    
    for i in range(0, len(right_r16_teams), 2):
        if i + 1 < len(right_r16_teams):
            team1 = right_r16_teams[i]
            team2 = right_r16_teams[i + 1]
            winner, prob1, prob2 = predict_match(team1, team2)
            
            bracket['right_r16'].append({
                'team1': team1,
                'team2': team2,
                'winner': winner,
                'prob1': prob1,
                'prob2': prob2
            })
    
    left_qf_teams = [match['winner'] for match in bracket['left_r16']]
    right_qf_teams = [match['winner'] for match in bracket['right_r16']]
    
    for i in range(0, len(left_qf_teams), 2):
        if i + 1 < len(left_qf_teams):
            team1 = left_qf_teams[i]
            team2 = left_qf_teams[i + 1]
            winner, prob1, prob2 = predict_match(team1, team2)
            
            bracket['left_qf'].append({
                'team1': team1,
                'team2': team2,
                'winner': winner,
                'prob1': prob1,
                'prob2': prob2
            })
    
    for i in range(0, len(right_qf_teams), 2):
        if i + 1 < len(right_qf_teams):
            team1 = right_qf_teams[i]
            team2 = right_qf_teams[i + 1]
            winner, prob1, prob2 = predict_match(team1, team2)
            
            bracket['right_qf'].append({
                'team1': team1,
                'team2': team2,
                'winner': winner,
                'prob1': prob1,
                'prob2': prob2
            })
    
    left_sf_teams = [match['winner'] for match in bracket['left_qf']]
    right_sf_teams = [match['winner'] for match in bracket['right_qf']]
    
    for i in range(0, len(left_sf_teams), 2):
        if i + 1 < len(left_sf_teams):
            team1 = left_sf_teams[i]
            team2 = left_sf_teams[i + 1]
            winner, prob1, prob2 = predict_match(team1, team2)
            
            bracket['left_sf'].append({
                'team1': team1,
                'team2': team2,
                'winner': winner,
                'prob1': prob1,
                'prob2': prob2
            })
    
    for i in range(0, len(right_sf_teams), 2):
        if i + 1 < len(right_sf_teams):
            team1 = right_sf_teams[i]
            team2 = right_sf_teams[i + 1]
            winner, prob1, prob2 = predict_match(team1, team2)
            
            bracket['right_sf'].append({
                'team1': team1,
                'team2': team2,
                'winner': winner,
                'prob1': prob1,
                'prob2': prob2
            })
    
    left_final_team = bracket['left_sf'][0]['winner'] if len(bracket['left_sf']) > 0 else None
    right_final_team = bracket['right_sf'][0]['winner'] if len(bracket['right_sf']) > 0 else None
    
    if left_final_team and right_final_team and left_final_team != right_final_team:
        winner, prob1, prob2 = predict_match(left_final_team, right_final_team)
        
        bracket['final'].append({
            'team1': left_final_team,
            'team2': right_final_team,
            'winner': winner,
            'prob1': prob1,
            'prob2': prob2
        })
    
    return bracket

def _match_card_html(team1, team2, winner, prob1, prob2, is_final=False):
    """Build a single match card as an HTML string — no Streamlit calls."""
    code1 = get_flag_code(team1)
    code2 = get_flag_code(team2)
    f1 = f"https://flagcdn.com/w40/{code1}.png"
    f2 = f"https://flagcdn.com/w40/{code2}.png"

    border  = "2px solid #FFD700" if is_final else "1px solid #30363d"
    bg      = "linear-gradient(135deg,#161b22 0%,#3a3000 100%)" if is_final else "linear-gradient(135deg,#161b22 0%,#1a2332 100%)"
    divider = "#FFD700" if is_final else "#30363d"

    def row(team, flag_url, prob, is_winner):
        bold   = "700" if is_winner else "400"
        color  = "#FFFFFF" if is_winner else "#aaaaaa"
        badge  = '<span style="background:#22863a;color:#fff;border-radius:3px;padding:1px 4px;font-size:9px;margin-left:4px;">✓</span>' if is_winner else ""
        return f"""
        <div style="display:flex;align-items:center;gap:6px;padding:5px 0;">
          <img src="{flag_url}" style="width:26px;height:17px;border-radius:2px;flex-shrink:0;object-fit:cover;"
               onerror="this.style.display='none'">
          <span style="flex:1;color:{color};font-weight:{bold};font-size:11px;
                       white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{team}{badge}</span>
          <span style="color:#888;font-size:10px;flex-shrink:0;">{prob:.0%}</span>
        </div>"""

    return f"""
    <div style="background:{bg};border:{border};border-radius:8px;
                padding:8px 10px;margin:0;box-sizing:border-box;width:100%;">
      {row(team1, f1, prob1, team1==winner)}
      <div style="height:1px;background:{divider};margin:2px 0;"></div>
      {row(team2, f2, prob2, team2==winner)}
    </div>"""


def _round_column_html(label, matches, spacer_top_px=0, gap_between_px=8, is_final=False):
    """Wrap a list of match cards into a vertically-spaced column div."""
    cards = ""
    for i, m in enumerate(matches):
        if i > 0:
            cards += f'<div style="height:{gap_between_px}px;"></div>'
        cards += _match_card_html(m['team1'], m['team2'], m['winner'],
                                  m['prob1'], m['prob2'], is_final=is_final)
    label_color = "#FFD700" if is_final else "#1f77b4"
    return f"""
    <div style="display:flex;flex-direction:column;align-items:stretch;">
      <div style="font-size:10px;font-weight:700;color:{label_color};text-align:center;
                  text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">{label}</div>
      <div style="height:{spacer_top_px}px;"></div>
      {cards}
    </div>"""


def display_knockout_bracket(bracket):
    st.markdown("### Tournament Bracket")

    r32_l  = bracket.get('left_r32',  [])
    r32_r  = bracket.get('right_r32', [])
    r16_l  = bracket.get('left_r16',  [])
    r16_r  = bracket.get('right_r16', [])
    qf_l   = bracket.get('left_qf',   [])
    qf_r   = bracket.get('right_qf',  [])
    sf_l   = bracket.get('left_sf',   [])
    sf_r   = bracket.get('right_sf',  [])
    final  = bracket.get('final',     [])

    # Each round needs progressively more top-padding so cards visually
    # sit at the midpoint of the pairs they emerged from.
    # Card height ≈ 68 px,  gap between cards in R32 ≈ 8 px
    CARD_H = 68
    GAP32  = 8

    def top_pad(round_idx):
        """Pixels to push the first card down so it centres between its R32 pair."""
        # round_idx: 0=R32, 1=R16, 2=QF, 3=SF, 4=Final
        if round_idx == 0:
            return 0
        pairs = 2 ** round_idx          # how many R32 cards per 1 card at this round
        block = pairs * CARD_H + (pairs - 1) * GAP32   # height of that block
        return (block - CARD_H) // 2

    def gap_between(round_idx):
        """Gap between cards at this round so they stay centred."""
        pairs = 2 ** round_idx
        block = pairs * CARD_H + (pairs - 1) * GAP32
        return block - CARD_H

    col_defs = [
        ("R32",   r32_l,  0,                    GAP32,                False),
        ("R16",   r16_l,  top_pad(1),            gap_between(1),       False),
        ("QF",    qf_l,   top_pad(2),            gap_between(2),       False),
        ("SF",    sf_l,   top_pad(3),            gap_between(3),       False),
        ("FINAL", final,  top_pad(4),            0,                    True),
        ("SF",    sf_r,   top_pad(3),            gap_between(3),       False),
        ("QF",    qf_r,   top_pad(2),            gap_between(2),       False),
        ("R16",   r16_r,  top_pad(1),            gap_between(1),       False),
        ("R32",   r32_r,  0,                     GAP32,                False),
    ]

    # Build all 9 columns as HTML and join them inside a single flex container
    columns_html = ""
    for label, matches, sp_top, gap, is_fin in col_defs:
        col_html = _round_column_html(label, matches, sp_top, gap, is_fin)
        flex = "1.6" if label == "R32" else ("1.2" if label in ("R16",) else "1")
        columns_html += f'<div style="flex:{flex};min-width:0;">{col_html}</div>'

    full_html = f"""
    <div style="display:flex;gap:6px;align-items:flex-start;
                overflow-x:auto;padding:8px 0;">
      {columns_html}
    </div>"""

    st.markdown(full_html, unsafe_allow_html=True)

def home_page():
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Logo with emoji fallback
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            response = requests.get(
                "https://toppng.com/uploads/preview/fifa-reveals-2026-world-cup-brand-logo-11717899905odpvrys9vo.webp",
                timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                st.image(img, width=500)
            else:
                st.markdown('<div style="text-align:center;font-size:5rem;">⚽🏆</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div style="text-align:center;font-size:5rem;">⚽🏆</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h1 class="home-title">WORLD CUP WINNER PREDICTION</h1>', unsafe_allow_html=True)
    st.markdown('<p class="home-subtitle">AI-Powered Football Analytics • Elo Ratings • Monte Carlo Simulation</p>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        if st.button("⚽ START ANALYSIS", use_container_width=True):
            st.session_state.page = "eda"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;">
        <div class="stat-box"><div class="stat-number">48</div><div class="stat-label">Teams</div></div>
        <div class="stat-box"><div class="stat-number">12</div><div class="stat-label">Groups</div></div>
        <div class="stat-box"><div class="stat-number">64</div><div class="stat-label">Matches</div></div>
        <div class="stat-box"><div class="stat-number">2026</div><div class="stat-label">Year</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:#666;font-size:0.9rem;">United States • Canada • Mexico</div>',
                unsafe_allow_html=True)

def eda_page():
    st.markdown("## Exploratory Data Analysis")
    st.markdown("---")
    
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': BG_DARK,
        'axes.facecolor': BG_CARD,
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
    })
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Top 15 Teams by Elo")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        top15 = elo_df.nlargest(15, 'elo_2026').sort_values('elo_2026')
        colors = [C_PRIMARY if e > 900 else C_ACCENT if e > 800 else C_NEUTRAL 
                  for e in top15['elo_2026']]
        
        bars = ax.barh(top15['team'], top15['elo_2026'], color=colors, edgecolor='white')
        
        for bar, elo in zip(bars, top15['elo_2026']):
            ax.text(elo + 5, bar.get_y() + bar.get_height()/2, 
                   f'{elo:.0f}', va='center', color='white', fontsize=9)
        
        ax.set_xlabel('Elo Rating', color='white')
        ax.set_title('Team Rankings', color='white', pad=10)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with c2:
        st.markdown("### Elo Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        ax.hist(elo_df['elo_2026'], bins=20, alpha=0.8, 
               color=C_ACCENT, edgecolor='white')
        
        ax.axvline(elo_df['elo_2026'].mean(), color=C_SECONDARY, linestyle='--', 
                  linewidth=2, label=f'Mean: {elo_df["elo_2026"].mean():.0f}')
        
        ax.set_xlabel('Elo Rating', color='white')
        ax.set_ylabel('Teams', color='white')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_COLOR, labelcolor='white')
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown("### Win Rate Distribution")
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        if 'win_rate' in elo_df.columns:
            ax.hist(elo_df['win_rate'], bins=15, alpha=0.8, 
                   color=C_ACCENT, edgecolor='white')
            ax.set_xlabel('Win Rate', color='white')
            ax.set_ylabel('Teams', color='white')
        else:
            ax.text(0.5, 0.5, 'Win rate data not available', 
                   ha='center', va='center', color='white', transform=ax.transAxes)
        
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with c4:
        st.markdown("### Form Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        form_values = list(form_dict.values())
        ax.hist(form_values, bins=15, alpha=0.8, 
               color=C_PRIMARY, edgecolor='white')
        
        ax.axvline(np.mean(form_values), color=C_SECONDARY, linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(form_values):.2f}')
        
        ax.set_xlabel('Form Score', color='white')
        ax.set_ylabel('Teams', color='white')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_COLOR, labelcolor='white')
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    c5, c6 = st.columns(2)
    
    with c5:
        st.markdown("### Elo vs Form Correlation")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        form_data = []
        for team in elo_df['team']:
            form_data.append(form_dict.get(team, 0.5))
        
        ax.scatter(elo_df['elo_2026'], form_data, alpha=0.6, 
                  color=C_PRIMARY, edgecolor='white', s=100)
        
        z = np.polyfit(elo_df['elo_2026'], form_data, 1)
        p = np.poly1d(z)
        ax.plot(elo_df['elo_2026'].sort_values(), p(elo_df['elo_2026'].sort_values()), 
               "r--", alpha=0.8, linewidth=2, color=C_SECONDARY)
        
        ax.set_xlabel('Elo Rating', color='white')
        ax.set_ylabel('Form Score', color='white')
        ax.set_title('Elo vs Form', color='white', pad=10)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with c6:
        st.markdown("### Team Tiers")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        if 'tier' in elo_df.columns:
            tier_counts = elo_df['tier'].value_counts()
            colors_tier = {tier: C_PRIMARY if tier == 'Strong' else C_ACCENT if tier == 'Medium' else C_NEUTRAL 
                          for tier in tier_counts.index}
            bars = ax.bar(tier_counts.index, tier_counts.values, 
                         color=[colors_tier.get(t, C_NEUTRAL) for t in tier_counts.index],
                         edgecolor='white')
            
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, 
                       f'{int(h)}', ha='center', color='white', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Tier data not available', 
                   ha='center', va='center', color='white', transform=ax.transAxes)
        
        ax.set_ylabel('Number of Teams', color='white')
        ax.set_title('Teams by Tier', color='white', pad=10)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def goal_prediction_page():
    st.markdown("## ⚽ Goal Prediction")
    st.markdown("---")
    
    teams = sorted(elo_df['team'].tolist())
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🏠 Home Team")
        home = st.selectbox("Select Home Team", teams, key="home_goal", index=0)
        home_elo = elo_df[elo_df['team'] == home]['elo_2026'].values[0]
        flag_code = get_flag_code(home)
        st.image(f"https://flagcdn.com/w80/{flag_code}.png", width=55)
        st.markdown(f"**Elo:** {home_elo:.0f}")
    
    with c2:
        st.markdown("#### ✈️ Away Team")
        away = st.selectbox("Select Away Team", teams, key="away_goal", index=1)
        away_elo = elo_df[elo_df['team'] == away]['elo_2026'].values[0]
        flag_code_a = get_flag_code(away)
        st.image(f"https://flagcdn.com/w80/{flag_code_a}.png", width=55)
        st.markdown(f"**Elo:** {away_elo:.0f}")
    
    st.markdown("---")
    
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        predict_btn = st.button("🎯 Predict Score", use_container_width=True)
    
    if predict_btn:
        if home == away:
            st.error("Please select different teams!")
        else:
            diff_ratio = (home_elo - away_elo) / 400
            hg = max(0, 1.5 + diff_ratio * 0.5)
            ag = max(0, 1.2 - diff_ratio * 0.5)
            
            st.markdown("---")
            st.markdown("### Expected Scoreline")
            
            r1, r2, r3 = st.columns([2, 1, 2])
            
            with r1:
                st.markdown(f"""<div class="result-card">
                    <img src="https://flagcdn.com/w80/{get_flag_code(home)}.png" width="60"
                         style="border-radius:4px;margin-bottom:10px;">
                    <div style="color:#888;margin-bottom:10px;">{home}</div>
                    <div style="font-size:3rem;font-weight:700;color:{C_PRIMARY};">{hg:.1f}</div>
                    <div style="color:#666;margin-top:8px;">Expected Goals</div>
                </div>""", unsafe_allow_html=True)
            
            with r2:
                st.markdown("""<div style="display:flex;align-items:center;
                            justify-content:center;height:100%;padding-top:60px;">
                    <div style="font-size:2rem;color:#666;">vs</div>
                </div>""", unsafe_allow_html=True)
            
            with r3:
                st.markdown(f"""<div class="result-card">
                    <img src="https://flagcdn.com/w80/{get_flag_code(away)}.png" width="60"
                         style="border-radius:4px;margin-bottom:10px;">
                    <div style="color:#888;margin-bottom:10px;">{away}</div>
                    <div style="font-size:3rem;font-weight:700;color:{C_SECONDARY};">{ag:.1f}</div>
                    <div style="color:#666;margin-top:8px;">Expected Goals</div>
                </div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Most Likely Score:** {home} {round(hg)} - {round(ag)} {away}")

def clustering_page():
    st.markdown("## Team Clustering Analysis")
    st.markdown("---")
    
    df = elo_df.copy()
    
    form_data = []
    for team in df['team']:
        form_data.append(form_dict.get(team, 0.5))
    
    df['form'] = form_data
    
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': BG_DARK,
        'axes.facecolor': BG_CARD,
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
    })
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### Elo vs Form")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        if 'tier' in df.columns:
            tier_colors = {'Strong': C_PRIMARY, 'Medium': C_ACCENT, 'Weak': C_NEUTRAL}
            for tier in df['tier'].unique():
                data = df[df['tier'] == tier]
                ax.scatter(data['elo_2026'], data['form'], 
                          label=tier, alpha=0.7, s=100, 
                          color=tier_colors.get(tier, C_NEUTRAL), edgecolor='white')
            ax.legend(facecolor=BG_CARD, edgecolor=BORDER_COLOR, labelcolor='white')
        else:
            ax.scatter(df['elo_2026'], df['form'], alpha=0.7, s=100, 
                      color=C_PRIMARY, edgecolor='white')
        
        top_teams = df.nlargest(5, 'elo_2026')
        for _, row in top_teams.iterrows():
            ax.text(row['elo_2026'] + 10, row['form'] + 0.01, 
                   row['team'], fontsize=9, color='white', fontweight='bold')
        
        ax.set_xlabel('Elo Rating', color='white')
        ax.set_ylabel('Form Score', color='white')
        ax.set_title('Team Clustering', color='white', pad=10)
        ax.grid(True, linestyle='--', alpha=0.15, color='white')
        
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with c2:
        st.markdown("### Cluster Summary")
        if 'tier' in df.columns:
            tier_colors_map = {'Strong': C_PRIMARY, 'Medium': C_ACCENT, 'Weak': C_NEUTRAL}
            for tier in ['Strong', 'Medium', 'Weak']:
                data = df[df['tier'] == tier]
                color = tier_colors_map[tier]
                st.markdown(f"""
                <div style="background:{color}20;border-left:4px solid {color};
                            border-radius:0 8px 8px 0;padding:1.2rem;margin-bottom:1rem;">
                    <div style="font-weight:700;color:{color};font-size:1.05rem;">{tier} Tier</div>
                    <div style="color:#aaa;">{len(data)} Teams</div>
                    <div style="color:white;margin-top:5px;">Avg Elo: {data['elo_2026'].mean():.0f}</div>
                </div>
                """, unsafe_allow_html=True)

def simulation_page():
    st.markdown("## World Cup 2026 Simulation")
    st.markdown("---")
    
    st.markdown("### Group Stage Teams")
    
    cols_per_row = 3
    groups_list = list(WC2026_GROUPS.items())
    
    for i in range(0, len(groups_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(groups_list):
                group_letter, teams = groups_list[i + j]
                with col:
                    team_rows_html = ""
                    for team in teams:
                        flag_code = get_flag_code(team)
                        flag_url  = f"https://flagcdn.com/w40/{flag_code}.png"
                        team_rows_html += f"""
                        <div style="display:flex;align-items:center;gap:8px;padding:4px 0;">
                            <img src="{flag_url}" style="width:24px;height:16px;border-radius:2px;
                                 object-fit:cover;flex-shrink:0;" onerror="this.style.display='none'">
                            <span style="color:#ddd;font-size:0.85rem;">{team}</span>
                        </div>"""
                    st.markdown(f"""
                    <div style="background:{BG_CARD};border:1px solid {BORDER_COLOR};border-radius:8px;
                                padding:1rem;margin-bottom:1rem;">
                        <div style="font-weight:700;color:{C_PRIMARY};font-size:1.1rem;
                                    margin-bottom:0.6rem;border-bottom:1px solid {BORDER_COLOR};
                                    padding-bottom:0.4rem;">Group {group_letter}</div>
                        {team_rows_html}
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.sidebar.markdown("## Simulation Controls")
    st.sidebar.markdown("---")
    
    if st.button("Simulate Tournament", use_container_width=True, type="primary"):
        with st.spinner("Simulating tournament..."):
            r32_data = get_round_of_32()
            bracket = simulate_full_bracket(r32_data)
        
        st.session_state.bracket = bracket
        st.session_state.r32_data = r32_data
    
    if 'bracket' in st.session_state:
        st.markdown("---")
        st.markdown("### Group Stage Results")
        
        group_results = st.session_state.r32_data['group_standings']
        cols_results = 4
        groups_results_list = list(group_results.items())
        
        for i in range(0, len(groups_results_list), cols_results):
            cols = st.columns(cols_results)
            for j, col in enumerate(cols):
                if i + j < len(groups_results_list):
                    group_letter, result = groups_results_list[i + j]
                    with col:
                        st.markdown(f"""
                        <div style="background: {BG_CARD}; border: 1px solid {C_PRIMARY}; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.5rem; font-size: 0.85rem;">
                            <div style="font-weight: 700; color: {C_PRIMARY}; margin-bottom: 0.5rem;">Group {group_letter}</div>
                            <div style="color: #00ff00; margin-bottom: 0.3rem;">🥇 {result['first']}</div>
                            <div style="color: #ffa500; margin-bottom: 0.3rem;">🥈 {result['second']}</div>
                            <div style="color: #cd7f32; font-size: 0.8rem;">🥉 {result['third']}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Best 3rd Place Qualifiers")
        
        best_third = st.session_state.r32_data['best_third']
        col_third1, col_third2, col_third3, col_third4 = st.columns(4)
        cols_third = [col_third1, col_third2, col_third3, col_third4]
        
        for idx, team in enumerate(best_third):
            with cols_third[idx % 4]:
                st.markdown(f"""
                <div style="background: {BG_CARD}; border: 1px solid {C_ACCENT}; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.5rem; text-align: center;">
                    <div style="color: {C_ACCENT}; font-weight: 700;">{team}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        display_knockout_bracket(st.session_state.bracket)

        # Explicit container ensures champion card renders after the HTML bracket
        champion_container = st.container()
        with champion_container:
            st.markdown("---")
            st.markdown("### 🏆 Tournament Champion")

            if 'final' in st.session_state.bracket and len(st.session_state.bracket['final']) > 0:
                champion = st.session_state.bracket['final'][0]['winner']
                champion_prob = (st.session_state.bracket['final'][0]['prob1']
                                 if st.session_state.bracket['final'][0]['team1'] == champion
                                 else st.session_state.bracket['final'][0]['prob2'])

                flag_url = f"https://flagcdn.com/w80/{get_flag_code(champion)}.png"

                col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
                with col_c2:
                    st.markdown(f"""<div class="winner-card">
                        <div style="color:#888;font-size:0.85rem;text-transform:uppercase;
                                    letter-spacing:1px;margin-bottom:12px;">🏆 World Cup Champion</div>
                        <img src="{flag_url}" width="90"
                             style="border-radius:6px;margin-bottom:15px;"
                             onerror="this.style.display='none'">
                        <div style="font-size:2rem;font-weight:700;color:white;
                                    margin-bottom:8px;">{champion}</div>
                        <div style="color:{C_PRIMARY};font-size:0.95rem;">
                            Win Probability: {champion_prob:.1%}</div>
                    </div>""", unsafe_allow_html=True)

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Sidebar always visible — but content changes on home vs other pages
    with st.sidebar:
        if st.session_state.page == 'home':
            st.markdown("### ⚽ World Cup 2026")
            st.markdown("---")
            st.markdown('<div style="color:#888;font-size:0.85rem;">Use the Start Analysis button to begin.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown("### 🧭 Navigation")
            st.markdown("---")

            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()

            st.markdown("---")

            pages = [
                ("📊", "EDA",             "eda"),
                ("🎯", "Goal Prediction", "goals"),
                ("🔵", "Clustering",      "clustering"),
                ("🏆", "Simulation",      "simulation"),
            ]
            for emoji, name, key in pages:
                is_active = st.session_state.page == key
                label = f"**{emoji} {name}**" if is_active else f"{emoji} {name}"
                if st.button(label, use_container_width=True):
                    st.session_state.page = key
                    st.rerun()

    # Back button at top of every non-home page
    if st.session_state.page != 'home':
        if st.button("← Back to Home", key="back_btn"):
            st.session_state.page = "home"
            st.rerun()

    page_map = {
        'home':       home_page,
        'eda':        eda_page,
        'goals':      goal_prediction_page,
        'clustering': clustering_page,
        'simulation': simulation_page,
    }
    page_map[st.session_state.page]()

if __name__ == "__main__":
    main()
