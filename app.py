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
    page_icon="house",
    layout="wide",
    initial_sidebar_state="expanded"
)

BG_DARK = "#0d1117"
BG_CARD = "#161b22"
BORDER_COLOR = "#30363d"
C_PRIMARY = "#1f77b4"
C_NEUTRAL = "#9CA3AF"
C_SECONDARY = "#F59E0B"
C_ACCENT = "#00d4aa"
TEXT_WHITE = "#FFFFFF"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_WHITE};
    }}
    
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    h1, h2, h3, h4, h5, h6, p, span, label {{
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
        margin: 2rem 0;
    }}
    
    .stButton > button {{
        background-color: {C_PRIMARY} !important;
        color: white !important;
    }}
    
    .stButton > button:hover {{
        background-color: #1557a0 !important;
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

def render_match_box(team1, team2, winner, prob1, prob2):
    code1 = get_flag_code(team1)
    code2 = get_flag_code(team2)
    flag1_url = f"https://flagcdn.com/w80/{code1}.png"
    flag2_url = f"https://flagcdn.com/w80/{code2}.png"
    
    team1_win = "WINNER" if team1 == winner else ""
    team2_win = "WINNER" if team2 == winner else ""
    
    html = f"""
    <div style="background: linear-gradient(135deg, #161b22 0%, #1a2332 100%); border: 1px solid #30363d; border-radius: 6px; padding: 8px; margin: 3px 0; min-width: 160px; font-size: 10px;">
        <div style="display: flex; align-items: center; gap: 5px; padding: 4px 0;">
            <img src="{flag1_url}" style="width: 24px; height: 16px; border-radius: 1px; flex-shrink: 0;" onerror="this.style.display='none'">
            <span style="flex: 1; color: white; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 8px;">{team1}</span>
            <span style="color: #888; font-size: 7px; min-width: 20px; text-align: right;">{prob1:.0%}</span>
            <span style="color: #90EE90; font-size: 6px; font-weight: bold;">{team1_win}</span>
        </div>
        <div style="height: 1px; background: #30363d; margin: 4px 0;"></div>
        <div style="display: flex; align-items: center; gap: 5px; padding: 4px 0;">
            <img src="{flag2_url}" style="width: 24px; height: 16px; border-radius: 1px; flex-shrink: 0;" onerror="this.style.display='none'">
            <span style="flex: 1; color: white; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 8px;">{team2}</span>
            <span style="color: #888; font-size: 7px; min-width: 20px; text-align: right;">{prob2:.0%}</span>
            <span style="color: #90EE90; font-size: 6px; font-weight: bold;">{team2_win}</span>
        </div>
    </div>
    """
    return html

def render_final_box(team1, team2, winner, prob1, prob2):
    code1 = get_flag_code(team1)
    code2 = get_flag_code(team2)
    flag1_url = f"https://flagcdn.com/w80/{code1}.png"
    flag2_url = f"https://flagcdn.com/w80/{code2}.png"
    
    team1_win = "WINNER" if team1 == winner else ""
    team2_win = "WINNER" if team2 == winner else ""
    
    html = f"""
    <div style="background: linear-gradient(135deg, #161b22 0%, #3a3000 100%); border: 2px solid #FFD700; border-radius: 8px; padding: 10px; min-width: 180px; font-size: 11px; box-shadow: 0 4px 12px rgba(255,215,0,0.3); text-align: center;">
        <div style="color: #FFD700; font-weight: bold; margin-bottom: 8px; font-size: 9px; text-transform: uppercase;">FINAL</div>
        <div style="display: flex; align-items: center; gap: 5px; padding: 4px 0;">
            <img src="{flag1_url}" style="width: 24px; height: 16px; border-radius: 1px; flex-shrink: 0;" onerror="this.style.display='none'">
            <span style="flex: 1; color: white; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 8px; text-align: left;">{team1}</span>
            <span style="color: #888; font-size: 7px; min-width: 20px; text-align: right;">{prob1:.0%}</span>
            <span style="color: #90EE90; font-size: 6px; font-weight: bold;">{team1_win}</span>
        </div>
        <div style="height: 1px; background: #FFD700; margin: 5px 0;"></div>
        <div style="display: flex; align-items: center; gap: 5px; padding: 4px 0;">
            <img src="{flag2_url}" style="width: 24px; height: 16px; border-radius: 1px; flex-shrink: 0;" onerror="this.style.display='none'">
            <span style="flex: 1; color: white; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 8px; text-align: left;">{team2}</span>
            <span style="color: #888; font-size: 7px; min-width: 20px; text-align: right;">{prob2:.0%}</span>
            <span style="color: #90EE90; font-size: 6px; font-weight: bold;">{team2_win}</span>
        </div>
    </div>
    """
    return html

def display_knockout_bracket(bracket):
    st.markdown("### Tournament Bracket")
    
    r32_left = bracket.get('left_r32', [])
    r32_right = bracket.get('right_r32', [])
    
    r16_left = bracket.get('left_r16', [])
    r16_right = bracket.get('right_r16', [])
    
    qf_left = bracket.get('left_qf', [])
    qf_right = bracket.get('right_qf', [])
    
    sf_left = bracket.get('left_sf', [])
    sf_right = bracket.get('right_sf', [])
    
    final = bracket.get('final', [])
    
    cols = st.columns([1.5, 1, 1, 0.8, 1, 0.8, 1, 1, 1.5])
    
    with cols[0]:
        st.markdown('<div style="font-size: 9px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; text-transform: uppercase;">R32</div>', unsafe_allow_html=True)
        for match in r32_left:
            html = render_match_box(match['team1'], match['team2'], match['winner'], match['prob1'], match['prob2'])
            st.markdown(html, unsafe_allow_html=True)
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown('<div style="font-size: 9px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; text-transform: uppercase;">R16</div>', unsafe_allow_html=True)
        for i, match in enumerate(r16_left):
            if i > 0:
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
            html = render_match_box(match['team1'], match['team2'], match['winner'], match['prob1'], match['prob2'])
            st.markdown(html, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown('<div style="font-size: 9px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; text-transform: uppercase;">QF</div>', unsafe_allow_html=True)
        for i, match in enumerate(qf_left):
            if i > 0:
                st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
            html = render_match_box(match['team1'], match['team2'], match['winner'], match['prob1'], match['prob2'])
            st.markdown(html, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown('<div style="font-size: 9px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; text-transform: uppercase;">SF</div>', unsafe_allow_html=True)
        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
        for match in sf_left:
            html = render_match_box(match['team1'], match['team2'], match['winner'], match['prob1'], match['prob2'])
            st.markdown(html, unsafe_allow_html=True)
    
    with cols[4]:
        st.markdown('<div style="font-size: 10px; font-weight: bold; color: #FFD700; text-align: center; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">FINAL</div>', unsafe_allow_html=True)
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        if len(final) > 0:
            html = render_final_box(final[0]['team1'], final[0]['team2'], final[0]['winner'], final[0]['prob1'], final[0]['prob2'])
            st.markdown(html, unsafe_allow_html=True)
    
    with cols[5]:
        st.markdown('<div style="font-size: 9px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; text-transform: uppercase;">SF</div>', unsafe_allow_html=True)
        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
        for match in sf_right:
            html = render_match_box(match['team1'], match['team2'], match['winner'], match['prob1'], match['prob2'])
            st.markdown(html, unsafe_allow_html=True)
    
    with cols[6]:
        st.markdown('<div style="font-size: 9px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; text-transform: uppercase;">QF</div>', unsafe_allow_html=True)
        for i, match in enumerate(qf_right):
            if i > 0:
                st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
            html = render_match_box(match['team1'], match['team2'], match['winner'], match['prob1'], match['prob2'])
            st.markdown(html, unsafe_allow_html=True)
    
    with cols[7]:
        st.markdown('<div style="font-size: 9px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; text-transform: uppercase;">R16</div>', unsafe_allow_html=True)
        for i, match in enumerate(r16_right):
            if i > 0:
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
            html = render_match_box(match['team1'], match['team2'], match['winner'], match['prob1'], match['prob2'])
            st.markdown(html, unsafe_allow_html=True)
    
    with cols[8]:
        st.markdown('<div style="font-size: 9px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 8px; text-transform: uppercase;">R32</div>', unsafe_allow_html=True)
        for match in r32_right:
            html = render_match_box(match['team1'], match['team2'], match['winner'], match['prob1'], match['prob2'])
            st.markdown(html, unsafe_allow_html=True)
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

def home_page():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            response = requests.get("https://toppng.com/uploads/preview/fifa-reveals-2026-world-cup-brand-logo-11717899905odpvrys9vo.webp", timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                st.image(img, width=500)
        except Exception as e:
            pass
    
    st.markdown('<div style="font-size: 3rem; font-weight: 800; color: white; text-align: center; margin: 1rem 0; letter-spacing: 2px;">World Cup 2026 Winner Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 1.2rem; color: #1f77b4; text-align: center; margin-bottom: 3rem;">AI-Powered Tournament Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        if st.button("Start Analysis", use_container_width=True, type="primary"):
            st.session_state.page = "eda"
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <div style="background: #161b22; border: 1px solid #1f77b4; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 800; color: #1f77b4;">48</div>
            <div style="font-size: 0.9rem; color: #888; margin-top: 0.3rem;">Teams</div>
        </div>
        <div style="background: #161b22; border: 1px solid #1f77b4; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 800; color: #1f77b4;">12</div>
            <div style="font-size: 0.9rem; color: #888; margin-top: 0.3rem;">Groups</div>
        </div>
        <div style="background: #161b22; border: 1px solid #1f77b4; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 800; color: #1f77b4;">64</div>
            <div style="font-size: 0.9rem; color: #888; margin-top: 0.3rem;">Matches</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown("## Goal Prediction")
    st.markdown("---")
    
    teams = sorted(elo_df['team'].tolist())
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Home Team")
        home = st.selectbox("Select Home Team", teams, key="home_goal", index=0)
        home_elo = elo_df[elo_df['team'] == home]['elo_2026'].values[0]
        st.write(f"Elo: {home_elo:.0f}")
    
    with c2:
        st.markdown("#### Away Team")
        away = st.selectbox("Select Away Team", teams, key="away_goal", index=1)
        away_elo = elo_df[elo_df['team'] == away]['elo_2026'].values[0]
        st.write(f"Elo: {away_elo:.0f}")
    
    st.markdown("---")
    
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        predict_btn = st.button("Predict Score", use_container_width=True, type="primary")
    
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
                st.markdown(f"""
                <div style="background: {BG_CARD}; border-radius: 10px; padding: 20px; text-align: center;">
                    <div style="font-size: 18px; color: #888; margin-bottom: 10px;">{home}</div>
                    <div style="font-size: 3rem; font-weight: 700; color: {C_PRIMARY};">{hg:.1f}</div>
                    <div style="color: #666; margin-top: 10px;">Expected Goals</div>
                </div>
                """, unsafe_allow_html=True)
            
            with r2:
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                    <div style="font-size: 2rem; color: #666;">vs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with r3:
                st.markdown(f"""
                <div style="background: {BG_CARD}; border-radius: 10px; padding: 20px; text-align: center;">
                    <div style="font-size: 18px; color: #888; margin-bottom: 10px;">{away}</div>
                    <div style="font-size: 3rem; font-weight: 700; color: {C_SECONDARY};">{ag:.1f}</div>
                    <div style="color: #666; margin-top: 10px;">Expected Goals</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"Most Likely Score: {home} {round(hg)} - {round(ag)} {away}")

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
            for tier in ['Strong', 'Medium', 'Weak']:
                data = df[df['tier'] == tier]
                st.markdown(f"""
                <div style="background: {BG_CARD}; border-left: 4px solid {C_PRIMARY}; border-radius: 0 8px 8px 0; padding: 1rem; margin-bottom: 1rem;">
                    <div style="font-weight: 700; color: {C_PRIMARY};">{tier} Tier</div>
                    <div style="color: #aaa;">Teams: {len(data)}</div>
                    <div style="color: white; margin-top: 5px;">Avg Elo: {data['elo_2026'].mean():.0f}</div>
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
                    st.markdown(f"""
                    <div style="background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                        <div style="font-weight: 700; color: {C_PRIMARY}; font-size: 1.2rem; margin-bottom: 0.5rem;">Group {group_letter}</div>
                        <div style="color: #aaa; font-size: 0.9rem;">
                            {', '.join(teams)}
                        </div>
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
        
        st.markdown("---")
        st.markdown("### Tournament Champion")
        
        if 'final' in st.session_state.bracket and len(st.session_state.bracket['final']) > 0:
            champion = st.session_state.bracket['final'][0]['winner']
            champion_prob = st.session_state.bracket['final'][0]['prob1'] if st.session_state.bracket['final'][0]['team1'] == champion else st.session_state.bracket['final'][0]['prob2']
            
            flag_code = get_flag_code(champion)
            flag_url = f"https://flagcdn.com/w80/{flag_code}.png"
            
            col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
            with col_c2:
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #161b22 0%, #1a2332 100%); border: 3px solid #1f77b4; border-radius: 12px; padding: 2rem; text-align: center; box-shadow: 0 8px 30px rgba(31,119,180,0.3);">
                    <div style="font-size: 13px; color: #888; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">World Cup Champion</div>
                    <img src="{flag_url}" style="width: 80px; height: 53px; border-radius: 4px; margin-bottom: 15px;" onerror="this.style.display='none'">
                    <div style="font-size: 2rem; font-weight: bold; color: #1f77b4; margin-bottom: 10px;">{champion}</div>
                    <div style="font-size: 13px; color: #1f77b4;">Win Probability: {champion_prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    with st.sidebar:
        st.markdown("## Navigation")
        st.markdown("---")
        
        if st.button("Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        
        if st.button("EDA", use_container_width=True):
            st.session_state.page = "eda"
            st.rerun()
        
        if st.button("Goal Prediction", use_container_width=True):
            st.session_state.page = "goals"
            st.rerun()
        
        if st.button("Clustering", use_container_width=True):
            st.session_state.page = "clustering"
            st.rerun()
        
        if st.button("Simulation", use_container_width=True):
            st.session_state.page = "simulation"
            st.rerun()
    
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'eda':
        eda_page()
    elif st.session_state.page == 'goals':
        goal_prediction_page()
    elif st.session_state.page == 'clustering':
        clustering_page()
    elif st.session_state.page == 'simulation':
        simulation_page()

if __name__ == "__main__":
    main()
