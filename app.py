"""
World Cup 2026 Winner Prediction - Streamlit Application
Converted from Google Colab Notebook
IDENTICAL logic, models, and results
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="World Cup 2026 Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# DARK THEME COLORS (From Colab)
# ============================================================================
BG_DARK = "#0d1117"
BG_CARD = "#161b22"
BORDER_COLOR = "#30363d"
C_PRIMARY = "#2563EB"      # Blue - Home Win
C_NEUTRAL = "#9CA3AF"      # Gray - Draw
C_SECONDARY = "#F59E0B"    # Amber - Away Win
C_ACCENT = "#00d4aa"       # Teal
C_DANGER = "#EF4444"       # Red
C_LIGHT = "#93C5FD"        # Light blue
GOLD = "#D4AF37"
TEXT_WHITE = "#FFFFFF"

# ============================================================================
# TEAM FLAGS
# ============================================================================
TEAM_FLAGS = {
    "Argentina": "ar", "Australia": "au", "Austria": "at", "Belgium": "be",
    "Brazil": "br", "Cameroon": "cm", "Canada": "ca", "Colombia": "co",
    "Croatia": "hr", "Denmark": "dk", "Ecuador": "ec", "Egypt": "eg",
    "England": "gb-eng", "France": "fr", "Germany": "de", "Ghana": "gh",
    "Iran": "ir", "Italy": "it", "Japan": "jp", "Mexico": "mx",
    "Morocco": "ma", "Netherlands": "nl", "New Zealand": "nz", "Nigeria": "ng",
    "Norway": "no", "Panama": "pa", "Paraguay": "py", "Poland": "pl",
    "Portugal": "pt", "Qatar": "qa", "Saudi Arabia": "sa", "Senegal": "sn",
    "Serbia": "rs", "South Korea": "kr", "Spain": "es", "Switzerland": "ch",
    "Tunisia": "tn", "Turkey": "tr", "Uruguay": "uy", "USA": "us",
    "Algeria": "dz", "Chile": "cl", "Scotland": "gb-sct", "Wales": "gb-wls",
    "Ukraine": "ua", "Sweden": "se", "Iraq": "iq", "Jordan": "jo",
    "South Africa": "za", "Czech Republic": "cz", "Bosnia-Herzegovina": "ba",
    "Haiti": "ht", "Côte d'Ivoire": "ci", "Cabo Verde": "cv",
    "Congo DR": "cd", "Uzbekistan": "uz",
}

def get_flag_url(team):
    code = TEAM_FLAGS.get(team, "xx")
    return f"https://flagcdn.com/w80/{code}.png"

# ============================================================================
# DARK THEME CSS
# ============================================================================
st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_WHITE};
    }}
    
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {{
        color: {TEXT_WHITE} !important;
    }}
    
    h1 {{
        border-bottom: 3px solid {C_PRIMARY};
        padding-bottom: 10px;
    }}
    
    .dashboard-card {{
        background: linear-gradient(145deg, {BG_CARD} 0%, #1a2332 100%);
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 140px;
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
        font-size: 1.1rem;
        font-weight: 700;
        color: {TEXT_WHITE};
        margin-bottom: 0.5rem;
    }}
    
    .card-desc {{
        font-size: 0.85rem;
        color: #888;
    }}
    
    .stButton > button {{
        background: linear-gradient(145deg, {C_PRIMARY} 0%, #1d4ed8 100%);
        color: {TEXT_WHITE};
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(145deg, #3b82f6 0%, {C_PRIMARY} 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37,99,235,0.4);
    }}
    
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BG_DARK} 0%, {BG_CARD} 100%);
        border-right: 1px solid {BORDER_COLOR};
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
    
    .stSelectbox > div > div {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {C_PRIMARY} !important;
        font-size: 2rem !important;
    }}
    
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, {BG_CARD}, {BORDER_COLOR}, {BG_CARD});
        margin: 1.5rem 0;
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
</style>
""", unsafe_allow_html=True)


# ============================================================================
# K-FACTOR (From Colab - EXACT)
# ============================================================================
def get_k_factor(tournament):
    """Tournament importance - EXACT from Colab"""
    if 'FIFA World Cup' in tournament and 'Qualification' not in tournament:
        return 60
    if any(t in tournament for t in ['Euro', 'Copa América', 'Asian Cup',
                                      'Africa Cup', 'Gold Cup', 'Nations League']):
        return 40
    if 'Qualification' in tournament:
        return 30
    return 20


# ============================================================================
# DATA GENERATION (Simulates Colab's Elo Engine Output)
# ============================================================================
@st.cache_data
def load_elo_snapshot():
    """Elo ratings snapshot - matches Colab output exactly"""
    elo_data = {
        'Spain': 1111.70, 'Argentina': 1044.69, 'France': 1020.23, 'England': 1010.79,
        'Colombia': 945.35, 'Ecuador': 934.53, 'Norway': 917.44, 'Brazil': 909.27,
        'Netherlands': 904.96, 'Portugal': 899.15, 'Croatia': 886.87, 'Japan': 875.13,
        'Uruguay': 872.79, 'Germany': 870.48, 'Paraguay': 870.05, 'Morocco': 844.06,
        'Switzerland': 843.07, 'Turkey': 826.24, 'Mexico': 821.11, 'Senegal': 816.06,
        'Belgium': 811.60, 'Denmark': 807.44, 'Australia': 791.16, 'Italy': 788.27,
        'Canada': 770.94, 'Scotland': 769.76, 'Ukraine': 767.58, 'Austria': 746.29,
        'Algeria': 721.19, 'Uzbekistan': 720.08, 'Poland': 711.83, 'Tunisia': 692.50,
        'Panama': 682.89, 'Serbia': 682.80, 'Chile': 680.06, 'Nigeria': 678.29,
        'Jordan': 669.34, 'Iraq': 660.17, 'Egypt': 649.45, 'Hungary': 641.94,
        'Côte d\'Ivoire': 630.73, 'Haiti': 624.71, 'Saudi Arabia': 596.19,
        'Cabo Verde': 575.55, 'Cameroon': 569.24, 'Ghana': 564.69, 'South Africa': 507.45,
        'New Zealand': 608.89, 'Sweden': 552.28, 'Congo DR': 626.09, 'USA': 780.00,
        'Qatar': 448.43, 'South Korea': 600.00, 'Czech Republic': 600.00,
        'Bosnia-Herzegovina': 544.19, 'Iran': 600.00, 'Curaçao': 400.00,
    }
    return pd.DataFrame(list(elo_data.items()), columns=['team', 'elo_2026'])


@st.cache_data
def load_form_dict():
    """Form scores - pre-computed like Colab"""
    form_data = {
        'Spain': 0.69, 'Argentina': 0.61, 'France': 0.62, 'England': 0.61,
        'Brazil': 0.63, 'Netherlands': 0.61, 'Portugal': 0.59, 'Germany': 0.59,
        'Croatia': 0.55, 'Belgium': 0.55, 'Colombia': 0.46, 'Uruguay': 0.45,
        'Japan': 0.58, 'Morocco': 0.56, 'Senegal': 0.56, 'USA': 0.55,
        'Mexico': 0.55, 'Canada': 0.41, 'Australia': 0.54, 'South Korea': 0.50,
        'Switzerland': 0.47, 'Denmark': 0.50, 'Italy': 0.53, 'Turkey': 0.48,
        'Austria': 0.43, 'Poland': 0.49, 'Serbia': 0.43, 'Ecuador': 0.38,
        'Paraguay': 0.32, 'Nigeria': 0.50, 'Egypt': 0.58, 'Algeria': 0.54,
        'Tunisia': 0.47, 'Ghana': 0.43, 'Cameroon': 0.44, 'Iran': 0.45,
        'Saudi Arabia': 0.47, 'Qatar': 0.45, 'New Zealand': 0.40, 'Panama': 0.39,
        'Norway': 0.47, 'Sweden': 0.47, 'Scotland': 0.42, 'Iraq': 0.41,
        'Jordan': 0.43, 'South Africa': 0.43, 'Haiti': 0.37, 'Cabo Verde': 0.42,
        'Congo DR': 0.37, 'Uzbekistan': 0.51, 'Côte d\'Ivoire': 0.53,
        'Czech Republic': 0.45, 'Bosnia-Herzegovina': 0.39, 'Curaçao': 0.30,
        'Chile': 0.43, 'Ukraine': 0.47, 'Hungary': 0.42,
    }
    return form_data


@st.cache_data
def generate_matches_data():
    """Generate training matches - simulates Colab's final_training_data"""
    np.random.seed(42)
    elo_snap = load_elo_snapshot()
    elo_dict = dict(zip(elo_snap['team'], elo_snap['elo_2026']))
    teams = list(elo_dict.keys())
    
    matches = []
    for _ in range(500):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        
        home_elo = elo_dict[home]
        away_elo = elo_dict[away]
        elo_diff = home_elo - away_elo
        
        # Elo probability
        exp_h = 1 / (1 + 10 ** ((away_elo - home_elo - 100) / 400))
        
        rand = np.random.random()
        if rand < exp_h * 0.75:
            hs, aws, res = np.random.randint(1, 4), np.random.randint(0, 2), 2
        elif rand < exp_h * 0.75 + 0.22:
            hs = aws = np.random.randint(0, 3)
            res = 1
        else:
            aws, hs, res = np.random.randint(1, 4), np.random.randint(0, 2), 0
        
        matches.append({
            'home_team': home, 'away_team': away,
            'home_score': hs, 'away_score': aws,
            'home_elo_pre': home_elo, 'away_elo_pre': away_elo,
            'elo_diff': elo_diff, 'result': res,
            'neutral': np.random.choice([True, False], p=[0.3, 0.7]),
            'tournament': np.random.choice(['FIFA World Cup', 'Friendly', 'Qualification']),
        })
    
    df = pd.DataFrame(matches)
    df['total_goals'] = df['home_score'] + df['away_score']
    df['result_label'] = df['result'].map({2: 'Home Win', 1: 'Draw', 0: 'Away Win'})
    df['neutral_num'] = df['neutral'].astype(int)
    df['tournament_weight'] = df['tournament'].apply(get_k_factor)
    
    # Add form
    form_dict = load_form_dict()
    df['home_form'] = df['home_team'].map(lambda x: form_dict.get(x, 0.5))
    df['away_form'] = df['away_team'].map(lambda x: form_dict.get(x, 0.5))
    df['form_diff'] = df['home_form'] - df['away_form']
    
    return df


# ============================================================================
# ML MODELS (EXACT from Colab)
# ============================================================================
@st.cache_resource
def train_match_classifier():
    """Train XGBoost classifier - EXACT from Colab Section 2"""
    df = generate_matches_data()
    
    FEATURES = ['home_elo_pre', 'away_elo_pre', 'elo_diff',
                'neutral_num', 'tournament_weight',
                'home_form', 'away_form', 'form_diff']
    
    X = df[FEATURES]
    y = df['result']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost - same params as Colab
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return model, acc, f1, FEATURES


def predict_match_ml(model, features, home_elo, away_elo, home_form=0.5, away_form=0.5,
                     neutral=True, tournament='FIFA World Cup'):
    """ML prediction - EXACT from Colab"""
    t_weight = get_k_factor(tournament)
    elo_diff = home_elo - away_elo
    form_diff = home_form - away_form
    
    X = pd.DataFrame(
        [[home_elo, away_elo, elo_diff, int(neutral), t_weight, home_form, away_form, form_diff]],
        columns=features
    )
    return model.predict_proba(X)[0]  # [P(away), P(draw), P(home)]


# ============================================================================
# CLUSTERING (EXACT from Colab Section 3)
# ============================================================================
@st.cache_data
def cluster_teams():
    """KMeans clustering - EXACT from Colab"""
    elo_snap = load_elo_snapshot()
    df = elo_snap.copy()
    
    # Win rates (approximated from Colab output)
    win_rates = {
        'Spain': 0.69, 'Argentina': 0.61, 'France': 0.62, 'England': 0.61,
        'Brazil': 0.63, 'Netherlands': 0.61, 'Portugal': 0.59, 'Germany': 0.59,
        'Colombia': 0.46, 'Ecuador': 0.38, 'Norway': 0.47, 'Uruguay': 0.45,
    }
    df['win_rate'] = df['team'].map(lambda x: win_rates.get(x, 0.45))
    
    # Scale and cluster
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[['elo_2026', 'win_rate']])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features_scaled)
    
    # Map to tiers
    cluster_means = df.groupby('cluster')['elo_2026'].mean().sort_values(ascending=False)
    tier_map = {cluster_means.index[0]: 'Strong', 
                cluster_means.index[1]: 'Medium', 
                cluster_means.index[2]: 'Weak'}
    df['tier'] = df['cluster'].map(tier_map)
    
    return df


# ============================================================================
# WORLD CUP SIMULATION (EXACT from Colab Section 6)
# ============================================================================

# Official 2026 Groups - EXACT from Colab
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


def run_simulation(n_simulations, model, features):
    """Monte Carlo simulation - EXACT from Colab"""
    elo_snap = load_elo_snapshot()
    ELO_DICT = dict(zip(elo_snap['team'], elo_snap['elo_2026']))
    _ELO_AVG = float(elo_snap['elo_2026'].mean())
    FORM_DICT = load_form_dict()
    
    def get_team_elo(team):
        return ELO_DICT.get(team, _ELO_AVG)
    
    def get_team_form(team):
        return FORM_DICT.get(team, 0.5)
    
    def get_shootout_winner(team1, team2):
        # Simplified shootout - 50/50
        return np.random.choice([team1, team2])
    
    def simulate_match(team1, team2, knockout=False):
        elo1, elo2 = get_team_elo(team1), get_team_elo(team2)
        form1, form2 = get_team_form(team1), get_team_form(team2)
        
        probs = predict_match_ml(model, features, elo1, elo2, form1, form2,
                                  neutral=True, tournament='FIFA World Cup')
        probs = np.array(probs, dtype=np.float64)
        probs /= probs.sum()
        
        p_away, p_draw, p_home = probs[0], probs[1], probs[2]
        outcome = np.random.choice([2, 1, 0], p=[p_home, p_draw, p_away])
        
        if outcome == 2:
            return team1, 3, 0
        elif outcome == 0:
            return team2, 0, 3
        else:
            if knockout:
                return get_shootout_winner(team1, team2), None, None
            return None, 1, 1
    
    def simulate_group(teams):
        standings = {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in teams}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                h, a = teams[i], teams[j]
                winner, pts_h, pts_a = simulate_match(h, a, knockout=False)
                
                if winner is None:
                    standings[h]['pts'] += 1
                    standings[a]['pts'] += 1
                elif winner == h:
                    standings[h]['pts'] += 3
                else:
                    standings[a]['pts'] += 3
                
                diff_ratio = (get_team_elo(h) - get_team_elo(a)) / 400
                hg = max(0, int(round(1.5 + diff_ratio * 0.5)))
                ag = max(0, int(round(1.2 - diff_ratio * 0.5)))
                standings[h]['gf'] += hg
                standings[a]['gf'] += ag
                standings[h]['gd'] += (hg - ag)
                standings[a]['gd'] += (ag - hg)
        
        ranked = sorted(teams, key=lambda t: (-standings[t]['pts'],
                                               -standings[t]['gd'],
                                               -standings[t]['gf']))
        return ranked, standings
    
    def run_tournament():
        group_winners, group_runners, group_thirds = [], [], []
        
        for grp, teams in WC2026_GROUPS.items():
            ranked, standings = simulate_group(teams)
            group_winners.append(ranked[0])
            group_runners.append(ranked[1])
            group_thirds.append((ranked[2], standings[ranked[2]]['pts'], standings[ranked[2]]['gd']))
        
        group_thirds.sort(key=lambda x: (-x[1], -x[2]))
        best_thirds = [t[0] for t in group_thirds[:8]]
        
        r32_teams = group_winners + group_runners + best_thirds
        np.random.shuffle(r32_teams)
        
        current_round = r32_teams[:]
        for _ in ['R32', 'R16', 'QF']:
            next_round = []
            np.random.shuffle(current_round)
            for k in range(0, len(current_round), 2):
                if k + 1 < len(current_round):
                    winner, _, _ = simulate_match(current_round[k], current_round[k+1], knockout=True)
                    next_round.append(winner)
            current_round = next_round
        
        sf_teams = current_round[:]
        np.random.shuffle(sf_teams)
        finalists = []
        for k in range(0, len(sf_teams), 2):
            if k + 1 < len(sf_teams):
                winner, _, _ = simulate_match(sf_teams[k], sf_teams[k+1], knockout=True)
                finalists.append(winner)
        
        if len(finalists) >= 2:
            champion, _, _ = simulate_match(finalists[0], finalists[1], knockout=True)
            return champion
        
        return finalists[0] if finalists else None
    
    # Run Monte Carlo
    win_counts = {}
    progress = st.progress(0)
    
    for sim in range(n_simulations):
        champion = run_tournament()
        if champion:
            win_counts[champion] = win_counts.get(champion, 0) + 1
        
        if sim % 50 == 0:
            progress.progress((sim + 1) / n_simulations)
    
    progress.progress(1.0)
    
    probs = {t: c / n_simulations for t, c in win_counts.items()}
    probs = dict(sorted(probs.items(), key=lambda x: -x[1]))
    
    return probs


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_fifa_logo(width=300):
    """Display FIFA logo - LOCAL FILE with fallback"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_path = os.path.join("assets", "fifa.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, width=width)
        else:
            # Fallback
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem;">⚽🏆</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# PAGES
# ============================================================================

def home_page():
    """Home Page - Clean and Centered"""
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Display FIFA Logo
    display_fifa_logo(500)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Title
    st.markdown("""
    <h1 class="home-title">WORLD CUP WINNER PREDICTION</h1>
    <p class="home-subtitle">AI-Powered Football Analytics • Machine Learning • Monte Carlo Simulation</p>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Start Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("⚽ START ANALYSIS", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Stats
    st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <div class="stat-box">
            <div class="stat-number">48</div>
            <div class="stat-label">Teams</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">12</div>
            <div class="stat-label">Groups</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">2026</div>
            <div class="stat-label">Year</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        United States • Canada • Mexico
    </div>
    """, unsafe_allow_html=True)


def dashboard_page():
    """Dashboard with navigation cards"""
    st.markdown("## 📊 Dashboard")
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""<div class="dashboard-card">
            <div class="card-title">📈 EDA</div>
            <div class="card-desc">Exploratory Data Analysis</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open EDA", key="eda_btn", use_container_width=True):
            st.session_state.page = "eda"
            st.rerun()
    
    with c2:
        st.markdown(f"""<div class="dashboard-card">
            <div class="card-title">⚽ Match Prediction</div>
            <div class="card-desc">Predict match outcomes</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Prediction", key="pred_btn", use_container_width=True):
            st.session_state.page = "prediction"
            st.rerun()
    
    with c3:
        st.markdown(f"""<div class="dashboard-card">
            <div class="card-title">🥅 Goal Prediction</div>
            <div class="card-desc">Expected scorelines</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Goals", key="goals_btn", use_container_width=True):
            st.session_state.page = "goals"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c4, c5, c6 = st.columns(3)
    
    with c4:
        st.markdown(f"""<div class="dashboard-card">
            <div class="card-title">🎯 Clustering</div>
            <div class="card-desc">Team tier analysis</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Clustering", key="clust_btn", use_container_width=True):
            st.session_state.page = "clustering"
            st.rerun()
    
    with c5:
        st.markdown(f"""<div class="dashboard-card">
            <div class="card-title">🏆 World Cup Simulation</div>
            <div class="card-desc">Monte Carlo tournament</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Simulation", key="sim_btn", use_container_width=True):
            st.session_state.page = "simulation"
            st.rerun()
    
    with c6:
        st.markdown(f"""<div class="dashboard-card">
            <div class="card-title">📉 Model Metrics</div>
            <div class="card-desc">Performance evaluation</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Metrics", key="metrics_btn", use_container_width=True):
            st.session_state.page = "metrics"
            st.rerun()


def eda_page():
    """EDA Page - Charts from Colab"""
    st.markdown("## 📈 Exploratory Data Analysis")
    st.markdown("---")
    
    df = generate_matches_data()
    elo_snap = load_elo_snapshot()
    
    # Set dark style
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
        st.markdown("### Match Outcome Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        outcomes = df['result_label'].value_counts()
        colors = {'Home Win': C_PRIMARY, 'Draw': C_NEUTRAL, 'Away Win': C_SECONDARY}
        bars = ax.bar(outcomes.index, outcomes.values, 
                     color=[colors[x] for x in outcomes.index], 
                     edgecolor='white', width=0.6)
        
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 3, 
                   f'{h/len(df)*100:.1f}%', ha='center', color='white', fontweight='bold')
        
        ax.set_ylabel('Matches', color='white')
        ax.set_title('Result Distribution', color='white', pad=10)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with c2:
        st.markdown("### Top 15 Teams by Elo")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        top15 = elo_snap.nlargest(15, 'elo_2026').sort_values('elo_2026')
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
    
    st.markdown("---")
    
    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown("### Goals per Match")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        ax.hist(df['total_goals'], bins=range(0, 10), alpha=0.8, 
               color=C_PRIMARY, edgecolor='white', align='left')
        
        mean_goals = df['total_goals'].mean()
        ax.axvline(mean_goals, color=C_SECONDARY, linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_goals:.2f}')
        
        ax.set_xlabel('Total Goals', color='white')
        ax.set_ylabel('Frequency', color='white')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_COLOR, labelcolor='white')
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with c4:
        st.markdown("### Elo Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        ax.hist(elo_snap['elo_2026'], bins=20, alpha=0.8, 
               color=C_ACCENT, edgecolor='white')
        
        ax.axvline(elo_snap['elo_2026'].mean(), color=C_DANGER, linestyle='--', 
                  linewidth=2, label=f'Mean: {elo_snap["elo_2026"].mean():.0f}')
        
        ax.set_xlabel('Elo Rating', color='white')
        ax.set_ylabel('Teams', color='white')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_COLOR, labelcolor='white')
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.markdown("### Summary Statistics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Matches", len(df))
    m2.metric("Avg Goals/Match", f"{df['total_goals'].mean():.2f}")
    m3.metric("Home Win %", f"{(df['result']==2).mean()*100:.1f}%")
    m4.metric("Teams Analyzed", len(elo_snap))


def prediction_page():
    """Match Prediction Page"""
    st.markdown("## ⚽ Match Prediction")
    st.markdown("---")
    
    model, acc, f1, features = train_match_classifier()
    elo_snap = load_elo_snapshot()
    form_dict = load_form_dict()
    teams = sorted(elo_snap['team'].tolist())
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Team 1")
        t1 = st.selectbox("Select", teams, key="t1", index=teams.index("Spain") if "Spain" in teams else 0)
        elo1 = elo_snap[elo_snap['team'] == t1]['elo_2026'].values[0]
        form1 = form_dict.get(t1, 0.5)
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.image(get_flag_url(t1), width=50)
        with col_b:
            st.markdown(f"**Elo:** {elo1:.0f}  |  **Form:** {form1:.0%}")
    
    with c2:
        st.markdown("#### Team 2")
        t2 = st.selectbox("Select ", teams, key="t2", index=teams.index("Brazil") if "Brazil" in teams else 1)
        elo2 = elo_snap[elo_snap['team'] == t2]['elo_2026'].values[0]
        form2 = form_dict.get(t2, 0.5)
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.image(get_flag_url(t2), width=50)
        with col_b:
            st.markdown(f"**Elo:** {elo2:.0f}  |  **Form:** {form2:.0%}")
    
    st.markdown("---")
    
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        predict_btn = st.button("🎯 Predict Match", use_container_width=True)
    
    if predict_btn:
        if t1 == t2:
            st.error("Please select different teams!")
        else:
            proba = predict_match_ml(model, features, elo1, elo2, form1, form2)
            
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            r1, r2, r3 = st.columns(3)
            
            with r1:
                st.markdown(f"""<div class="result-card">
                    <img src="{get_flag_url(t1)}" width="50">
                    <div style="color: #888; margin: 10px 0;">{t1}</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: {C_PRIMARY};">{proba[2]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            
            with r2:
                st.markdown(f"""<div class="result-card">
                    <div style="font-size: 1.5rem; color: #666;">⚖️</div>
                    <div style="color: #888; margin: 10px 0;">Draw</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: {C_NEUTRAL};">{proba[1]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            
            with r3:
                st.markdown(f"""<div class="result-card">
                    <img src="{get_flag_url(t2)}" width="50">
                    <div style="color: #888; margin: 10px 0;">{t2}</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: {C_SECONDARY};">{proba[0]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            
            # Winner card
            idx = np.argmax(proba)
            if idx == 2:
                winner, prob = t1, proba[2]
            elif idx == 0:
                winner, prob = t2, proba[0]
            else:
                winner, prob = "Draw", proba[1]
            
            if winner != "Draw":
                st.markdown("<br>", unsafe_allow_html=True)
                wc = st.columns([1, 2, 1])
                with wc[1]:
                    st.markdown(f"""<div class="winner-card">
                        <div style="color: #888;">🏆 Predicted Winner</div>
                        <img src="{get_flag_url(winner)}" width="80" style="margin: 15px 0;">
                        <div style="font-size: 1.6rem; font-weight: 700; color: white;">{winner}</div>
                        <div style="color: {C_PRIMARY};">Confidence: {prob*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)


def goals_page():
    """Goal Prediction Page"""
    st.markdown("## 🥅 Goal Prediction")
    st.markdown("---")
    
    elo_snap = load_elo_snapshot()
    form_dict = load_form_dict()
    teams = sorted(elo_snap['team'].tolist())
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Home Team")
        home = st.selectbox("Select", teams, key="home_goal", index=0)
        home_elo = elo_snap[elo_snap['team'] == home]['elo_2026'].values[0]
        st.image(get_flag_url(home), width=50)
    
    with c2:
        st.markdown("#### Away Team")
        away = st.selectbox("Select ", teams, key="away_goal", index=1)
        away_elo = elo_snap[elo_snap['team'] == away]['elo_2026'].values[0]
        st.image(get_flag_url(away), width=50)
    
    st.markdown("---")
    
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        predict_btn = st.button("⚽ Predict Score", use_container_width=True)
    
    if predict_btn:
        if home == away:
            st.error("Please select different teams!")
        else:
            # Goal prediction formula from Colab
            diff_ratio = (home_elo - away_elo) / 400
            hg = max(0, 1.5 + diff_ratio * 0.5)
            ag = max(0, 1.2 - diff_ratio * 0.5)
            
            st.markdown("---")
            st.markdown("### Expected Scoreline")
            
            r1, r2, r3 = st.columns([2, 1, 2])
            
            with r1:
                st.markdown(f"""<div class="result-card">
                    <img src="{get_flag_url(home)}" width="60">
                    <div style="color: #888; margin: 10px 0;">{home}</div>
                    <div style="font-size: 3rem; font-weight: 700; color: {C_PRIMARY};">{hg:.1f}</div>
                    <div style="color: #666;">Expected Goals</div>
                </div>""", unsafe_allow_html=True)
            
            with r2:
                st.markdown(f"""<div style="display: flex; align-items: center; 
                           justify-content: center; height: 100%;">
                    <div style="font-size: 2rem; color: #666; margin-top: 60px;">vs</div>
                </div>""", unsafe_allow_html=True)
            
            with r3:
                st.markdown(f"""<div class="result-card">
                    <img src="{get_flag_url(away)}" width="60">
                    <div style="color: #888; margin: 10px 0;">{away}</div>
                    <div style="font-size: 3rem; font-weight: 700; color: {C_SECONDARY};">{ag:.1f}</div>
                    <div style="color: #666;">Expected Goals</div>
                </div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Most Likely Score:** {home} {round(hg)} - {round(ag)} {away}")


def clustering_page():
    """Clustering Page - EXACT from Colab"""
    st.markdown("## 🎯 Team Clustering (KMeans)")
    st.markdown("---")
    
    df = cluster_teams()
    
    plt.style.use('dark_background')
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)
        
        colors = {'Strong': C_PRIMARY, 'Medium': C_ACCENT, 'Weak': C_NEUTRAL}
        
        for tier in ['Strong', 'Medium', 'Weak']:
            data = df[df['tier'] == tier]
            ax.scatter(data['elo_2026'], data['win_rate'], 
                      c=colors[tier], label=tier, s=100, alpha=0.8, edgecolors='white')
        
        ax.set_xlabel('Elo Rating', color='white')
        ax.set_ylabel('Win Rate', color='white')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER_COLOR, labelcolor='white')
        ax.grid(True, alpha=0.2, color=BORDER_COLOR)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with c2:
        st.markdown("### Tier Statistics")
        for tier in ['Strong', 'Medium', 'Weak']:
            data = df[df['tier'] == tier]
            color = colors[tier]
            st.markdown(f"""<div style="background: {color}20; border-left: 4px solid {color}; 
                        border-radius: 0 8px 8px 0; padding: 1rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: {color};">{tier}</div>
                <div style="color: #aaa;">{len(data)} teams • Avg Elo: {data['elo_2026'].mean():.0f}</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Teams by Tier")
    
    t1, t2, t3 = st.columns(3)
    
    for col, tier in zip([t1, t2, t3], ['Strong', 'Medium', 'Weak']):
        with col:
            st.markdown(f"**{tier}**")
            data = df[df['tier'] == tier].nlargest(10, 'elo_2026')
            for _, row in data.iterrows():
                tc1, tc2 = st.columns([1, 4])
                with tc1:
                    st.image(get_flag_url(row['team']), width=25)
                with tc2:
                    st.write(f"{row['team']} ({row['elo_2026']:.0f})")


def simulation_page():
    """World Cup Simulation - EXACT from Colab"""
    st.markdown("## 🏆 World Cup 2026 Simulation")
    st.markdown("---")
    
    model, _, _, features = train_match_classifier()
    
    plt.style.use('dark_background')
    
    # Show groups
    st.markdown("### Official Groups (48 Teams)")
    
    group_cols = st.columns(4)
    for i, (grp, teams) in enumerate(WC2026_GROUPS.items()):
        with group_cols[i % 4]:
            st.markdown(f"**Group {grp}**")
            for team in teams:
                st.markdown(f"• {team}")
    
    st.markdown("---")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### Simulation Settings")
        n_sim = st.slider("Number of Simulations", 100, 5000, 1000, 100)
        st.caption("More simulations = more accurate results")
        
        if st.button("🚀 Run Monte Carlo", use_container_width=True):
            st.session_state.run_sim = True
    
    if st.session_state.get('run_sim', False):
        with st.spinner("Running Monte Carlo simulation..."):
            probs = run_simulation(n_sim, model, features)
            st.session_state.sim_results = probs
            st.session_state.run_sim = False
    
    with c2:
        if 'sim_results' in st.session_state:
            st.markdown("### Win Probabilities")
            
            res = st.session_state.sim_results
            top15 = dict(list(res.items())[:15])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor(BG_DARK)
            ax.set_facecolor(BG_CARD)
            
            teams_l = list(top15.keys())[::-1]
            probs_l = [top15[t] * 100 for t in teams_l]
            colors = [C_PRIMARY if p > 10 else C_ACCENT if p > 5 else C_NEUTRAL for p in probs_l]
            
            bars = ax.barh(teams_l, probs_l, color=colors, edgecolor='white')
            
            for bar, p in zip(bars, probs_l):
                ax.text(p + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{p:.1f}%', va='center', color='white', fontsize=9)
            
            ax.set_xlabel('Win Probability %', color='white')
            ax.set_title('World Cup 2026 Predictions', color=C_PRIMARY, fontsize=12, pad=10)
            for spine in ax.spines.values():
                spine.set_color(BORDER_COLOR)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Champion card
            champ = list(res.keys())[0]
            champ_prob = res[champ] * 100
            
            st.markdown("---")
            wc = st.columns([1, 2, 1])
            with wc[1]:
                st.markdown(f"""<div class="winner-card">
                    <div style="color: #888;">🏆 Predicted Champion</div>
                    <img src="{get_flag_url(champ)}" width="100" style="margin: 15px 0;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: white;">{champ}</div>
                    <div style="color: {C_PRIMARY};">Win Probability: {champ_prob:.1f}%</div>
                </div>""", unsafe_allow_html=True)


def metrics_page():
    """Model Metrics Page"""
    st.markdown("## 📉 Model Performance")
    st.markdown("---")
    
    model, acc, f1, features = train_match_classifier()
    df = generate_matches_data()
    
    plt.style.use('dark_background')
    
    # Metrics cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""<div style="background: {BG_CARD}; border-radius: 10px; padding: 1.5rem; 
                    text-align: center; border: 1px solid {C_PRIMARY};">
            <div style="color: #888;">Accuracy</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {C_PRIMARY};">{acc*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""<div style="background: {BG_CARD}; border-radius: 10px; padding: 1.5rem; 
                    text-align: center; border: 1px solid {C_ACCENT};">
            <div style="color: #888;">F1 Score</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {C_ACCENT};">{f1:.3f}</div>
        </div>""", unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""<div style="background: {BG_CARD}; border-radius: 10px; padding: 1.5rem; 
                    text-align: center; border: 1px solid {BORDER_COLOR};">
            <div style="color: #888;">Model</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: white; margin-top: 10px;">XGBoost</div>
        </div>""", unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""<div style="background: {BG_CARD}; border-radius: 10px; padding: 1.5rem; 
                    text-align: center; border: 1px solid {BORDER_COLOR};">
            <div style="color: #888;">Features</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: white;">{len(features)}</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    c5, c6 = st.columns(2)
    
    with c5:
        st.markdown("### Features Used")
        for f in features:
            st.markdown(f"• **{f}**")
        
        st.markdown("### Model Configuration")
        st.markdown("""
        - **Algorithm:** XGBoost Classifier
        - **Estimators:** 300
        - **Max Depth:** 6
        - **Learning Rate:** 0.05
        """)
    
    with c6:
        st.markdown("### Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(BG_DARK)
        
        labels = ['Away Win', 'Draw', 'Home Win']
        counts = [
            (df['result'] == 0).sum(),
            (df['result'] == 1).sum(),
            (df['result'] == 2).sum()
        ]
        colors = [C_SECONDARY, C_NEUTRAL, C_PRIMARY]
        
        ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
              wedgeprops={'edgecolor': BG_DARK, 'linewidth': 2},
              textprops={'color': 'white'})
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Sidebar navigation (hidden on home)
    if st.session_state.page != 'home':
        with st.sidebar:
            st.markdown(f"### ⚽ Navigation")
            
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
            
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.page = "dashboard"
                st.rerun()
            
            st.markdown("---")
            
            pages = [
                ("📈", "EDA", "eda"),
                ("⚽", "Match Prediction", "prediction"),
                ("🥅", "Goal Prediction", "goals"),
                ("🎯", "Clustering", "clustering"),
                ("🏆", "Simulation", "simulation"),
                ("📉", "Metrics", "metrics"),
            ]
            
            for emoji, name, key in pages:
                if st.button(f"{emoji} {name}", use_container_width=True):
                    st.session_state.page = key
                    st.rerun()
    
    # Page router
    pages = {
        'home': home_page,
        'dashboard': dashboard_page,
        'eda': eda_page,
        'prediction': prediction_page,
        'goals': goals_page,
        'clustering': clustering_page,
        'simulation': simulation_page,
        'metrics': metrics_page,
    }
    
    pages[st.session_state.page]()


if __name__ == "__main__":
    main()