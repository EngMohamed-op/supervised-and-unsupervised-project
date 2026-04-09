📊 World Cup 2026 Prediction using Machine Learning & Monte Carlo Simulation
📌 Project Overview
This project focuses on predicting the outcome of the FIFA World Cup 2026 (the first 48-team edition) using a combination of historical football data, dynamic Elo Rating systems, and an XGBoost classifier.
The main goal is to move beyond simple rankings by simulating the tournament thousands of times to identify the most likely champions, taking into account team momentum, historical "upset" patterns, and the new tournament bracket logic.
👥 Project Phases & Methodology
The project was developed through distinct experimental phases to ensure the most robust prediction engine:PhaseApproachKey TechniqueData EngineeringMulti-source HarmonizationFormer Name mapping, FIFA Filter, Chronological SortingElo EngineDynamic Strength Calculation$K$-factor scaling, $G$-factor (Goal margin), Home AdvantageML ModelingSupervised ClassificationXGBoost, Feature Engineering (elo_diff), Class balancingSimulationStochastic Monte Carlo5,000 tournament runs, "Best 3rd-place" logic, Shootout resolution
🎯 Objectives
Build a Dynamic Elo Engine: Reconstruct team strengths match-by-match from 2002 to 2026.
Quantify Footballing Logic: Measure the impact of home advantage and tournament importance.
Develop a Probabilistic Classifier: Use XGBoost to generate Win/Draw/Loss probabilities for any given matchup.
Simulate the 48-Team Format: Implement the complex 12-group structure and the Round of 32 knockout path.
Uncertainty Analysis: Use Monte Carlo simulations to account for upsets and "footballing miracles."
⚙️ Tools and Libraries
Python (Core Development)
XGBoost (Primary ML Classifier)
Pandas & NumPy (Data Manipulation)
Scikit-learn (Preprocessing, Metrics, Label Encoding)
Matplotlib & Seaborn (EDA and Trajectory Plots)
📊 Model Performance — Machine Learning Metrics
The XGBoost model was evaluated on a test set of historical international matches to ensure predictive accuracy before being used in the 2026 simulation.
🏆 Selected Simulation Results — Top 15 Win ProbabilitiesAfter 5,000 Monte Carlo simulations, 
the following win probabilities were generated for the 2026 World  
Key Components of the Engine
1. The "Anti-Jersey" Quality FilterProblem: Closed-loop systems (small islands playing only each other) caused Elo inflation (e.g., Jersey appearing in the Top 5).
2. Solution: A strict FIFA-Member filter was implemented, requiring teams to exist in official FIFA datasets to be included in the simulation.
3.  Threshold & Outcome LogicGroup Stage: Matches can end in a Draw (1 point each).
4.  Knockout Stage: Matches ending in a Draw are resolved via a get_shootout_winner() function, which uses a dedicated historical shootout win-rate dataset.
5. Home Advantage (+100 Elo)EDA confirmed that home teams score more and win significantly more often (~48%). The engine adds 100 points to the host nations (USA, Mexico, Canada) unless they face each other.
6. 🏆 Why XGBoost + Monte Carlo?This approach was selected over simple "highest rank wins" logic for several reasons:
7. Robustness to Upsets: Monte Carlo allows weaker teams (like Japan or Senegal) to win the cup in a small percentage of universes, reflecting reality.
8. Feature Synergy: XGBoost captures non-linear relationships between team form, venue, andElo difference that a simple formula would miss.Tournament Logic: It is the only way to handle the "Best 3rd-place teams" advancement rule accurately.
9. 📉 Visualizations Included Elo Trajectories: Time-series of top teams' strength evolution (2002-2026).
10. Confederation Boxplots: Comparing UEFA vs. CONMEBOL vs. others.
11. Monte Carlo Probability Bars: Visual distribution of the 5,000 outcomes.Knockout Bracket: A single "Detailed Walkthrough" path from Round of 32 to Final.
12. 🚀 Conclusion
13. By combining a refined historical Elo engine with an XGBoost classifier, this project provides a high-fidelity simulation of the 2026 World Cup. The model successfully filters out "data ghosts" (obscure teams), accounts for the new 48-team format, and identifies Spain and Argentina as the primary favorites. The inclusion of a "Detailed Walkthrough" function allows users to see not just the percentages, but a specific "story" of how a champion might emerge.
14. 👤 Contributors:
15. 1-
16. 2-
17. 3-
18. 4-
