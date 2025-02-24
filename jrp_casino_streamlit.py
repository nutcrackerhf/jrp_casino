# Import python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Switch matplotlib backend to avoid backend errors in Streamlit
plt.switch_backend('Agg')

# =============================================================================
# Sidebar Controls
# =============================================================================
st.sidebar.title("User Controls")

st.sidebar.markdown("**CASINO GAME**")

# 1) CASINO GAME: Probability of Heads in the Biased Coin
prob_heads = st.sidebar.slider(
    "Probability of Heads",
    min_value=0.50,
    max_value=0.70,
    value=0.51,  # default
    step=0.001
)

st.sidebar.markdown("**ACTIVE PORTFOLIOS SIMULATION**")

# 2) PORTFOLIO SIMULATION: Probability of + Alpha in Stock Simulation
p_alpha_pos = st.sidebar.slider(
    "Probability of +5% Alpha (P_ALPHA_POS)",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.01
)

# 3) PORTFOLIO SIMULATION: Positive/Negative Alpha Values
alpha_pos = st.sidebar.slider(
    "Positive Alpha (ALPHA_POS)",
    min_value=0.0,
    max_value=0.20,
    value=0.05,
    step=0.01
)

alpha_neg = st.sidebar.slider(
    "Negative Alpha (ALPHA_NEG)",
    min_value=-0.20,
    max_value=0.0,
    value=-0.05,
    step=0.01
)

# 4) PORTFOLIO SIMULATION: Additional slider for demonstration: 
# How many simulations for the portfolio experiment?
n_simulations_user = st.sidebar.slider(
    "Number of Monte Carlo simulations",
    min_value=1000,
    max_value=30000,
    value=10000,
    step=1000
)

st.sidebar.write("---")


# =============================================================================
# Title and Introduction
# =============================================================================
st.title(":blue[Biased Coin Casinos & the Fundamental Law of Active Management]")

# Add an image to break up text (coin flipping)
st.image("https://media.istockphoto.com/vectors/coin-flip-vector-id1163915620?k=20&m=1163915620&s=612x612&w=0&h=txJTMl9C40qJ9nnPxSj35kWKIXxPRiFANrWlLVL3FmY=", 
         caption="May the odds be forever in your favor")

st.markdown("""In this app we will explore the **expected returns** and **risk** of different betting strategies (all-in vs. distributed), 
show how **risk-adjusted returns (Sharpe ratios)** grow with more independent bets, and run a stylized **portfolio simulation** 
where we can begin to apply some of the learnings to sizing independent alpha bets.""")

st.header(":blue[SCENARIO]",divider=True)

st.markdown("""**Imagine you walk into a *special* casino with **1000 tables**, each offering a **biased coin** which lands on heads nore often than tails**.

You have **$1000** as a bankroll. You can bet on heads at as many or as few tables as you'd like, and you can bet any amount. The bets are all "even-money" bets, meaning if you bet and win, 
you make the amount that you bet. If you bet and lose, you lose the same amount.""")

# =============================================================================
# Section 1: Betting Strategies (All-In vs. Spreading Bets)
# =============================================================================
st.header(":blue[1) Return of Two Strategies: All-In vs. $1 on Each Table]",divider=True)

st.write(f"**Current Probability of Heads** (choose this in the sidebar) = {prob_heads:.3f}")

bankroll = 1000.0
prob_tails = 1 - prob_heads

st.subheader("Strategy 1: Bet All Bankroll on 1 Table")

# -- Strategy 1 calculations
strat_1_num_tables = 1
bet_per_table_1 = bankroll / strat_1_num_tables

expected_return_per_table_1 = (prob_heads * bet_per_table_1) + (prob_tails * -bet_per_table_1)
expected_return_strat1 = expected_return_per_table_1 * strat_1_num_tables

st.write(f"- **Bet**: ${bet_per_table_1:.2f} on {strat_1_num_tables} table(s)")
st.write(f"- **Expected Return**: ${expected_return_strat1:.2f}")

st.subheader("Strategy 2: Bet $1 on Each of 1000 Tables")

strat_2_num_tables = 1000
bet_per_table_2 = bankroll / strat_2_num_tables  # = $1 if bankroll=1000

expected_return_per_table_2 = (prob_heads * bet_per_table_2) + (prob_tails * -bet_per_table_2)
expected_return_strat2 = expected_return_per_table_2 * strat_2_num_tables

st.write(f"- **Bet**: ${bet_per_table_2:.2f} on {strat_2_num_tables} table(s)")
st.write(f"- **Expected Return**: ${expected_return_strat2:.2f}")

st.markdown("""  
Both strategies have about the **same** expected return. However, as we will see, **risk** differs.
""")


# =============================================================================
# Section 2: Risk (Standard Deviation) of the Two Strategies
# =============================================================================
st.header(":blue[2) Risk (Standard Deviation) of the Two Strategies]",divider=True)

# We'll assume the following are defined earlier in your code (e.g., in the sidebar or top-level):
#   bankroll = 1000.0
#   prob_heads = st.sidebar.slider("Probability of Heads", min_value=0.50, max_value=0.70, value=0.51, step=0.001)
#   prob_tails = 1 - prob_heads

# --- Strategy 1: All-in
ex_all_in = bankroll * (2 * prob_heads - 1)  # E[X]
var_all_in = (bankroll ** 2) - (ex_all_in ** 2)
if var_all_in < 0:
    var_all_in = 0.0  # to avoid small negative due to float precision
std_strat1 = np.sqrt(var_all_in)
expected_return_strat1 = prob_heads * bankroll + prob_tails * (-bankroll)

# --- Strategy 2: Spread $1 across 1000 tables
bet_each = 1.0
ex_one_dollar = bet_each * (2 * prob_heads - 1)  # E[X] for a single $1 bet
var_one_dollar = (bet_each ** 2) - (ex_one_dollar ** 2)
if var_one_dollar < 0:
    var_one_dollar = 0.0
std_strat2 = np.sqrt(1000 * var_one_dollar)
expected_return_strat2 = (prob_heads * bet_each + prob_tails * -bet_each) * 1000

st.write(f"- **Std Dev (All-In on One Table)**: ${std_strat1:,.2f}")
st.write(f"- **Std Dev (1 dollar on each of 1000 Tables)**: ${std_strat2:,.2f}")

with st.expander("Detailed Explainer of Standard Deviation Calculations"):
    st.markdown(rf"""## Why So Much Lower Variance When You Spread Bets?

### A Single “All-In” Bet Has Extreme Swings

When you put $1000 on one coin flip, the outcome is either **+1000** or **-1000**.

This yields a **huge spread** (i.e., **variance** and **standard deviation**) around the expected return—even if your 
probability of winning is slightly above 50%.

### Many Small Bets Benefit From Diversification

Splitting 1000 into 1000 bets of 1 dollar each is a large number of independent coin flips.

By the **law of large numbers**, many small, uncorrelated bets tend to **offset each other’s ups and downs**, 
producing a **narrower distribution** around the same average. When you split your 1000 dollars into numerous small bets, 
each outcome will either be +1 dollar or -1 dollar, with a small edge in your favor. Because the coin flips at each table are 
independent, the high outcomes tend to be balanced by the low outcomes across many trials. This “averaging out” 
effect narrows the range of possible total outcomes, resulting in a distribution that is tightly clustered around the 
same mean you’d get from one giant bet—but without the extreme all-or-nothing swings.

Variances add across independent bets, but each bet’s variance is tiny (it’s just 1dollar squared minus a very 
small expected payoff term). The total is far lower in absolute dollar terms than the variance of one giant 1000 
dollar bet. In a large number of small, independent bets, the variance of each bet is simply 1^2−(small expected payoff
)^2, which is far less than 1000^2. When you add up the variances of these small bets, the total variance is still much 
smaller than that of a single 1000 dollar bet. This is because the variance of one 1000 dollar bet is on the order of 
(1000)^2, whereas the variance of a $1 bet is on the order of (1)^2 — and even multiplied by 1000 such bets, it 
remains significantly lower.

### Mathematically:

The single big bet has variance $\sim 1000^2$.

A A thousand $1 bets have a total variance of 
1000 × (small variance per $1 bet). Even multiplying by 1000, it’s an order 
of magnitude smaller than a single $1000 bet’s variance 
because each $1 bet’s variance is only on the scale of 1 (not 1000²).

""")



# with st.expander("Show Detailed Standard Deviation Calculations"):
#     st.markdown(rf"""
# **Mathematical Derivation**  

# Let \( p = \text{{prob_heads}} \) be the probability of winning a single coin flip.

# ### Strategy 1: All-in on One Table

# - You bet your entire bankroll, \( b = 1000 \), on a single coin flip.
# - The random payoff \( X \) has two outcomes:
# \[
# X =
# \begin{cases}
#   +b, & \text{{with probability }} p,\\
#   -b, & \text{{with probability }} 1 - p.
# \end{cases}
# \]

# - **Expected value**:
# \[
# \mathbb{{E}}[X] = p\cdot b + (1 - p)\cdot(-b) = b\, (2p - 1).
# \]

# - **Second moment** (\(\mathbb{{E}}[X^2]\)):
# \[
# \mathbb{{E}}[X^2] = p\, (b^2) + (1-p)\, (b^2) = b^2.
# \]

# - **Variance**:
# \[
# \mathrm{{Var}}(X) = b^2 - \bigl[b\,(2p - 1)\bigr]^2.
# \]

# - **Standard Deviation**:
# \[
# \sigma = \sqrt{{\mathrm{{Var}}(X)}}.
# \]

# ### Strategy 2: Spread \$1 across 1000 Tables

# - You split your bankroll into 1000 bets of \$1 each.
# - Each \$1 bet has payoff \(\pm 1\), with probabilities \(p\) and \(1-p\).

#   - **Expected payoff** per bet:
#   \[
#   \mathbb{{E}}[X_i] = p\cdot(+1) + (1-p)\cdot(-1) = (2p - 1).
#   \]
#   - \(\mathbb{{E}}[X_i^2] = 1\) implies
#   \[
#   \mathrm{{Var}}(X_i) = 1 - (2p - 1)^2.
#   \]

# - Summing 1000 independent bets gives
# \[
# X_\mathrm{{total}} = \sum_{{i=1}}^{{1000}} X_i.
# \]
#   - **Expected total**:
#   \[
#   \mathbb{{E}}[X_\mathrm{{total}}] = 1000 \,\bigl(2p - 1\bigr).
#   \]
#   - **Variance** (via independence):
#   \[
#   \mathrm{{Var}}(X_\mathrm{{total}}) 
#   = 1000\bigl[1 - (2p - 1)^2\bigr].
#   \]
#   - **Standard Deviation**:
#   \[
#   \sigma_\mathrm{{total}}
#   = \sqrt{{1000\bigl[1 - (2p - 1)^2\bigr]}}.
#   \]

# Hence, both strategies have **similar expected profit** (~\$20), but 
# the *all-in bet* has a far larger standard deviation (\(\approx \$1000\)) 
# compared to the diversified approach (\(\approx \$31.62\)).
# """)

st.markdown("""
The above shows that while the expected return of strategies above is the 
same ($20), the different volatilities mean that their sharpe ratios are different. 
This difference in **volatility** directly translates into a higher 
**risk-adjusted** return (Sharpe ratio) for the more diversified strategy.
""")

### PLOT ###

# Create a two-column subplot figure
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Expected Return", "Standard Deviation"),
    horizontal_spacing=0.15  # Adjust spacing as needed
)

# Left Subplot: Expected Return
fig.add_trace(
    go.Bar(
        x=["Strategy 1", "Strategy 2"],
        y=[expected_return_strat1, expected_return_strat2],
        marker_color=["royalblue", "orange"],
        name="Expected Return"
    ),
    row=1, col=1
)

# Right Subplot: Standard Deviation
fig.add_trace(
    go.Bar(
        x=["Strategy 1", "Strategy 2"],
        y=[std_strat1, std_strat2],
        marker_color=["royalblue", "orange"],
        name="Std Dev"
    ),
    row=1, col=2
)

# Update layout for better aesthetics
fig.update_layout(
    title_text="Side-by-Side Comparison: Return vs. Volatility",
    showlegend=False,  # If you want to hide the legend 
    height=400, 
    width=800
)
fig.update_yaxes(title_text="Dollars ($)", row=1, col=1)
fig.update_yaxes(title_text="Dollars ($)", row=1, col=2)

# In Streamlit, render with:
st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Section 3: Sharpe Ratio and the Fundamental Law
# =============================================================================
st.header(":blue[3) Sharpe Ratio and the Fundamental Law of Active Management]",divider=True)

# Sharpe ratio for strategy 1
sharpe_strat1 = ex_all_in / std_strat1 if std_strat1 > 0 else 0

# Sharpe ratio for strategy 2
ex_strat2 = expected_return_strat2
sharpe_strat2 = ex_strat2 / std_strat2 if std_strat2 > 0 else 0

st.write(f"- **Sharpe Ratio (Strategy 1)**: {sharpe_strat1:.4f}")
st.write(f"- **Sharpe Ratio (Strategy 2)**: {sharpe_strat2:.4f}")

product_sqrt_breadth = sharpe_strat1 * np.sqrt(strat_2_num_tables)
st.write(f"Now, there's a relatinonship between these two numbers. Strategy 2's Sharpe ratio can be arrived at by taking the Strategy 1 Sharpe ratio and multiplying it by the square root of breadth - in other words by the square root of 1000.")
st.write(f"**If we multiply Strategy 1’s Sharpe by √1000, we get: {product_sqrt_breadth:.4f}**")
st.markdown(r"""
:green[**Fundamental Law of Active Management**]:  
More independent bets (Breadth) → Higher risk-adjusted performance (IR).
""")

st.latex(r"""  
{IR} = \text{IC} \times \sqrt{\text{Breadth}}""")

st.markdown(r"""  
Where IR = Information Ratio (a measure of risk-adjusted alpha) and IC = Information Coefficient 
(a measure of predictive ability (skill/edge)""")

st.markdown("*Generalizing, it turns out that the sharpe of a multi-table strategy is equivalent to the sharpe of a single-table strategy multiplied by the square root of the number of times we play that bet. SO there are two sure ways of increasing your risk adjusted return: either raise the edge or raise the number of bets you place.*")

# =============================================================================
# Section 4: Sharpe Ratio vs. Number of Tables (2D)
# =============================================================================
st.header(":blue[4) Sharpe Ratio vs. Number of Tables (2D)]",divider=True)

st.markdown("""
We now calculate how the Sharpe ratio evolves as we increase the number
of independent coin-flip bets—each with the same edge defined by our chosen probability of winning.
""")

def calculate_sharpe(num_tables, p=0.51):
    """Return a simple 'edge-based' Sharpe ratio for a given # bets."""
    edge = (2*p - 1)
    ex1 = edge
    var1 = 1**2 - ex1**2
    sd1 = np.sqrt(var1) if var1>0 else 0
    ex_total = num_tables * ex1
    sd_total = np.sqrt(num_tables) * sd1
    return ex_total / sd_total if sd_total>0 else 0

num_tables_range = np.arange(1, 1001)
sharpe_values = [calculate_sharpe(n, p=prob_heads) for n in num_tables_range]

fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(num_tables_range, sharpe_values, label="Sharpe Ratio", color="blue")
ax1.axhline(y=sharpe_values[0], color='red', linestyle='--', label="Sharpe at 1 Table")
ax1.set_title("Sharpe Ratio Improvement with More Tables")
ax1.set_xlabel("Number of Tables (Bets)")
ax1.set_ylabel("Sharpe Ratio")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)


# =============================================================================
# Section 5: 3D Visualization (Sharpe vs. #Tables vs. Edge)
# =============================================================================
st.header(":blue[5) 3D Visualization: Sharpe vs. Number of Tables vs. Probability of Winning]",divider=True)

st.markdown("""
Use this surface plot to see how Sharpe changes with both the **# of bets** AND
the **probability of winning** (i.e. your edge).
""")

def calculate_sharpe_3d(num_tables, p):
    edge = (2*p - 1)
    ex1 = edge
    var1 = 1.0**2 - ex1**2
    sd1 = np.sqrt(var1) if var1>0 else 0
    ex_total = num_tables * ex1
    sd_total = np.sqrt(num_tables) * sd1
    return ex_total/sd_total if sd_total>0 else 0

num_tables_axis = np.linspace(1, 1000, 100, dtype=int)
edge_axis = np.linspace(0.51, 0.60, 50)  # from 51% to 60%
X, Y = np.meshgrid(num_tables_axis, edge_axis)
Z = np.zeros_like(X, dtype=float)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = calculate_sharpe_3d(X[i,j], Y[i,j])

fig2 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig2.update_layout(
    title='Sharpe Ratio: #Tables vs. Probability of Win',
    scene=dict(
        xaxis_title='Number of Tables (Bets)',
        yaxis_title='Prob. of Winning',
        zaxis_title='Sharpe Ratio'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)
st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# Section 6: Portfolio Simulation (Mean, Median, etc.)
# =============================================================================
st.header(":blue[6) Portfolio Simulation: Multiple Stocks & Idiosyncratic Alpha]",divider=True)

st.markdown(f"""
Finally, let's do a multi-stock simulation with user-chosen alpha parameters:
- Edge/Probability of +Alpha (P_ALPHA_POS): **{p_alpha_pos:.2f}**  
- Positive Alpha (ALPHA_POS): **{alpha_pos:.2%}**  
- Negative Alpha (ALPHA_NEG): **{alpha_neg:.2%}**  
- Simulations: **{n_simulations_user}**  

We'll keep the market drift at 5% and volatility at 15% for a single period.
""")

# Let the user pick some portfolio sizes to test (small input example)
portfolio_sizes_str = st.text_input(
    "Enter a comma-separated list of portfolio sizes (default: 5,10,20,40,80,200)",
    "5,10,20,40,80,200"
)
try:
    portfolio_sizes = [int(x.strip()) for x in portfolio_sizes_str.split(",")]
except:
    portfolio_sizes = [5,10,20,40,80,200]

MARKET_DRIFT = 0.05
MARKET_VOL   = 0.15
INITIAL_WEALTH = 100.0

results = []

for n_stocks in portfolio_sizes:
    final_wealths = []
    
    for _ in range(n_simulations_user):
        stock_returns = []
        
        for __ in range(n_stocks):
            market_ret = np.random.normal(MARKET_DRIFT, MARKET_VOL)
            alpha_draw = alpha_pos if np.random.rand() < p_alpha_pos else alpha_neg
            total_ret = market_ret + alpha_draw
            stock_returns.append(total_ret)
        
        portfolio_ret = np.mean(stock_returns)  # equally weighted
        end_wealth = INITIAL_WEALTH * (1.0 + portfolio_ret)
        final_wealths.append(end_wealth)
    
    final_wealths = np.array(final_wealths)
    
    mean_wealth   = np.mean(final_wealths)
    median_wealth = np.median(final_wealths)
    pct25_wealth  = np.percentile(final_wealths, 25)
    pct75_wealth  = np.percentile(final_wealths, 75)
    
    final_returns = (final_wealths / INITIAL_WEALTH) - 1.0
    mean_return   = np.mean(final_returns)
    std_return    = np.std(final_returns)
    sharpe_like = mean_return / std_return if std_return > 0 else np.nan
    
    results.append({
        'n_stocks': n_stocks,
        'mean_wealth': mean_wealth,
        'median_wealth': median_wealth,
        'pct25_wealth': pct25_wealth,
        'pct75_wealth': pct75_wealth,
        'std_return': std_return,
        'mean_return': mean_return,
        'sharpe_like': sharpe_like
    })

df_results = pd.DataFrame(results).sort_values('n_stocks')
st.write("**Simulation Results** (sorted by # of stocks):")
st.dataframe(df_results)

# --- Plot 1: Median Wealth vs. # of Stocks ---
fig3, ax3 = plt.subplots(figsize=(8,5))
ax3.plot(df_results['n_stocks'], df_results['median_wealth'], marker='o', label='Median Wealth')
ax3.set_title("Median Ending Wealth vs. Number of Stocks")
ax3.set_xlabel("Number of Stocks")
ax3.set_ylabel("Median Ending Wealth")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)

# --- Plot 2: Std Dev of Returns vs. # of Stocks ---
fig4, ax4 = plt.subplots(figsize=(8,5))
ax4.plot(df_results['n_stocks'], df_results['std_return'], marker='s', color='red', label='Std Dev of Returns')
ax4.set_title("Portfolio Return Volatility vs. Number of Stocks")
ax4.set_xlabel("Number of Stocks")
ax4.set_ylabel("Std Dev of Single-Period Return")
ax4.grid(True)
ax4.legend()
st.pyplot(fig4)

# --- Plot 3: Sharpe-like Ratio vs. # of Stocks ---
fig5, ax5 = plt.subplots(figsize=(8,5))
ax5.plot(df_results['n_stocks'], df_results['sharpe_like'], marker='^', color='green', label='Sharpe-like Ratio')
ax5.set_title("Risk-Adjusted Return vs. Number of Stocks")
ax5.set_xlabel("Number of Stocks")
ax5.set_ylabel("Mean Return / Std Dev of Returns")
ax5.grid(True)
ax5.legend()
st.pyplot(fig5)

st.markdown("""
**Observation**:  
- As we increase the number of stocks, the **volatility** (std dev of returns) typically *decreases*,  
- The **median** (and mean) wealth might not increase drastically, but the **risk-adjusted** measure (Sharpe-like ratio) often goes *up*, illustrating the benefit of diversification when alpha is uncorrelated.

---
""")

# =============================================================================
# Section 6: Concluding Thoughts
# =============================================================================
st.header(":blue[6) Open Questions]",divider=True)

st.markdown("""
There are a couple of reasons why we may not want to indefinitely increase the breadth, the number of tables that we play at: 
1) Does my IC drop as breadth increases? 
2) Are the bets independent or correlated? (Menchero's "Diversification Coefficient") 
3) Another complication: does my skill vary depending on the bet? 
i.e. do the odds change depending on the coin's country of origin? (Menchero's "Signal Quality").
""")


st.success("The End. Adjust the sliders or text inputs in the sidebar to explore different scenarios.")
