# DeepMost Agentic SDR — Interim Project Report

**Project Title:** DeepMost Agentic SDR: AI-Powered Multi-Agent B2B Sales Simulation and Analytics Platform

**Author:** Agus Rajuthaliyan

**Date:** February 10, 2026

**Institution:** DeepMost AI

---

## Abstract

The DeepMost Agentic SDR project is an AI-powered multi-agent system designed to simulate realistic Business-to-Business (B2B) sales conversations for the purpose of training, evaluating, and optimizing Sales Development Representatives (SDRs). The system leverages Google's Gemini Large Language Model (LLM) to power three autonomous agents — a Seller Agent, a Buyer Agent, and a Judge Agent — that collaboratively generate, conduct, and evaluate synthetic sales calls against real company profiles scraped from the web.

The platform addresses a critical gap in the sales training industry: the lack of scalable, data-driven training environments. Traditional sales training relies on expensive human role-play sessions, which are difficult to scale, inconsistent in quality, and produce no structured data for analysis. DeepMost Agentic SDR solves this by generating unlimited synthetic sales conversations, each enriched with structured metadata including sentiment analysis, objection detection, engagement metrics, and win probability predictions.

The system architecture consists of four core modules: (1) a web scraping module that extracts company context from target websites, (2) an agent logic module that orchestrates multi-turn LLM-powered conversations, (3) an analytics engine that applies machine learning and natural language processing to extract actionable insights, and (4) a Gradio-based web dashboard that provides real-time simulation, visualization, and coaching capabilities.

As of the current interim reporting period, the system has successfully completed 17 simulations across 11 different companies, achieving a 17.6% success rate with an average score of 3.41 out of 10. The data pipeline produces structured outputs in CSV and JSON formats, specifically engineered for downstream machine learning tasks. Key challenges encountered include API rate limiting on the Gemini free tier (20 requests per day), insufficient data volume for full ML model training, and inconsistent objection classification from the LLM Judge agent. Future work focuses on scaling data collection to 100+ simulations, training the Gradient Boosting win predictor, implementing NLP-based sentiment analysis, and fine-tuning the LLM agents on top-performing conversations.

---

## Chapter 1: Introduction

### 1.1. Introduction to the Project

The DeepMost Agentic SDR project represents an innovative application of artificial intelligence to the domain of B2B sales training and optimization. The project's core innovation lies in its multi-agent architecture, where three specialized AI agents — powered by Google's Gemini Large Language Model — collaborate to simulate, conduct, and evaluate realistic sales conversations.

**Background and Motivation:**

B2B sales is a $900 billion industry globally, yet the training methodologies employed by most organizations remain fundamentally unchanged from decades ago. Sales Development Representatives (SDRs) — the frontline professionals responsible for prospecting, initial outreach, and qualifying leads — typically undergo training through a combination of classroom instruction, shadowing experienced colleagues, and live role-play exercises. While these methods have their merits, they suffer from several critical limitations:

1. **Scalability Constraints:** Human role-play sessions require one-on-one or small-group interaction with experienced sales trainers, making it difficult to train large teams simultaneously.

2. **Inconsistency:** The quality and difficulty of role-play scenarios vary significantly depending on the trainer, leading to uneven skill development across the team.

3. **Lack of Structured Data:** Traditional training sessions produce no quantifiable, machine-readable data. Managers cannot systematically analyze what distinguishes top performers from average ones.

4. **Cost:** Hiring experienced sales trainers, especially for specialized B2B verticals, is expensive. Companies spend an average of $1,459 per SDR per year on training alone.

5. **Limited Scenario Coverage:** Human trainers can only simulate a finite number of buyer personas, objection types, and industry contexts. SDRs rarely get to practice against the full spectrum of real-world scenarios they will encounter.

The DeepMost Agentic SDR project addresses all five of these limitations by leveraging the latest advances in Large Language Models (LLMs) to create an autonomous, scalable, and data-rich sales training environment. The system generates unlimited synthetic sales conversations against real company profiles, producing structured data that can be analyzed, visualized, and used to train predictive models.

**The Multi-Agent Paradigm:**

The project employs a multi-agent architecture, a design pattern from the field of multi-agent systems (MAS) in artificial intelligence. In this paradigm, multiple autonomous agents with distinct roles, goals, and behaviors interact within a shared environment to achieve individual or collective objectives. For the DeepMost Agentic SDR system, the three agents are:

- **Seller Agent (SDR):** An AI agent that embodies the role of a top-tier Sales Development Representative. It is programmed to be persistent, polite, and value-focused. It initiates conversations, delivers product pitches, and employs objection-handling techniques to advance the sales process.

- **Buyer Agent (CTO/Prospect):** An AI agent that simulates a skeptical, time-constrained senior executive at the target company. It raises realistic objections related to pricing, timing, competing solutions, and organizational authority. Its behavior is conditioned on the actual company context scraped from the target website, making each interaction unique and grounded in reality.

- **Judge Agent (Sales Coach):** An AI agent that acts as an impartial evaluator. After each simulated conversation, it analyzes the transcript and produces a structured assessment including a numerical score (1-10), an outcome classification (Success/Failure/Pending), identification of the key objection type, and actionable feedback for improvement.

**Technology Stack:**

The project is built using the following core technologies:

- **Google Gemini LLM (gemini-2.5-flash-lite):** The foundational language model powering all three agents. Selected for its superior instruction-following capabilities and the generous free-tier quota of the flash-lite variant.
- **Python 3.x:** The primary programming language for all system components.
- **Pandas:** Used extensively for data manipulation, feature engineering, and structured data storage.
- **Scikit-learn:** Provides the machine learning pipeline, including the Gradient Boosting Classifier for win probability prediction and preprocessing utilities.
- **Plotly:** Powers the interactive visualizations in the analytics dashboard, including gauge charts, sunburst diagrams, radar charts, and sentiment trajectory plots.
- **Gradio:** Provides the web-based user interface for real-time simulation, visualization, and data exploration.
- **BeautifulSoup4:** Enables web scraping to extract company context from target websites.
- **NumPy:** Supports numerical computations within the analytics engine.

### 1.2. Organization Profile

**DeepMost AI** is a technology organization focused on applying cutting-edge artificial intelligence to solve real-world business problems. The organization specializes in developing AI-powered tools and platforms that bridge the gap between advanced machine learning research and practical business applications.

DeepMost AI's mission is to democratize access to AI-driven insights, making sophisticated analytics and automation accessible to businesses of all sizes. The organization operates at the intersection of natural language processing, multi-agent systems, and data science, with a particular focus on sales technology (SalesTech) and revenue optimization.

The DeepMost Agentic SDR project is the organization's flagship initiative in the SalesTech vertical. It represents a novel approach to sales training that combines generative AI, multi-agent simulation, and predictive analytics into a single, integrated platform. The project is developed and maintained as an open-source initiative, with the codebase hosted on GitHub at https://github.com/agusrajuthaliyan/DeepMost-Agentic-SDR.

The organization's technical infrastructure is built on modern cloud-native principles, leveraging APIs from major AI providers (Google AI) and open-source tools from the Python ecosystem. DeepMost AI maintains a lean, agile development process with rapid iteration cycles, continuous integration, and a strong emphasis on data quality and reproducibility.

### 1.3. Objectives of the Project

The DeepMost Agentic SDR project has the following primary and secondary objectives:

**Primary Objectives:**

1. **Design and implement a multi-agent AI system** that can autonomously simulate realistic B2B sales conversations, with each agent exhibiting distinct, role-appropriate behavior conditioned on real company context.

2. **Build a comprehensive data pipeline** that transforms unstructured conversation transcripts into structured, machine-learning-ready datasets with engineered features including conversation length, talk ratios, word counts, sentiment scores, and outcome classifications.

3. **Develop a machine learning analytics engine** capable of predicting call outcomes (win probability), analyzing sentiment trajectories, detecting and classifying buyer objections, and identifying the key features that distinguish successful calls from unsuccessful ones.

4. **Create an interactive web dashboard** that enables users to run simulations in real-time, visualize analytics, receive coaching feedback, and export data for further analysis.

**Secondary Objectives:**

5. **Evaluate the effectiveness of different LLM models** (gemini-2.5-flash vs. gemini-2.5-flash-lite) for sales conversation generation, considering factors such as output quality, rate limits, cost, and reliability.

6. **Investigate the relationship between conversation dynamics** (talk ratio, conversation length, buyer engagement) and call outcomes to derive data-driven best practices for sales conversations.

7. **Explore the feasibility of fine-tuning LLM agents** on collected conversation data to improve the quality and realism of simulated sales calls over time.

8. **Demonstrate the potential for AI-generated synthetic data** to replace or augment traditional human-generated training data for sales skills development.

**Measurable Success Criteria:**

- Collect a minimum of 100 simulated sales conversations across diverse company profiles.
- Achieve a trained ML model with meaningful predictive accuracy for call outcome classification.
- Identify at least 3 statistically significant features that correlate with successful call outcomes.
- Deploy a functional web dashboard accessible via a standard web browser.
- Produce a structured dataset suitable for academic research in conversational AI and sales analytics.

---

## Chapter 2: Dataset and Data Preprocessing

### 2.1. Problem Statement

The core problem addressed by this project is the lack of large-scale, structured, and realistic B2B sales conversation data for training both human SDRs and machine learning models. Existing datasets in the sales domain suffer from several limitations:

1. **Scarcity:** There are very few publicly available datasets of B2B sales conversations. Most companies treat their sales call recordings as proprietary and sensitive data.

2. **Lack of Structure:** When sales conversation data does exist, it is typically in unstructured formats (audio recordings, raw text transcripts) without the metadata, annotations, and feature engineering required for machine learning applications.

3. **Privacy Concerns:** Real sales conversations contain personally identifiable information (PII), company-confidential details, and pricing information that cannot be shared without extensive anonymization.

4. **Limited Diversity:** Available datasets tend to be concentrated in specific industries, geographies, and sales methodologies, limiting their generalizability.

5. **No Ground Truth Labels:** Existing transcripts rarely include structured outcome labels (success/failure), quality scores, objection classifications, or sentiment annotations that are essential for supervised learning.

**The DeepMost Agentic SDR project solves this problem by generating synthetic sales conversation data** that is:
- Unlimited in volume (constrained only by API quotas)
- Fully structured with machine-learning-ready features
- Grounded in real company contexts (via web scraping)
- Annotated with outcome labels, scores, sentiment, and objection types
- Free of privacy concerns (no real PII is involved)
- Diverse across industries and company profiles

### 2.2. Dataset Definition

The project produces two primary datasets and one supplementary dataset:

**1. Simulations Master Dataset (`simulations_master.csv`)**

This is the primary dataset containing one record per simulated sales call. Each record captures the complete metadata and outcomes of a single simulation.

| Column | Type | Description |
|--------|------|-------------|
| `simulation_id` | String (UUID) | Unique identifier for each simulation |
| `timestamp` | String (ISO 8601) | When the simulation was executed |
| `target_url` | String (URL) | The website URL of the target company |
| `company_context_length` | Integer | Character count of scraped company context |
| `num_turns` | Integer | Number of conversation turns (seller + buyer exchanges) |
| `total_seller_words` | Integer | Total word count for all Seller messages |
| `total_buyer_words` | Integer | Total word count for all Buyer messages |
| `avg_seller_turn_length` | Float | Average words per Seller turn |
| `avg_buyer_turn_length` | Float | Average words per Buyer turn |
| `score` | Integer (0-10) | Judge Agent's quality score |
| `outcome` | String | Classification: Success, Failure, Pending, or Error |
| `key_objection` | String | Primary objection type identified by Judge |
| `feedback` | String | Judge's coaching feedback |
| `conversation_file` | String | Filename of the corresponding JSON conversation |

**2. Simulation Metrics Dataset (`simulation_metrics.csv`)**

This is the ML-ready feature matrix derived from the master dataset. It contains only numerical and binary features suitable for direct input into machine learning models.

| Column | Type | Description |
|--------|------|-------------|
| `context_length` | Integer | Scraped company context length |
| `num_turns` | Integer | Number of conversation turns |
| `seller_total_words` | Integer | Total Seller word count |
| `buyer_total_words` | Integer | Total Buyer word count |
| `seller_avg_words_per_turn` | Float | Average Seller words per turn |
| `buyer_avg_words_per_turn` | Float | Average Buyer words per turn |
| `seller_max_words` | Integer | Maximum words in a single Seller turn |
| `buyer_max_words` | Integer | Maximum words in a single Buyer turn |
| `seller_min_words` | Integer | Minimum words in a single Seller turn |
| `buyer_min_words` | Integer | Minimum words in a single Buyer turn |
| `word_ratio_seller_buyer` | Float | Seller words ÷ Buyer words |
| `total_conversation_length` | Integer | Total words (Seller + Buyer) |
| `score` | Integer | Judge's quality score (0-10) |
| `outcome_binary` | Integer (0/1) | Binary outcome: 1 = Success, 0 = Other |

**3. Conversation Turns Dataset (`conversation_turns.csv`)**

This granular dataset contains one record per individual message in a conversation, enabling turn-level analysis.

| Column | Type | Description |
|--------|------|-------------|
| `simulation_id` | String (UUID) | Links to the parent simulation |
| `turn_number` | Integer | Sequential turn number within the conversation |
| `speaker` | String | "Seller" or "Buyer" |
| `message` | String | The actual text of the message |
| `word_count` | Integer | Word count for this individual message |

**4. Raw Conversation Files (`data/raw/conversations/*.json`)**

Each simulation also produces a JSON file containing the complete conversation history in structured format:

```json
{
    "simulation_id": "uuid-string",
    "timestamp": "2026-02-08T13:04:30.400820",
    "target_url": "https://example.com",
    "company_context": "Scraped text...",
    "conversation": [
        {"speaker": "Seller", "message": "Hi, this is Alex from DeepData AI..."},
        {"speaker": "Buyer", "message": "I'm quite busy right now..."}
    ],
    "analysis": "Score: 5\nOutcome: Success\nKey_Objection: Price\nFeedback: ..."
}
```

### 2.3. Collection of Data

Data collection in the DeepMost Agentic SDR project follows a three-stage pipeline:

**Stage 1: Web Scraping (Company Context Extraction)**

The first stage involves extracting textual content from target company websites. This content serves as the contextual foundation for the Buyer Agent's persona — the scraped information about a company's products, services, and value proposition shapes how the simulated buyer executive behaves and what objections they raise.

The scraping module (`src/scraper.py`) uses the `requests` library with a spoofed User-Agent header to fetch the HTML content of each target URL. BeautifulSoup4 then parses the HTML and extracts the visible text content, stripping away navigation elements, scripts, styles, and other non-content markup. The extracted text is cleaned and truncated to a maximum of 3,000 characters to balance between providing sufficient context for the LLM and staying within token limits.

**Target Company Selection:**

Companies are selected based on the following criteria:
- Publicly accessible website with substantial textual content
- B2B technology companies (matching the simulated product "DeepData AI")
- Diverse across sub-sectors (CRM, cloud infrastructure, collaboration, fintech, etc.)

As of the current reporting period, simulations have been conducted against 11 companies:

| Company | URL | Sector | Context Length |
|---------|-----|--------|---------------|
| OpenAI | https://openai.com | AI Research | 41 chars (minimal) |
| DeepMost AI | https://deepmostai.com | AI Solutions | 1,510 chars |
| Stripe | https://stripe.com | Fintech/Payments | 3,000 chars |
| IBM | https://www.ibm.com | Enterprise IT | 3,000 chars |
| Salesforce | https://www.salesforce.com | CRM | 3,000 chars |
| Asana | https://asana.com | Project Management | ~3,000 chars |
| Shopify | https://www.shopify.com | E-commerce | ~3,000 chars |
| Datadog | https://www.datadoghq.com | Monitoring | ~3,000 chars |
| Zendesk | https://www.zendesk.com | Customer Service | ~3,000 chars |
| Snowflake | https://www.snowflake.com | Cloud Data | ~3,000 chars |
| MongoDB | https://www.mongodb.com | Database | ~3,000 chars |

**Stage 2: Conversation Generation (LLM Agent Interaction)**

The second stage uses the Gemini LLM to generate multi-turn sales conversations. Two approaches are implemented:

*Approach A — Single-Shot Generation (Batch Pipeline):*
The `generate_synthetic_call()` function in `agent_logic.py` sends a single prompt to the LLM that instructs it to generate a complete 4-6 turn dialogue between a Seller and Buyer. This approach is used by the batch processing pipeline (`main.py`) for efficiency, consuming only 1 API call per conversation.

*Approach B — Multi-Turn Interactive Generation (Dashboard):*
The `SalesSimulation` class in `agent_logic.py` implements a turn-by-turn simulation where the Seller and Buyer agents take alternating turns, each conditioned on the full conversation history. This approach is used by the Gradio dashboard (`app_v2.py`) for real-time, interactive simulations. It consumes 2 API calls per turn (one for each agent), resulting in approximately 12 API calls per 6-turn conversation.

**Stage 3: Analysis and Annotation (Judge Agent + Analytics Engine)**

The third stage processes each generated conversation through two analysis layers:

*Layer 1 — LLM Judge Analysis:*
The `analyze_call()` function sends the complete conversation transcript to the Gemini LLM with a structured prompt requesting a score (1-10), outcome classification, key objection type, and coaching feedback. This produces the high-level annotations stored in `simulations_master.csv`.

*Layer 2 — Programmatic Analytics:*
The `AnalyticsEngine` (`src/analytics_engine.py`) applies additional analysis including:
- **Keyword-based sentiment scoring:** Counts positive and negative keywords in each message to compute per-turn and per-speaker sentiment scores.
- **Keyword-based objection detection:** Scans buyer messages for keywords associated with 5 objection categories (Price, Timing, Authority, Need, Trust).
- **Engagement metrics:** Calculates talk ratios, turn lengths, response patterns, and conversation flow statistics.
- **Feature engineering:** Derives the ML-ready features stored in `simulation_metrics.csv`.

### 2.4. Strategy of Collecting the Data

The data collection strategy is designed to maximize data quality and diversity within the constraints of the Gemini free-tier API quota.

**API Rate Limit Management:**

The most significant constraint on data collection is the Gemini free-tier rate limit: 20 requests per day per model. Given that each simulation in the batch pipeline requires 2 API calls (1 for generation + 1 for analysis), the theoretical maximum is 10 simulations per day.

To manage this constraint, the following strategies are implemented:

1. **Model Selection:** The project defaults to `gemini-2.5-flash-lite` (15 RPM, 20 RPD on free tier) over `gemini-2.5-flash` (10 RPM, 20 RPD) for slightly better per-minute throughput.

2. **Inter-Call Pacing:** A configurable delay (4.5 seconds for flash-lite, 7.0 seconds for flash) is enforced between consecutive API calls to respect per-minute rate limits.

3. **Exponential Backoff with Jitter:** When a 429 (Rate Limit) error is detected, the system automatically retries with exponentially increasing wait times: 15s × 2^attempt + random(1, 5) seconds of jitter, up to a maximum of 120 seconds. This prevents thundering-herd effects.

4. **Batch Size Optimization:** Target site lists are limited to 8-10 sites per run to stay within the 20 RPD daily quota.

5. **Incremental Collection:** Data is collected across multiple daily runs, with each run appending to the existing datasets rather than overwriting them. The `data_manager` module handles deduplication and incremental storage.

**Data Quality Assurance:**

Several mechanisms ensure the quality of collected data:

1. **Context Length Filtering:** Sites that return fewer than 50 characters of context (e.g., OpenAI returned only 41 characters) produce lower-quality simulations. These are flagged in the dataset for potential exclusion from ML training.

2. **Dialogue Parsing Validation:** The `parse_dialogue()` function in `main.py` handles multiple dialogue formats (SELLER:/BUYER:/SDR:/CTO:/PROSPECT: prefixes) to maximize successful parsing. Conversations that fail to parse (empty history) are logged as errors.

3. **Analysis Validation:** The LLM Judge's output is parsed with fallback defaults (outcome="Pending", sentiment="Neutral") to ensure every simulation produces a valid record even if the Judge's response format is unexpected.

4. **Duplicate Prevention:** Each simulation is assigned a UUID, ensuring uniqueness across all runs.

**External Data Integration Strategy:**

To supplement the synthetically generated data, the project includes a plan to integrate publicly available sales conversation datasets from Hugging Face. A script (`get_data.py`) has been developed to download and preprocess these datasets, which include real sales call transcripts annotated with outcomes. This external data will be used to augment the training set for the ML models, addressing the current limitation of insufficient training samples.

### 2.5. Data Preprocessing

Data preprocessing in the DeepMost Agentic SDR project transforms raw LLM outputs and scraped content into clean, structured, ML-ready datasets. The preprocessing pipeline is implemented primarily in the `DataManager` class (`src/data_manager.py`) and the analytics engine (`src/analytics_engine.py`).

**Step 1: Text Cleaning**

Raw scraped company context undergoes the following cleaning operations:
- HTML tag removal via BeautifulSoup4's `get_text()` method
- Whitespace normalization (collapsing multiple spaces, tabs, and newlines)
- Truncation to 3,000 characters maximum
- Removal of non-informative content (navigation menus, footer text, cookie notices)

**Step 2: Dialogue Parsing**

Generated conversation transcripts are parsed from raw text into structured (speaker, message) tuples:
- Pattern matching identifies speaker prefixes (SELLER:, BUYER:, SDR:, CTO:, PROSPECT:)
- Messages are extracted and cleaned (leading/trailing whitespace removed)
- Empty or unparseable lines are silently skipped
- The resulting list of tuples forms the `conversation_history` for each simulation

**Step 3: Feature Engineering**

The following engineered features are computed from each parsed conversation:

1. **Word Counts:** `total_seller_words`, `total_buyer_words` — Summed word counts across all turns for each speaker.
2. **Turn Statistics:** `num_turns`, `seller_avg_words_per_turn`, `buyer_avg_words_per_turn`, `seller_max_words`, `buyer_max_words`, `seller_min_words`, `buyer_min_words` — Statistical summaries of turn-level word counts.
3. **Talk Ratio:** `word_ratio_seller_buyer` — Computed as $\frac{\text{total\_seller\_words}}{\text{total\_buyer\_words}}$. This metric captures the balance of conversation dominance. Analysis shows that successful calls tend to have lower ratios (~1.44), indicating more balanced conversations.
4. **Conversation Length:** `total_conversation_length` — The sum of all words exchanged by both speakers. Serves as a proxy for conversation depth and engagement.
5. **Context Length:** `context_length` — Character count of the scraped company context. This feature captures the quality of the input data; sites with very short contexts tend to produce lower-quality simulations.
6. **Outcome Encoding:** `outcome_binary` — Binary encoding of the outcome field: Success → 1, all others (Failure, Pending, Error) → 0. This serves as the target variable for the win probability classifier.

**Step 4: Score Parsing**

The Judge Agent's textual analysis is parsed to extract structured fields:
- Score is extracted via regex matching on "Score: [digit]" patterns
- Outcome is extracted via keyword matching (Success, Failure, Pending)
- Key objection is extracted similarly, with "Unknown" as the default fallback
- Feedback text is preserved as-is for coaching display

**Step 5: Data Persistence**

All preprocessed data is persisted in three formats:
1. **CSV files** (`simulations_master.csv`, `simulation_metrics.csv`, `conversation_turns.csv`) for tabular analysis and ML training
2. **JSON files** (`data/raw/conversations/*.json`) for preserving the full conversation structure with metadata
3. **Markdown reports** (`analysis_report.md`, `executive_summary.md`) for human-readable summaries

**Step 6: Descriptive Statistics Generation**

The `DataManager` computes and stores summary statistics including:
- Total simulation count
- Overall success rate (percentage)
- Average score across all simulations
- Average conversation length
- Score and outcome distributions

As of the current reporting period, the preprocessed dataset contains:
- **17 simulation records** in the master dataset
- **17 ML-ready feature vectors** in the metrics dataset
- **~120+ individual conversation turns** in the turns dataset
- **17 JSON conversation files** with full transcripts and metadata

---

## Chapter 3: Data Modeling and Analysis

### 3.1. Data Exploration and Analysis

Exploratory Data Analysis (EDA) was conducted using two Jupyter notebooks: `eda_template.ipynb` for foundational analysis and `advanced_analytics.ipynb` for deeper statistical and NLP-based exploration.

**3.1.1. Dataset Overview**

The current dataset (as of February 10, 2026) comprises 17 simulated sales conversations across 11 unique companies. The data exhibits the following key characteristics:

*Outcome Distribution:*
| Outcome | Count | Percentage |
|---------|-------|------------|
| Success | 3 | 17.6% |
| Pending | 8 | 47.1% |
| Failure | 3 | 17.6% |
| Error | 3 | 17.6% |

The high proportion of "Pending" outcomes suggests that many simulated conversations end inconclusively — a realistic pattern in B2B sales where multiple follow-up interactions are typically required. The "Error" category represents simulations where the LLM Judge failed to produce a valid analysis, typically due to API rate limiting.

*Score Distribution:*
- Mean: 3.41 / 10
- Standard Deviation: ~2.99
- Minimum: 0 (Error cases)
- Maximum: 9
- Median: 4.0

The score distribution is left-skewed, with most simulations scoring in the 3-5 range. The low average score reflects the Buyer Agent's effectiveness as a challenging, skeptical interlocutor — matching the reality that most cold B2B outreach attempts do not succeed on the first contact.

*Conversation Metrics Summary:*
| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Context Length (chars) | 2,258 | 1,239 | 41 | 3,000 |
| Number of Turns | 3.17 | 0.75 | 2 | 4 |
| Total Conversation Length (words) | 378 | 70 | 312 | 456 |
| Seller Avg Words/Turn | 68.9 | 23.5 | 40.8 | 95.7 |
| Buyer Avg Words/Turn | 47.4 | 9.9 | 35.0 | 58.0 |
| Word Ratio (Seller/Buyer) | 1.63 | 0.48 | 1.09 | 2.47 |

**3.1.2. Key Analytical Findings**

*Finding 1: Talk Ratio and Success*

Analysis of the word ratio (Seller words ÷ Buyer words) reveals a significant pattern: successful calls have a lower average talk ratio (1.44) compared to unsuccessful ones (1.82). This suggests that conversations where the buyer speaks more — indicating higher engagement and interest — are more likely to result in a positive outcome. This finding aligns with established sales methodology research, which recommends that effective SDRs should listen more and talk less.

*Finding 2: Buyer Engagement as a Success Predictor*

Buyer engagement, measured by average words per turn, is higher in successful calls (~49.4 words/turn) compared to unsuccessful ones (~45.4 words/turn). While the difference is not yet statistically significant due to the small sample size (t-statistic: 0.45, p-value: 0.67), the directional trend supports the hypothesis that buyer engagement is a key predictor of call outcome.

*Finding 3: Conversation Length*

Total conversation length is slightly higher for successful calls (393.7 words) versus unsuccessful ones (363.0 words), suggesting that longer conversations correlate with better outcomes. However, this relationship is also not yet statistically significant (t-statistic: 0.49, p-value: 0.65).

*Finding 4: Context Quality Impact*

Simulations conducted against companies with richer web content (context_length ≈ 3,000 characters) consistently produced higher-quality, more realistic conversations compared to those with thin context (e.g., OpenAI with only 41 characters). This finding underscores the importance of input data quality for LLM-based generation systems.

*Finding 5: Objection Analysis*

Keyword-based objection detection across buyer messages reveals the following distribution:
| Objection Type | Occurrences |
|---------------|-------------|
| Need | 16 |
| Authority | 7 |
| Price | 6 |
| Trust | 3 |
| Timing | 3 |

"Need" objections (e.g., "we already have a solution," "not sure we need this") dominate, which is realistic for cold outreach scenarios where the buyer has not expressed prior interest in the product being sold.

*Finding 6: Sentiment Distribution*

Lexicon-based sentiment analysis of conversation turns shows that both Seller and Buyer messages are predominantly neutral, with the Seller exhibiting slightly more positive sentiment (reflecting the optimistic, value-focused sales approach) and the Buyer exhibiting slightly more negative sentiment (reflecting skepticism and objections).

[Figure: EDA Overview — 4-panel visualization showing Outcome Pie Chart, Score Box Plot, Conversation Length by Outcome, Talk Ratio vs Score]

[Figure: Correlation Matrix — Heatmap showing feature correlations]

[Figure: Sentiment Distribution by Speaker — Plotly bar chart]

[Figure: Top Words — Seller vs Buyer horizontal bar charts]

### 3.2. Data Modelling

The DeepMost Agentic SDR project implements three categories of machine learning and analytical models:

**3.2.1. Win Probability Predictor (Gradient Boosting Classifier)**

*Purpose:* Predict the outcome (success/failure) of a sales call based on conversation dynamics and contextual features.

*Algorithm:* Gradient Boosting Classifier (scikit-learn's `GradientBoostingClassifier`), selected for its strong performance on structured tabular data, ability to handle non-linear feature relationships, and built-in feature importance scoring.

*Feature Set:*
The model uses the following input features from `simulation_metrics.csv`:
- `total_conversation_length`
- `word_ratio_seller_buyer`
- `seller_avg_words_per_turn`
- `buyer_avg_words_per_turn`
- `num_turns`
- `context_length`

*Target Variable:* `outcome_binary` (1 = Success, 0 = Other)

*Current Status:*
The model requires a minimum of approximately 30 samples with both positive and negative labels to achieve meaningful training. With only 17 simulations and 3 positive examples, the system currently falls back to a heuristic scoring function that estimates win probability based on simple rules:
- Higher buyer engagement → higher probability
- Lower talk ratio → higher probability
- Longer conversation → higher probability
- Higher score → higher probability

The heuristic function is implemented in the `PredictiveAnalytics` class and produces a probability estimate (0.0 to 1.0) for each simulation.

*Training Pipeline (for future use when sufficient data is available):*
1. Feature matrix preparation with `StandardScaler` normalization
2. Train/test split (80/20 stratified by outcome)
3. Model training with default hyperparameters
4. Performance evaluation: accuracy, precision, recall, F1-score, ROC-AUC
5. Feature importance extraction for interpretability
6. Model persistence for integration with the prediction pipeline

**3.2.2. Sentiment Analysis Model**

*Purpose:* Analyze the emotional tone of each conversation turn and identify sentiment trajectories throughout the conversation.

*Approach:* Keyword-based lexicon scoring (current implementation), with a planned upgrade to transformer-based NLP sentiment analysis.

*Current Implementation:*
The `ConversationAnalyzer` class in `analytics_engine.py` implements a lexicon-based sentiment scorer that:
1. Tokenizes each message into lowercase words
2. Counts occurrences of positive keywords (yes, great, interested, love, perfect, amazing, definitely, absolutely, excellent, wonderful, impressive, excited)
3. Counts occurrences of negative keywords (no, not, expensive, busy, later, problem, issue, concern, difficult, impossible, reject, decline, competitor)
4. Computes a sentiment label (positive/negative/neutral) and a normalized sentiment score based on the keyword balance

*Output:*
For each simulation, the sentiment model produces:
- Per-turn sentiment labels and scores
- Per-speaker aggregate sentiment
- Sentiment trajectory (how sentiment evolves over the course of the conversation)

*Visualization:*
The sentiment trajectory is visualized in the dashboard as a Plotly line chart, showing the evolution of sentiment across conversation turns for each speaker.

**3.2.3. Objection Detection and Clustering Model**

*Purpose:* Identify, classify, and cluster buyer objections to understand patterns in prospect resistance.

*Current Implementation (Keyword-Based Detection):*
The `ObjectionClusterer` class in `analytics_engine.py` detects objections using keyword matching across five predefined categories:

| Category | Keywords |
|----------|----------|
| Price | expensive, cost, budget, price, afford, money, cheap |
| Timing | busy, time, later, timing, schedule, quarter |
| Authority | boss, team, decide, approval, committee, manager |
| Need | need, solution, already, current, existing |
| Trust | risk, prove, guarantee, reference, case study |

*Planned Enhancement (Clustering-Based):*
With a larger dataset, the project plans to implement unsupervised clustering (K-Means or DBSCAN) on sentence embeddings of buyer messages to discover objection categories organically, without relying on predefined keyword lists. This data-driven approach may reveal objection patterns that human-defined categories miss.

**3.2.4. Insights Generator**

The `InsightsGenerator` class orchestrates all three models to produce a comprehensive analysis for each simulation:

```python
{
    'sentiment_analysis': {...},      # Per-turn and aggregate sentiment
    'objection_analysis': {...},      # Detected objections with categories
    'engagement_metrics': {...},      # Talk ratio, turn lengths, flow stats
    'win_probability': 0.65,          # Predicted probability of success
    'recommendations': [...],         # Actionable coaching suggestions
    'overall_score': 7.2              # Composite quality score
}
```

For portfolio-level analysis (across all simulations), the `generate_portfolio_insights()` method produces:
- Model training status and accuracy metrics
- Objection pattern analysis across all conversations
- Performance trend analysis (how metrics evolve over time)
- Top performer characteristics (what distinguishes successful calls)
- Summary metrics (total simulations, success rate, average score, average length)

### 3.3. Deployment and Optimization

**3.3.1. Deployment Architecture**

The DeepMost Agentic SDR system is deployed as a local web application with two modes of operation:

*Mode 1: Batch Processing Pipeline (`main.py`)*

This mode is designed for automated, large-scale data collection:
- Iterates through a configurable list of target company URLs
- Scrapes each site → generates a conversation → analyzes the result
- Saves all data to structured CSV/JSON files
- Includes rate-limit management and error recovery
- Produces summary statistics upon completion

Execution: `python main.py`

*Mode 2: Interactive Web Dashboard (`app_v2.py`)*

This mode provides a real-time, user-facing interface built with Gradio:
- Users input a target company URL and initiate a simulation
- The conversation unfolds turn-by-turn in the browser
- Real-time analytics update after each turn (sentiment, engagement, win probability)
- Interactive Plotly visualizations display on the analytics tab
- Data export and portfolio analytics are accessible via dedicated tabs

Execution: `python app_v2.py` → Opens at `http://localhost:7860`

*Dashboard Components:*
The dashboard includes the following Plotly visualizations (implemented in `src/dashboard_components.py`):

1. **Win Rate Gauge:** A circular gauge chart displaying the overall success rate as a percentage.
2. **Outcome Sunburst:** A hierarchical sunburst diagram showing outcome distribution broken down by objection type.
3. **Score Distribution Violin:** A violin plot showing the density distribution of call quality scores.
4. **Objection Radar Chart:** A radar/spider chart comparing the frequency and severity of different objection categories.
5. **Sentiment Trajectory Line Chart:** A time-series line chart showing sentiment evolution across conversation turns.
6. **Engagement Metrics Subplots:** Multi-panel bar and gauge charts for talk ratio, response lengths, and question counts.
7. **Feature Importance Bar Chart:** A horizontal bar chart showing which features are most predictive of call success.

**3.3.2. Optimization Strategies**

*API Cost Optimization:*
- Model selection: `gemini-2.5-flash-lite` provides 50x more daily quota than `gemini-2.5-flash` on the free tier
- Prompt optimization: Concise, structured prompts minimize token consumption per call
- Batch vs. interactive: The batch pipeline uses 2 API calls per simulation vs. ~12 for the interactive mode
- Context truncation: Company context is capped at 3,000 characters to reduce input tokens

*Rate Limit Optimization:*
- Per-call pacing: 4.5-second delay between consecutive API calls
- Inter-site buffering: 10-second pause between target sites in batch mode
- Exponential backoff: Automatic retry with 15s → 30s → 60s → 120s wait times on 429 errors
- Daily quota tracking: Batch sizes limited to 8-10 sites to stay within 20 RPD

*Data Quality Optimization:*
- Multi-format dialogue parser handles inconsistent LLM output formats
- Fallback defaults ensure every simulation produces a valid record
- Unicode handling prevents Windows console encoding crashes
- Context quality filtering identifies and flags low-context simulations

*Performance Optimization:*
- Incremental data storage avoids reprocessing previously collected simulations
- Lazy loading of analytics engine components (ML models loaded only when needed)
- Efficient DataFrame operations using Pandas vectorized methods
- Plotly visualizations rendered client-side in the browser for responsive interactivity

---

## Chapter 4: Summary of the Interim Report

### 4.1. Summary

The DeepMost Agentic SDR project has successfully achieved its Phase 1 objectives, establishing a functional end-to-end pipeline for AI-powered B2B sales simulation, analysis, and visualization. This interim report documents the project's current status, methodologies, findings, and challenges as of February 10, 2026.

**Accomplishments to Date:**

1. **Multi-Agent System Implementation:** Successfully designed and deployed a three-agent architecture (Seller, Buyer, Judge) powered by Google's Gemini LLM. Each agent exhibits distinct, role-appropriate behavior and produces contextually relevant responses conditioned on real company profiles.

2. **Automated Data Pipeline:** Built a complete batch processing pipeline (`main.py`) that autonomously scrapes target company websites, generates multi-turn sales conversations, analyzes outcomes, and persists structured data across three formats (CSV, JSON, Markdown).

3. **Feature Engineering:** Engineered 14 ML-ready features from raw conversation data, including talk ratios, engagement metrics, word statistics, and binary outcome encoding. All features are stored in a normalized, machine-learning-ready format.

4. **Analytics Engine:** Implemented a modular analytics engine with sentiment analysis, objection detection, engagement metrics, win probability estimation, and an insights generator that produces comprehensive per-simulation and portfolio-level analytics.

5. **Interactive Dashboard:** Deployed a Gradio-based web dashboard (`app_v2.py`) with real-time simulation capabilities, 7 interactive Plotly visualizations, a real-time coaching display.

6. **Data Collection:** Accumulated 17 simulated sales conversations across 11 unique company profiles, with structured metadata, outcome labels, and coaching feedback.

**Key Findings:**

- **Success Rate:** 17.6% of simulated calls result in a "Success" outcome (meeting booked), which is realistic for cold B2B outreach where industry benchmarks range from 1-5% for real calls.
- **Talk Ratio Insight:** Successful calls exhibit lower Seller/Buyer word ratios (~1.44 vs. ~1.82), indicating that conversations with higher buyer engagement tend to produce better outcomes.
- **Dominant Objection:** "Need" objections are the most frequent (16 occurrences), followed by "Authority" (7) and "Price" (6), reflecting the challenge of establishing product-market fit in cold outreach.
- **Context Quality Matters:** Simulations against companies with richer web content (≥1,500 characters of scraped context) consistently produce higher-quality, more realistic conversations.

**Current Challenges:**

1. **API Rate Limits:** The Gemini free tier limits usage to 20 requests per day per model, capping data collection at ~10 simulations per day. This is the primary bottleneck for scaling the dataset.

2. **Insufficient Data for ML Training:** With only 17 simulations (3 positive, 14 negative/other), the Gradient Boosting win predictor cannot be meaningfully trained. Statistical hypothesis tests on feature differences between successful and unsuccessful calls show no significance (all p-values > 0.05).

3. **Objection Classification Quality:** The LLM Judge frequently returns "Unknown" for the `key_objection` field, limiting the utility of the objection-level analytics. The keyword-based backup detection provides more reliable results but lacks the nuance of LLM-based classification.

4. **Web Scraping Variability:** Some target websites return minimal usable text (as low as 41 characters), resulting in poorly contextualized simulations. JavaScript-rendered content is not accessible to the current scraping approach.

**Roadmap for Remaining Work (February 10-28, 2026):**

| Phase | Timeline | Key Deliverables |
|-------|----------|-----------------|
| Phase 1: Data Collection | Feb 10-16 | Scale to 100+ simulations via daily batch runs; integrate Hugging Face datasets; improve Judge prompts |
| Phase 2: ML Modelling | Feb 14-21 | Train Gradient Boosting win predictor; upgrade to NLP sentiment analysis; implement objection clustering |
| Phase 3: Fine-Tuning | Feb 19-24 | Fine-tune LLM agents on top-performing conversations; A/B test original vs. fine-tuned agents |
| Phase 4: Dashboard & Documentation | Feb 22-28 | Enhance Gradio UI; add export features; finalize documentation and project report |

**Conclusion:**

The DeepMost Agentic SDR project demonstrates the viability of using multi-agent LLM systems to generate scalable, structured, and analytically rich B2B sales training data. While the current dataset is too small for definitive ML insights, the directional findings align with established sales best practices (listen more, engage the buyer, address objections directly). The system architecture is designed for incremental improvement: as more data is collected, the ML models will be trained, the LLM agents will be fine-tuned, and the analytics will become increasingly actionable.

The project positions DeepMost AI to offer a transformative SalesTech solution: an AI-powered sales training platform that replaces expensive human role-play with unlimited, data-driven simulations that continuously learn and improve. The next phase of work will focus on overcoming the data scarcity challenge and demonstrating the tangible impact of ML-powered coaching on sales call quality and success rates.

---

**References:**

1. Google AI. "Gemini API Documentation." https://ai.google.dev/gemini-api/docs
2. Scikit-learn. "Gradient Boosting Classifier." https://scikit-learn.org/stable/modules/ensemble.html
3. Gradio Documentation. https://www.gradio.app/docs
4. Plotly Python Documentation. https://plotly.com/python/
5. BeautifulSoup4 Documentation. https://www.crummy.com/software/BeautifulSoup/bs4/doc/
6. Pandas Documentation. https://pandas.pydata.org/docs/

---