# Zomato AI Restaurant Recommendation Service — Architecture

## Overview

This document describes the **phase-wise architecture** for a Zomato-style AI Restaurant Recommendation Service. Users provide **city**, **cuisine**, and **price**; the system returns ranked restaurant recommendations, optionally enhanced by an LLM. No deployment phase is included.

**Dataset:** [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)  
**Format:** CSV (Hugging Face Datasets / Parquet)  
**Size:** ~51.7k rows (train split)

**Relevant dataset fields (summary):**

| Field | Type | Notes |
|-------|------|--------|
| `name` | string | Restaurant name |
| `url` | string | Zomato URL |
| `address` | string | Full address |
| `location` | string | Area/neighborhood (93 values) |
| `listed_in(city)` | string | City (30 values) — **user input: city** |
| `cuisines` | string | Comma-separated — **user input: cuisine** |
| `rate` | string | e.g. "4.1/5" |
| `votes` | int64 | Number of ratings |
| `approx_cost(for two people)` | string | Cost range — **user input: price** |
| `rest_type` | string | e.g. Casual Dining, Cafe |
| `online_order`, `book_table` | string | Yes/No |
| `dish_liked`, `reviews_list`, `menu_item` | string | Optional for LLM/display |

---

## High-Level Flow

```
[Zomato Dataset] → [STEP 1: Ingest] → [STEP 2: User Input] → [STEP 3: LLM] → [STEP 4: Recommendation] → [STEP 5: Display]
```

---

## STEP 1 — Input the Zomato Data

**Goal:** Load, validate, and expose the Zomato dataset so downstream steps can filter and rank by city, cuisine, and price.

### 1.1 Data source

- **Primary:** Hugging Face Datasets — `load_dataset("ManikaSaini/zomato-restaurant-recommendation")`.
- **Fallback:** Use dataset CSV/Parquet from the repo if needed (e.g. offline or cached).

### 1.2 Load and parse

- Load the `train` (or default) split.
- Convert to a tabular structure (e.g. pandas DataFrame or in-memory list of dicts) for fast filtering.
- Normalize column names (e.g. handle `listed_in(city)` and `approx_cost(for two people)` for code-friendly access).

### 1.3 Validation and cleaning

- **City:** Map raw values to a canonical list (e.g. trim, lowercase, or use `listed_in(city)` as-is).
- **Cuisines:** Parse comma-separated `cuisines`; normalize (trim, optional lowercase) for matching.
- **Rate:** Parse numeric part from strings like `"4.1/5"` (and handle "NEW", "-", or missing).
- **Cost/Price:** Parse `approx_cost(for two people)` to numeric value (or range) for filtering and sorting; handle non-numeric/missing values. This field will be used to match user's price preference in Step 2 and filter in Step 4.
- **Missing values:** Define rules for optional fields (e.g. `phone`, `dish_liked`) — drop or fill for display only.

### 1.4 Output of Step 1

- A **single in-memory dataset** (or cached artifact) that:
  - Is keyed / indexable by city, cuisine, and price range for fast lookup.
  - Has at least: `name`, `url`, `address`, `location`, `listed_in(city)`, `cuisines`, `rate`, `votes`, `approx_cost(for two people)` (normalized to numeric), `rest_type`, and optionally `reviews_list` / `dish_liked` for LLM/display.

**No persistence to DB or API is required in this step;** the result is the “source of truth” for the rest of the pipeline.

---

## STEP 2 — User Input

**Goal:** Capture and validate user preferences (city, cuisine, and price) and pass them to the recommendation and LLM steps.

### 2.1 Input contract

- **City:** Free text or selection; must be matched to `listed_in(city)` (or `location` if design uses area instead of city).
- **Cuisine:** Free text or selection; must be matched to one or more values in the `cuisines` field (substring or exact match).
- **Price:** Numeric value or range (e.g., "500", "500-1000", "budget", "moderate", "premium") representing the approximate cost for two people; must be matched against `approx_cost(for two people)` field in the dataset.

### 2.2 Validation and normalization

- **City:**
  - Normalize input (trim, case-insensitive).
  - Resolve against the list of known cities from the dataset (from Step 1).
  - If no match: return a clear error or suggest nearest matches (e.g. “Did you mean Bangalore?”).
- **Cuisine:**
  - Normalize input.
  - Match against unique cuisine tokens in the dataset (e.g. “North Indian”, “Chinese”).
  - Support single or multiple cuisines if the product requires it.

### 2.3 Output of Step 2

- **Validated parameters:** `city`, `cuisine`, `price` (and optionally `max_results`, `sort_by`, etc.).
- Price should be normalized to a numeric value or range (min, max) for filtering.
- These parameters are the only inputs required for **STEP 4 (Recommendation)** and, if used, for **STEP 3 (LLM)**.

---

## STEP 3 — Integrate LLM (Groq)

**Goal:** Integrate the **Groq-hosted LLM** that will be used in **Step 4** to generate the final recommendations. (Optional: also used to generate short “why this place” explanations.)

### 3.1 Role of the LLM

- **Used for:**
  - **Recommendation generation (Step 4):** Given (city, cuisine, price) and a set of candidate restaurants from the dataset, select and rank the best matches.
  - **Optional explanation:** Short “why this place” / “best for” summary per restaurant.
  - **Optional intro:** One consolidated “city + cuisine + price” intro (e.g. “Best budget-friendly North Indian restaurants in Bangalore…”).
  - **Optional Q&A:** Answering follow-up questions in natural language (if scope includes a chat interface).

### 3.2 Inputs to the LLM

- **From Step 2:** `city`, `cuisine`, `price`.
- **From Step 4 (candidate set):** A bounded list of candidate restaurant records (name, location, cuisines, cost, rating/votes, rest_type, optional review excerpt).
- **Prompt design:**
  - Constrain the model to **recommend only from provided candidates** (no hallucinated restaurants).
  - Ask for **structured output** (JSON) containing ranked restaurant names/ids and short rationales.
  - Include price context so the model can prefer “value for money” when appropriate.

### 3.3 Integration pattern

- **Provider:** Groq (LLM inference).
- **Interface:** Single function or module that:
  - Accepts (city, cuisine, price, candidate restaurant summaries).
  - Returns: **ranked recommendations** (structured JSON) + optional per-restaurant blurbs.
- **Safety:** Sanitize dataset text (reviews, dish names) before sending to the LLM to avoid prompt-injection or overly long inputs.
- **Fallback:** If Groq/LLM is unavailable, fall back to deterministic ranking (rating/votes/price match) so recommendations still work.

### 3.4 Output of Step 3

- **Enrichment payload:** Text (or structured content) to be shown in **STEP 5 (Display)** alongside each recommendation and/or at the top of the results.

---

## STEP 4 — Recommendation

**Goal:** Produce an ordered list of restaurants using a **hybrid pipeline**:
1) deterministic filtering from the dataset, then 2) **Groq LLM** ranking/selection from a bounded candidate set.

### 4.1 Filtering

- **By city:** Keep rows where `listed_in(city)` (or chosen city field) matches the validated city.
- **By cuisine:** Keep rows where `cuisines` contains the validated cuisine (substring or token match, case-insensitive).
- **By price:** Keep rows where `approx_cost(for two people)` (normalized to numeric) matches the user's price preference:
  - If user provided exact value: match within a tolerance range (e.g., ±100 or ±10%).
  - If user provided range (min-max): keep restaurants where cost falls within the range.
  - If user provided category (budget/moderate/premium): apply predefined ranges.
  - Handle missing/null cost values: either exclude or show with a warning.

### 4.2 Ranking

- **Candidate generation (deterministic):** After filtering, take the top-K candidates using a simple score (e.g., rating + votes) to keep the prompt small.
- **LLM recommendation (Groq):** Send the candidate list + (city, cuisine, price) to the Groq model and ask it to:
  - pick top-N restaurants,
  - order them,
  - provide short rationales,
  - return structured JSON.
- **Fallback ranking:** If the LLM call fails, return the deterministic top-N.
- **Tie-breaking (fallback):** e.g. by `votes`, then by `name`.

### 4.3 Result set

- Limit to top N (e.g. 10 or 20).
- Build a **summary structure** per restaurant for **STEP 3** and **STEP 5:**  
  name, url, address, location, rate, votes, cost, rest_type, cuisines, dish_liked, short review excerpt.

### 4.4 Output of Step 4

- **Ordered list of recommended restaurants** (with the above fields).
- **Optional:** Same list in a format ready for the LLM (Step 3) and for the UI (Step 5).

---

## STEP 5 — Display to the User

**Goal:** Present the recommendations and any LLM-generated content in a clear, readable way.

### 5.1 Content to show per restaurant

- **From Step 4:** Name, rating, votes, cost for two, address, location, restaurant type, cuisines, dish_liked, link (url).
- **From Step 3 (if enabled):** Short LLM summary or “best for” line.

### 5.2 Layout and UX

- **List or card layout:** One block per restaurant (e.g. card with title, rating, cost, address, link).
- **Order:** Same as Step 4 (e.g. best-rated first).
- **Optional:** Filters/sort (e.g. by cost) that re-run Step 4 logic with different sort and re-render.
- **Development UI (now):** CLI that prints the ranked list and rationales.
- **Future UI (later):** WebUI that reuses the same Step 2–4 pipeline and renders results as cards/tables (no deployment details here).

### 5.3 Error and edge cases

- **No results:** Message like “No restaurants found for this city, cuisine, and price range. Try adjusting your filters.”
- **Invalid city/cuisine/price:** Show Step 2 validation message (e.g. “City not found. Did you mean …?”).
- **LLM unavailable:** Hide LLM sections; show only Step 4 data.

### 5.4 Output of Step 5

- **User-visible output:**
  - **Now:** CLI output listing recommendations + optional Groq rationales.
  - **Later:** WebUI view using the same outputs.
No deployment step is defined; this is the final phase of the architecture.

---

## Data Flow Summary

| Step | Input | Output |
|------|--------|--------|
| **1** | Hugging Face dataset (or CSV/Parquet) | Cleaned, in-memory dataset (city, cuisine, rate, cost, etc.) |
| **2** | Raw user input (city, cuisine, price) | Validated (city, cuisine, price) |
| **3** | City, cuisine, price, top-N restaurant summaries | LLM text (per-restaurant and/or intro) |
| **4** | Dataset + validated (city, cuisine, price) | Sorted list of top-N restaurants + summary structures |
| **5** | Top-N list + LLM text | Rendered UI / response to user |

---

## Technology Suggestions (No Implementation Required Here)

- **Language:** Python (pandas, Hugging Face `datasets`).
- **Step 1:** `datasets.load_dataset(...)` or pandas `read_csv` / `read_parquet`.
- **Step 2:** Simple validation functions + mapping from dataset’s unique cities/cuisines.
- **Step 3:** REST or SDK call to chosen LLM (e.g. Qwen API or OpenAI-compatible endpoint).
- **Step 4:** DataFrame filtering (by city, cuisine, price) + sorting; output list of dicts or small DTOs.
- **Step 5:** CLI (print/rich) or a minimal web template (e.g. Flask/FastAPI + HTML) or static JSON response.

---

## Out of Scope (By Design)

- Deployment (hosting, containers, production infra).
- User accounts, auth, or persistence of user preferences.
- Real-time data sync with Zomato; dataset is static/snapshot.

This architecture stays within the five steps above and does not include implementation or deployment details.
