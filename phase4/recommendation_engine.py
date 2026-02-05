"""
Phase 4: Recommendation Engine

This module implements the core recommendation logic:
- Filters the dataset based on user input (city, cuisine, price)
- Generates a candidate set for the LLM
- Calls the Groq LLM for final ranking/recommendation
- Provides a fallback mechanism if LLM fails
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
import numpy as np
from rich.console import Console # Added for console printing
from io import StringIO # Added for default console setup

from phase1.data_loader import ZomatoDataLoader
from phase2.input_validation import PricePreference, ValidatedUserInput
from phase3.groq_client import GroqClient, GroqError, LLMRecommendationResponse


class RecommendationError(RuntimeError):
    """Raised for errors during the recommendation process."""


@dataclass(frozen=True)
class RecommendedRestaurant:
    """
    A single recommended restaurant, with optional LLM reason.
    """

    name: str
    address: str
    city: str
    cuisines: List[str]
    rating: Optional[float]
    cost_for_two: Optional[float]
    url: Optional[str]
    reason: Optional[str] = None  # From LLM


class RecommendationEngine:
    """
    Core engine for generating restaurant recommendations.
    """

    def __init__(
        self,
        data_loader: ZomatoDataLoader,
        groq_client: GroqClient,
        llm_candidate_limit: int = 20,
        top_n_recommendations: int = 10,
        console: Optional[Console] = None # Added console argument
    ):
        self.data_loader = data_loader
        self.groq_client = groq_client
        self.llm_candidate_limit = llm_candidate_limit
        self.top_n_recommendations = top_n_recommendations
        self.console = console if console is not None else Console(file=StringIO(), markup=True) # Use provided or default

        # Ensure data is processed (implicitly loads if not already)
        self.data_loader.clean_and_validate()
        self.df = self.data_loader.get_processed_data()

    def _filter_by_city(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        return df[df["city_normalized"] == city]

    def _filter_by_cuisine(self, df: pd.DataFrame, cuisine: str) -> pd.DataFrame:
        # Check if cuisine is in the list of cuisines for each restaurant
        # Handles cases where cuisines_list might be empty or NaN
        return df[df["cuisines_list"].apply(lambda x: cuisine in x if isinstance(x, list) else False)]

    def _filter_by_price(
        self, df: pd.DataFrame, price_pref: PricePreference
    ) -> pd.DataFrame:
        self.console.print(f"DEBUG: _filter_by_price - price_pref: {price_pref}")
        self.console.print(f"DEBUG: _filter_by_price - initial df length: {len(df)}")

        if df.empty or "cost_numeric" not in df.columns:
            self.console.print("DEBUG: _filter_by_price - empty df or missing cost_numeric, returning original df")
            return df

        filtered_df = df.copy()
        min_cost, max_cost = None, None

        if price_pref.exact is not None:
            min_cost, max_cost = price_pref.as_range(
                default_tolerance_pct=0.15, default_tolerance_abs=150.0
            )
            self.console.print(f"DEBUG: _filter_by_price - exact price, range: [{min_cost}, {max_cost}]")
            filtered_df = filtered_df[
                (filtered_df["cost_numeric"] >= min_cost)
                & (filtered_df["cost_numeric"] <= max_cost)
            ]
        elif price_pref.min_value is not None or price_pref.max_value is not None:
            min_cost, max_cost = price_pref.as_range()
            self.console.print(f"DEBUG: _filter_by_price - range price, range: [{min_cost}, {max_cost}]")
            if min_cost is not None and max_cost is not None:
                filtered_df = filtered_df[
                    (filtered_df["cost_numeric"] >= min_cost)
                    & (filtered_df["cost_numeric"] <= max_cost)
                ]
            elif min_cost is not None:
                filtered_df = filtered_df[filtered_df["cost_numeric"] >= min_cost]
            elif max_cost is not None:
                filtered_df = filtered_df[filtered_df["cost_numeric"] <= max_cost]
        elif price_pref.category is not None: # New block for category
            if price_pref.category == "budget": # Example: <= 500
                max_cost = 500.0
                filtered_df = filtered_df[filtered_df["cost_numeric"] <= max_cost]
                self.console.print(f"DEBUG: _filter_by_price - budget category, max_cost: {max_cost}")
            elif price_pref.category == "moderate": # Example: 501-1000
                min_cost = 501.0
                max_cost = 1000.0
                filtered_df = filtered_df[
                    (filtered_df["cost_numeric"] >= min_cost)
                    & (filtered_df["cost_numeric"] <= max_cost)
                ]
                self.console.print(f"DEBUG: _filter_by_price - moderate category, range: [{min_cost}, {max_cost}]")
            elif price_pref.category == "premium": # Example: > 1000
                min_cost = 1001.0
                filtered_df = filtered_df[filtered_df["cost_numeric"] >= min_cost]
                self.console.print(f"DEBUG: _filter_by_price - premium category, min_cost: {min_cost}")
            else:
                self.console.print(f"DEBUG: _filter_by_price - Unknown price category: {price_pref.category}")

        self.console.print(f"DEBUG: _filter_by_price - final df length: {len(filtered_df)}")
        self.console.print(f"DEBUG: _filter_by_price - final df costs: {filtered_df['cost_numeric'].tolist()}")

        return filtered_df

    def _deterministic_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic ranking: by rating (desc), then votes (desc).
        Used for candidate generation and LLM fallback.
        """
        # Fill NaN ratings with 0 for sorting purposes
        df_copy = df.copy()
        df_copy["rating_for_sort"] = df_copy["rating_numeric"].fillna(0)
        # Fill NaN votes with 0 for sorting
        df_copy["votes"] = df_copy["votes"].fillna(0).astype(int)

        # Simple weighted score (can be improved)
        df_copy["score"] = df_copy["rating_for_sort"] * 100 + df_copy["votes"]

        return df_copy.sort_values(
            by=["score", "name"], ascending=[False, True]
        ).drop(columns=["rating_for_sort", "score"])

    def get_recommendations(
        self,
        user_input: ValidatedUserInput,
        *,
        top_n: Optional[int] = None,
        llm_candidate_limit: Optional[int] = None,
    ) -> List[RecommendedRestaurant]:
        """
        Generate restaurant recommendations using a hybrid approach.

        1. Deterministic filtering based on city, cuisine, and price.
        2. Generate top-K candidates using deterministic ranking.
        3. Send candidates to Groq LLM for final ranking and reasons.
        4. Fallback to deterministic ranking if LLM fails.
        """
        if top_n is None:
            top_n = self.top_n_recommendations
        if llm_candidate_limit is None:
            llm_candidate_limit = self.llm_candidate_limit

        self.console.print(
            f"Generating recommendations for {user_input.city}, "
            f"{user_input.cuisine}, price {user_input.price.as_range()}..."
        )

        # 1. Deterministic Filtering
        filtered_df = self.df.copy()
        self.console.print(f"DEBUG: Initial restaurants: {len(filtered_df)}")

        filtered_df = self._filter_by_city(filtered_df, user_input.city)
        self.console.print(f"DEBUG: After city filter: {len(filtered_df)} restaurants")

        filtered_df = self._filter_by_cuisine(filtered_df, user_input.cuisine)
        self.console.print(f"DEBUG: After cuisine filter: {len(filtered_df)} restaurants")

        filtered_df = self._filter_by_price(filtered_df, user_input.price)
        self.console.print(f"DEBUG: After price filter: {len(filtered_df)} restaurants")

        if filtered_df.empty:
            self.console.print("No restaurants found after initial filtering.")
            return []

        # 2. Candidate Generation (deterministic top-K)
        candidates_df = self._deterministic_rank(filtered_df).head(llm_candidate_limit)
        candidates_for_llm = candidates_df.to_dict(orient="records")
        self.console.print(f"DEBUG: Candidates for LLM: {len(candidates_for_llm)}")

        if not candidates_for_llm:
            self.console.print("No candidates generated for LLM. Returning empty list.")
            return []

        # 3. LLM Recommendation (Groq)
        llm_recs_response: Optional[LLMRecommendationResponse] = None
        try:
            self.console.print("Calling Groq LLM for final ranking...")
            llm_recs_response = self.groq_client.get_recommendations(
                city=user_input.city,
                cuisine=user_input.cuisine,
                price=str(user_input.price.as_range()),  # Pass price as a string for prompt
                candidates=candidates_for_llm,
                top_n=top_n,
            )
            self.console.print(f"Groq LLM returned {len(llm_recs_response.recommendations)} recommendations.")
        except GroqError as e:
            self.console.print(f"Groq LLM call failed: {e}. Falling back to deterministic ranking.")

        final_recommendations: List[RecommendedRestaurant] = []
        if llm_recs_response and llm_recs_response.recommendations:
            # Map LLM results back to full restaurant data
            llm_ranked_names = [r.name for r in llm_recs_response.recommendations]
            llm_reasons_map = {r.name: r.reason for r in llm_recs_response.recommendations}

            # Re-order candidates_df based on LLM's ranking, keeping only LLM picks
            ordered_df = candidates_df[candidates_df["name"].isin(llm_ranked_names)].set_index("name").reindex(llm_ranked_names).reset_index()

            for _, row in ordered_df.iterrows():
                final_recommendations.append(
                    RecommendedRestaurant(
                        name=row["name"],
                        address=row["address"],
                        city=row["city_normalized"],
                        cuisines=row["cuisines_list"],
                        rating=row["rating_numeric"],
                        cost_for_two=row["cost_numeric"],
                        url=row["url"],
                        reason=llm_reasons_map.get(row["name"], None),
                    )
                )
        else:
            # 4. Fallback to Deterministic Ranking
            self.console.print("Using deterministic ranking for final recommendations.")
            for _, row in candidates_df.head(top_n).iterrows():
                final_recommendations.append(
                    RecommendedRestaurant(
                        name=row["name"],
                        address=row["address"],
                        city=row["city_normalized"],
                        cuisines=row["cuisines_list"],
                        rating=row["rating_numeric"],
                        cost_for_two=row["cost_numeric"],
                        url=row["url"],
                        reason="Deterministically ranked based on rating and votes.", # Default reason
                    )
                )

        return final_recommendations[:top_n]
