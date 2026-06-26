from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger("environmental_risk", "environmental_risk.log")

class EnvironmentalRiskAgent:
    def assess_risk(self, disease_class: str, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministically calculates environmental disease risk based on weather data.
        """
        if weather_data.get("status") in ["unavailable", "timeout"] or "error" in weather_data:
            logger.info("Weather service unavailable. Skipping risk assessment.")
            return {
                "status": "skipped",
                "reason": "Weather service unavailable",
                "risk_level": None,
                "spread_probability": None
            }

        w = weather_data.get("weather", {})
        temp = w.get("temperature", 25)
        humidity = w.get("humidity", 50)
        rain = w.get("precipitation", 0)

        # Default defaults if API returns None
        temp = temp if temp is not None else 25
        humidity = humidity if humidity is not None else 50
        rain = rain if rain is not None else 0

        reasons = []

        # Fungal Risk Logic
        if humidity > 85:
            fungal_risk = "Very High"
            f_score = 90
            reasons.append("High humidity (>85%) strongly favors fungal spore germination.")
        elif humidity > 70:
            fungal_risk = "High"
            f_score = 70
            reasons.append("Moderate to high humidity creates a favorable environment for fungi.")
        else:
            fungal_risk = "Low"
            f_score = 20

        # Bacterial Risk Logic
        if humidity > 80 and 25 <= temp <= 35:
            bacterial_risk = "High"
            b_score = 80
            reasons.append("Warm temperatures combined with high humidity favor bacterial multiplication.")
        elif rain > 0:
            bacterial_risk = "Medium"
            b_score = 50
            reasons.append("Rainfall can cause bacterial splash dispersal.")
        else:
            bacterial_risk = "Low"
            b_score = 20

        # Heat Stress Risk Logic
        if temp >= 35:
            heat_stress_risk = "High"
            h_score = 85
            reasons.append("Temperatures >= 35°C induce severe heat stress, weakening plant defenses.")
        elif temp >= 30:
            heat_stress_risk = "Medium"
            h_score = 50
            reasons.append("Temperatures >= 30°C may cause moderate thermal stress.")
        else:
            heat_stress_risk = "Low"
            h_score = 10

        # Spread Probability Calculation (0-100)
        # Weighting factors based on the highest risk
        spread_probability = max(f_score, b_score, h_score)
        
        # Add a modifier if it's already a diagnosed disease
        if "healthy" not in disease_class.lower():
            spread_probability = min(100, spread_probability + 10)
            reasons.append(f"Active infection of {disease_class} increases baseline spread likelihood.")
        else:
            spread_probability = min(100, spread_probability - 20)
            spread_probability = max(0, spread_probability)

        # Overall Risk Level
        if spread_probability >= 80:
            overall_risk = "High"
        elif spread_probability >= 50:
            overall_risk = "Medium"
        else:
            overall_risk = "Low"

        if not reasons:
            reasons.append("Current environmental conditions are relatively benign.")

        result = {
            "overall_risk": overall_risk,
            "fungal_risk": fungal_risk,
            "bacterial_risk": bacterial_risk,
            "heat_stress_risk": heat_stress_risk,
            "spread_probability": spread_probability,
            "reasons": reasons
        }
        
        logger.info(f"Risk Assessment Complete: Overall={overall_risk}, Spread={spread_probability}%, Fungal={fungal_risk}, Bacterial={bacterial_risk}, Heat={heat_stress_risk}")
        return result
