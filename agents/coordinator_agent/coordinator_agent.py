import os
import time
import datetime
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger("coordinator", "coordinator.log")
performance_logger = get_logger("legacy_performance", "performance.log")
pipeline_logger = get_logger("legacy_pipeline_runs", "pipeline_runs.log")

from agents.disease_detection_agent.disease_detection_agent import DiseaseDetectionAgent
from agents.severity_agent.severity_agent import SeverityAgent
from agents.weather_agent.weather_agent import WeatherAgent
from agents.environmental_risk_agent.environmental_risk_agent import EnvironmentalRiskAgent
from agents.advisory_agent.advisory_agent import AdvisoryAgent

class CoordinatorAgent:
    def __init__(self):
        self.disease_agent = DiseaseDetectionAgent()
        self.severity_agent = SeverityAgent()
        self.weather_agent = WeatherAgent()
        self.risk_agent = EnvironmentalRiskAgent()
        self.advisory_agent = AdvisoryAgent()
        self.workflow_version = "v1.1"

    def process_image(self, image_path: str, location_input: str = None, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """
        Orchestrates the entire inference pipeline: Disease -> Weather -> Severity -> EnvRisk -> Advisory.
        Handles errors gracefully to ensure partial success.
        """
        logger.info(f"Pipeline execution started. Workflow: Legacy Coordinator. Location: {location_input}")
        start_time = time.time()
        
        response = {
            "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(),
            "workflow_version": self.workflow_version,
            "workflow_engine": "legacy_coordinator",
            "status": "success",
            "warnings": [],
            "agent_trace": [],
            "execution_summary": [],
            "pipeline_summary": {
                "overall_status": "success",
                "total_execution_time_ms": 0,
                "successful_agents": 0,
                "failed_agents": 0,
                "skipped_agents": 0
            },
            "execution_time_ms": 0,
            "prediction": None,
            "weather": None,
            "severity": None,
            "environmental_risk": None,
            "advice": None
        }

        # Validate image
        if not os.path.exists(image_path):
            response["status"] = "error"
            response["pipeline_summary"]["overall_status"] = "error"
            response["warnings"].append(f"Image not found at path: {image_path}")
            response["execution_time_ms"] = int((time.time() - start_time) * 1000)
            response["pipeline_summary"]["total_execution_time_ms"] = response["execution_time_ms"]
            return response

        # Step 1: Disease Detection
        t0 = time.time()
        try:
            prediction = self.disease_agent.predict(image_path)
            t1 = time.time()
            response["prediction"] = prediction
            response["agent_trace"].append("DiseaseDetectionAgent")
            response["execution_summary"].append({
                "agent": "DiseaseDetectionAgent",
                "status": "success",
                "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                "execution_time_ms": int((t1 - t0) * 1000)
            })
            response["pipeline_summary"]["successful_agents"] += 1
            disease_class = prediction["disease"]
        except Exception as e:
            t1 = time.time()
            response["status"] = "error"
            response["pipeline_summary"]["overall_status"] = "error"
            response["warnings"].append(f"Disease detection failed: {e}")
            response["execution_summary"].append({
                "agent": "DiseaseDetectionAgent",
                "status": "failed",
                "reason": str(e),
                "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                "execution_time_ms": int((t1 - t0) * 1000)
            })
            response["pipeline_summary"]["failed_agents"] += 1
            response["execution_time_ms"] = int((time.time() - start_time) * 1000)
            response["pipeline_summary"]["total_execution_time_ms"] = response["execution_time_ms"]
            return response
            
        # Step 2: Weather Retrieval
        weather_data = None
        t0 = time.time()
        if location_input or (lat and lon):
            try:
                weather_data = self.weather_agent.get_weather(location_input, lat, lon)
                t1 = time.time()
                if weather_data.get("status") in ["unavailable", "timeout"] or "error" in weather_data:
                    status = weather_data.get("status", "failed")
                    if status == "unavailable": status = "failed"
                    err_msg = weather_data.get("error", "Unknown error")
                    response["warnings"].append(f"Weather retrieval failed: {err_msg}")
                    response["execution_summary"].append({
                        "agent": "WeatherAgent",
                        "status": status,
                        "reason": err_msg,
                        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                        "execution_time_ms": int((t1 - t0) * 1000)
                    })
                    response["pipeline_summary"]["failed_agents"] += 1
                    response["weather"] = weather_data # Keep to pass to risk agent
                else:
                    response["weather"] = weather_data
                    response["agent_trace"].append("WeatherAgent")
                    response["execution_summary"].append({
                        "agent": "WeatherAgent",
                        "status": "success",
                        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                        "execution_time_ms": int((t1 - t0) * 1000)
                    })
                    response["pipeline_summary"]["successful_agents"] += 1
            except Exception as e:
                t1 = time.time()
                response["warnings"].append(f"Weather retrieval failed: {e}")
                response["execution_summary"].append({
                    "agent": "WeatherAgent",
                    "status": "failed",
                    "reason": str(e),
                    "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                    "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                    "execution_time_ms": int((t1 - t0) * 1000)
                })
                response["pipeline_summary"]["failed_agents"] += 1

        # Step 3: Severity Analysis
        severity = None
        t0 = time.time()
        try:
            severity = self.severity_agent.analyze_severity(
                image_path=image_path,
                disease_class=disease_class,
                confidence=prediction["confidence"]
            )
            t1 = time.time()
            response["severity"] = severity
            response["agent_trace"].append("SeverityAgent")
            response["execution_summary"].append({
                "agent": "SeverityAgent",
                "status": "success",
                "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                "execution_time_ms": int((t1 - t0) * 1000)
            })
            response["pipeline_summary"]["successful_agents"] += 1
        except Exception as e:
            t1 = time.time()
            response["status"] = "partial_success"
            response["pipeline_summary"]["overall_status"] = "partial_success"
            response["warnings"].append(f"Severity analysis failed: {e}")
            response["execution_summary"].append({
                "agent": "SeverityAgent",
                "status": "failed",
                "reason": str(e),
                "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                "execution_time_ms": int((t1 - t0) * 1000)
            })
            response["pipeline_summary"]["failed_agents"] += 1

        # Step 4: Environmental Risk Analysis
        risk_data = None
        t0 = time.time()
        if weather_data:
            try:
                risk_data = self.risk_agent.assess_risk(disease_class, weather_data)
                t1 = time.time()
                
                if risk_data.get("status") == "skipped":
                    response["execution_summary"].append({
                        "agent": "EnvironmentalRiskAgent",
                        "status": "skipped",
                        "reason": risk_data.get("reason", "Weather unavailable"),
                        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                        "execution_time_ms": int((t1 - t0) * 1000)
                    })
                    response["pipeline_summary"]["skipped_agents"] += 1
                    response["environmental_risk"] = risk_data
                elif "error" in risk_data:
                    response["warnings"].append(f"Risk assessment failed: {risk_data['error']}")
                    response["execution_summary"].append({
                        "agent": "EnvironmentalRiskAgent",
                        "status": "failed",
                        "reason": risk_data['error'],
                        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                        "execution_time_ms": int((t1 - t0) * 1000)
                    })
                    response["pipeline_summary"]["failed_agents"] += 1
                else:
                    response["environmental_risk"] = risk_data
                    response["agent_trace"].append("EnvironmentalRiskAgent")
                    response["execution_summary"].append({
                        "agent": "EnvironmentalRiskAgent",
                        "status": "success",
                        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                        "execution_time_ms": int((t1 - t0) * 1000)
                    })
                    response["pipeline_summary"]["successful_agents"] += 1
            except Exception as e:
                t1 = time.time()
                response["warnings"].append(f"Environmental risk assessment failed: {e}")
                response["execution_summary"].append({
                    "agent": "EnvironmentalRiskAgent",
                    "status": "failed",
                    "reason": str(e),
                    "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                    "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                    "execution_time_ms": int((t1 - t0) * 1000)
                })
                response["pipeline_summary"]["failed_agents"] += 1

        from src.services.knowledge_base.models import AdvisoryContext
        # Step 5: Advisory Generation
        t0 = time.time()
        try:
            severity_str = severity["severity"] if severity else "Unknown"
            context = AdvisoryContext(
                disease_data={"disease": disease_class, "confidence": prediction["confidence"]},
                severity_data={"severity": severity_str},
                weather_data=weather_data if weather_data else {},
                risk_data=risk_data if risk_data else {},
                knowledge_context=None
            )
            advice = self.advisory_agent.generate_advice(context)
            t1 = time.time()
            response["advice"] = advice
            response["agent_trace"].append("AdvisoryAgent")
            response["execution_summary"].append({
                "agent": "AdvisoryAgent",
                "status": "success",
                "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                "execution_time_ms": int((t1 - t0) * 1000)
            })
            response["pipeline_summary"]["successful_agents"] += 1
        except Exception as e:
            t1 = time.time()
            response["status"] = "partial_success"
            response["pipeline_summary"]["overall_status"] = "partial_success"
            response["warnings"].append(f"Advisory generation failed: {e}")
            response["execution_summary"].append({
                "agent": "AdvisoryAgent",
                "status": "failed",
                "reason": str(e),
                "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                "execution_time_ms": int((t1 - t0) * 1000)
            })
            response["pipeline_summary"]["failed_agents"] += 1

        response["execution_time_ms"] = int((time.time() - start_time) * 1000)
        response["pipeline_summary"]["total_execution_time_ms"] = response["execution_time_ms"]
        
        # Build chronological timeline
        timeline_str = f"\n{response['timestamp']}\nPipeline Started\n|\n"
        perf_str = f"Run Summary:\n"
        
        for idx, step in enumerate(response["execution_summary"]):
            timeline_str += f"{step['agent']}\n{step['execution_time_ms']} ms\n"
            if step['status'] != 'success':
                timeline_str += f"Status: {step['status']}\n"
            if idx < len(response["execution_summary"]) - 1:
                timeline_str += "|\n"
                
            perf_str += f"{step['agent']}: {step['execution_time_ms']} ms\n"
            
        timeline_str += f"|\nPipeline Finished\nTotal\n{response['execution_time_ms']} ms"
        perf_str += f"Total: {response['execution_time_ms']} ms"
        
        pipeline_logger.info(timeline_str)
        performance_logger.info(perf_str)
        logger.info(f"Pipeline execution finished. Status: {response['status']}. Total Time: {response['execution_time_ms']}ms")
        
        return response
