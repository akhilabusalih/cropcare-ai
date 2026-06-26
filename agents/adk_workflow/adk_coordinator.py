import os
import time
import datetime
from typing import Dict, Any
from src.utils.logger import get_logger, pipeline_run_id_var

logger = get_logger("adk_coordinator", "adk.log")
performance_logger = get_logger("performance", "performance.log")
pipeline_logger = get_logger("pipeline_runs", "pipeline_runs.log")

try:
    from google.adk import Agent
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

from agents.adk_workflow.tools import detect_disease_tool, analyze_severity_tool, generate_advice_tool, get_weather_tool, assess_risk_tool
from agents.coordinator_agent.coordinator_agent import CoordinatorAgent
from src.services.knowledge_base.kb_manager import KnowledgeBaseManager

try:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from pydantic import BaseModel, Field
    
    class AdvisoryState(BaseModel):
        disease_data: Dict[str, Any] = Field(default_factory=dict)
        weather_data: Dict[str, Any] = Field(default_factory=dict)
        severity_data: Dict[str, Any] = Field(default_factory=dict)
        risk_data: Dict[str, Any] = Field(default_factory=dict)
        knowledge_context: Dict[str, Any] = Field(default_factory=dict)
        
except ImportError:
    pass

class ADKCoordinatorAgent:
    def __init__(self):
        self.legacy_coordinator = CoordinatorAgent()
        self.kb_manager = KnowledgeBaseManager()
        self.workflow_version = "v2.1-adk"
        
        # Define ADK Agents matching the architecture requirements
        if ADK_AVAILABLE:
            self.disease_agent = Agent(
                name="disease_agent",
                model="gemini-2.5-flash",
                instruction="You are a Disease Detection Agent. You must invoke the detect_disease_tool.",
                tools=[detect_disease_tool]
            )
            
            self.weather_agent = Agent(
                name="weather_agent",
                model="gemini-2.5-flash",
                instruction="You are a Weather Agent. You retrieve weather data for farm locations.",
                tools=[get_weather_tool]
            )
            
            self.severity_agent = Agent(
                name="severity_agent",
                model="gemini-2.5-flash",
                instruction="You are a Severity Agent. You assess disease severity based on the disease_class and confidence.",
                tools=[analyze_severity_tool]
            )
            
            self.risk_agent = Agent(
                name="risk_agent",
                model="gemini-2.5-flash",
                instruction="You are an Environmental Risk Agent. You assess the likelihood of disease spread based on weather data.",
                tools=[assess_risk_tool]
            )
            
            self.advisory_agent = Agent(
                name="advisory_agent",
                model="gemini-2.5-flash",
                instruction="You are an Advisory Agent. Read the context from the state and use the generate_advice_tool to generate agricultural advice.",
                tools=[generate_advice_tool],
                state_schema=AdvisoryState
            )
            
            logger.info("Initializing ADK InMemorySessionService.")
            self.session_service = InMemorySessionService()

    def process_image(self, image_path: str, location_input: str = None, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """
        Executes the ADK multi-agent workflow.
        If ADK is unavailable or fails, implements a mandatory fallback to the legacy CoordinatorAgent.
        """
        logger.info(f"Pipeline execution started. Workflow: Google ADK Workflow. Location: {location_input}")
        start_time = time.time()
        
        response = {
            "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(),
            "workflow_version": self.workflow_version,
            "workflow_engine": "google_adk",
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

        if not ADK_AVAILABLE or not os.getenv("GEMINI_API_KEY"):
            reason = "Missing GEMINI_API_KEY" if ADK_AVAILABLE else "google-adk package unavailable"
            logger.warning(f"ADK Workflow unavailable ({reason}). Executing mandatory fallback to Legacy Coordinator.")
            legacy_resp = self.legacy_coordinator.process_image(image_path, location_input, lat, lon)
            legacy_resp["warnings"].append(f"ADK Workflow initialization failed ({reason}) -> Switched to Legacy Coordinator.")
            return legacy_resp

        try:
            # 1. Disease Agent Execution
            t0 = time.time()
            response["agent_trace"].append("DiseaseAgent")
            prediction_result = detect_disease_tool(image_path)
            t1 = time.time()
            if "error" in prediction_result:
                raise Exception(f"Disease detection failed: {prediction_result['error']}")
            
            response["prediction"] = prediction_result
            response["execution_summary"].append({
                "agent": "DiseaseAgent",
                "status": "success",
                "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                "execution_time_ms": int((t1 - t0) * 1000)
            })
            response["pipeline_summary"]["successful_agents"] += 1
            disease_class = prediction_result["disease"]
            confidence = prediction_result["confidence"]

            # 2. Weather Agent Execution
            weather_data = None
            t0 = time.time()
            if location_input or (lat and lon):
                response["agent_trace"].append("WeatherAgent")
                weather_data = get_weather_tool(location_input, lat, lon)
                t1 = time.time()
                if weather_data.get("status") in ["unavailable", "timeout"] or "error" in weather_data:
                    status = weather_data.get("status", "failed")
                    if status == "unavailable": status = "failed"
                    response["agent_trace"][-1] = "WeatherAgent (Timeout)" if status == "timeout" else "WeatherAgent (Failed)"
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
                    # Keep weather_data so it can be passed to subsequent agents
                    response["weather"] = weather_data
                else:
                    response["weather"] = weather_data
                    response["execution_summary"].append({
                        "agent": "WeatherAgent",
                        "status": "success",
                        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                        "execution_time_ms": int((t1 - t0) * 1000)
                    })
                    response["pipeline_summary"]["successful_agents"] += 1

            # 3. Severity Agent Execution
            t0 = time.time()
            response["agent_trace"].append("SeverityAgent")
            severity_result = analyze_severity_tool(image_path, disease_class, confidence)
            t1 = time.time()
            if "error" in severity_result:
                response["status"] = "partial_success"
                response["pipeline_summary"]["overall_status"] = "partial_success"
                response["warnings"].append(f"Severity analysis failed: {severity_result['error']}")
                response["execution_summary"].append({
                    "agent": "SeverityAgent",
                    "status": "failed",
                    "reason": severity_result['error'],
                    "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                    "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                    "execution_time_ms": int((t1 - t0) * 1000)
                })
                response["pipeline_summary"]["failed_agents"] += 1
            else:
                response["severity"] = severity_result
                response["execution_summary"].append({
                    "agent": "SeverityAgent",
                    "status": "success",
                    "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                    "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                    "execution_time_ms": int((t1 - t0) * 1000)
                })
                response["pipeline_summary"]["successful_agents"] += 1

            # 4. Environmental Risk Agent Execution
            risk_data = None
            t0 = time.time()
            if weather_data:
                response["agent_trace"].append("EnvironmentalRiskAgent")
                risk_data = assess_risk_tool(disease_class, weather_data)
                t1 = time.time()
                
                if risk_data.get("status") == "skipped":
                    response["agent_trace"][-1] = "EnvironmentalRiskAgent (Skipped)"
                    response["environmental_risk"] = risk_data
                    response["execution_summary"].append({
                        "agent": "EnvironmentalRiskAgent",
                        "status": "skipped",
                        "reason": risk_data.get("reason", "Weather unavailable"),
                        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                        "execution_time_ms": int((t1 - t0) * 1000)
                    })
                    response["pipeline_summary"]["skipped_agents"] += 1
                elif "error" in risk_data:
                    response["agent_trace"][-1] = "EnvironmentalRiskAgent (Failed)"
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
                    response["execution_summary"].append({
                        "agent": "EnvironmentalRiskAgent",
                        "status": "success",
                        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                        "execution_time_ms": int((t1 - t0) * 1000)
                    })
                    response["pipeline_summary"]["successful_agents"] += 1

            # 5. Retrieve Knowledge Base Context
            knowledge_data = {}
            if disease_class and "healthy" not in disease_class.lower():
                try:
                    record = self.kb_manager.get_disease(disease_class)
                    # Convert to dict for state payload
                    knowledge_data = record.model_dump()
                except Exception as e:
                    logger.warning(f"Could not retrieve knowledge base info for {disease_class}: {e}")

            # 6. Advisory Agent Execution via ADK Runner
            t0 = time.time()
            response["agent_trace"].append("AdvisoryAgent")
            
            run_id = pipeline_run_id_var.get() if pipeline_run_id_var.get() else "default_session"
            
            logger.info(f"Creating ADK Runner for advisory_agent. Session ID: {run_id}")
            runner = Runner(
                app_name="CropGuardian", 
                agent=self.advisory_agent, 
                session_service=self.session_service, 
                auto_create_session=True
            )
            
            state_payload = {
                "disease_data": prediction_result if prediction_result else {},
                "weather_data": weather_data if weather_data else {},
                "severity_data": severity_result if severity_result else {},
                "risk_data": risk_data if risk_data else {},
                "knowledge_context": knowledge_data
            }
            
            logger.info(f"Invoking ADK Agent. Session reuse ID: {run_id}")
            import json
            prompt_text = (
                "Read the following context and execute the generate_advice_tool to produce the advisory report.\n"
                f"Context: {json.dumps(state_payload, indent=2)}"
            )
            events = runner.run(
                user_id="user_1", 
                session_id=run_id, 
                new_message=types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)]),
                state_delta=state_payload
            )
            
            advice_result = None
            tool_invoked = False
            for event in events:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            logger.info(f"ADK Event: Tool invocation detected: {part.function_call.name}")
                            tool_invoked = True
                        if hasattr(part, 'function_response') and part.function_response:
                            logger.info(f"ADK Event: Tool return detected: {part.function_response.name}")
                            try:
                                # Extract structured dictionary directly from the tool response
                                advice_result = part.function_response.response
                                
                                # Sometimes Gemini nests the response
                                if isinstance(advice_result, dict) and "generate_advice_tool_response" in advice_result:
                                    advice_result = advice_result["generate_advice_tool_response"]
                                    
                                logger.info("ADK Event: Successfully extracted JSON payload from FunctionResponse.")
                            except Exception as e:
                                logger.error(f"Failed to parse ADK FunctionResponse: {e}")
                                advice_result = {"error": f"Failed to parse output: {str(e)}"}
                        if hasattr(part, 'text') and part.text:
                            # The model generates a conversational text summary after the tool returns.
                            # We ignore this text because we already extracted the raw structured JSON from the tool.
                            pass
            
            t1 = time.time()
            execution_time = round(t1 - t0, 2)
            logger.info(f"ADK Runner execution completed in {execution_time}s")
            
            if not advice_result:
                advice_result = {"error": "No output from ADK Runner"}
                
            if "error" in advice_result:
                # Triggers Mandatory Fallback if the runner swallowed an exception (e.g., ResourceExhausted)
                raise Exception(f"ADK Runner failed to generate advice: {advice_result['error']}")
            else:
                response["advice"] = advice_result
                response["execution_summary"].append({
                    "agent": "AdvisoryAgent",
                    "status": "success",
                    "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
                    "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
                    "execution_time_ms": int((t1 - t0) * 1000)
                })
                response["pipeline_summary"]["successful_agents"] += 1
                
            response["execution_time_ms"] = int((time.time() - start_time) * 1000)
            response["pipeline_summary"]["total_execution_time_ms"] = response["execution_time_ms"]
            
            # Build chronological timeline (ASCII-safe for Windows console)
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
            
        except Exception as e:
            # Mandatory Fallback 2: Pipeline execution failure
            logger.error(f"ADK Workflow execution failed: {e}. Executing mandatory fallback.", exc_info=True)
            legacy_resp = self.legacy_coordinator.process_image(image_path, location_input, lat, lon)
            legacy_resp["warnings"].append(f"ADK Workflow failed ({e}) -> Switched to Legacy Coordinator.")
            return legacy_resp
