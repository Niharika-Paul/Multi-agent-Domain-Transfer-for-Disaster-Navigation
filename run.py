import os
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

# ------------------------
# WORLD STATE
# ------------------------
world = {
    "map": """
House_A is north of Hospital_B.
Road_1 connects Bridge_C.
Bridge_C leads to School_D.
Road_2 connects Warehouse_E.
""",
    "status": """
Bridge_C is damaged.
Road_1 is blocked.
Hospital_B is overcrowded.
""",
    "task": """
Evacuate civilians from House_A to School_D safely.
"""
}

shared_memory = {
    "mapping": "",
    "risk": "",
    "resource": "",
    "routing": ""
}

# ------------------------
# MODEL CALL
# ------------------------
def call_model(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()

# ------------------------
# AGENT PROMPTS
# ------------------------

MAPPING_PROMPT = """
You are the Mapping Agent.
List reachable connections only.

Output format:
REACHABLE_PATHS:
A -> B
"""

RISK_PROMPT = """
You are the Risk Agent.
List unsafe areas based on conditions.

Output format:
RISKS:
Location unsafe
"""

RESOURCE_PROMPT = """
You are the Resource Agent.
Suggest alternative facilities if needed.

Output format:
RESOURCE_UPDATE:
text
"""

ROUTING_PROMPT = """
You are the Routing Agent.
Find a safe path using mapping and risk messages.

Output format:
ROUTE:
A -> B -> C

If impossible:
ROUTE: NO SAFE PATH
"""

# ------------------------
# AGENT RUNNER
# ------------------------
def run_agent(role_prompt):
    input_text = f"""
{role_prompt}

CITY MAP:
{world['map']}

CURRENT CONDITIONS:
{world['status']}

TASK:
{world['task']}

MESSAGES FROM OTHER AGENTS:
{shared_memory}
"""
    return call_model(input_text)

# ------------------------
# MAIN LOOP
# ------------------------
for step in range(3):
    print(f"\n--- STEP {step+1} ---")

    shared_memory["mapping"] = run_agent(MAPPING_PROMPT)
    print("\nMapping:\n", shared_memory["mapping"])

    shared_memory["risk"] = run_agent(RISK_PROMPT)
    print("\nRisk:\n", shared_memory["risk"])

    shared_memory["resource"] = run_agent(RESOURCE_PROMPT)
    print("\nResource:\n", shared_memory["resource"])

    shared_memory["routing"] = run_agent(ROUTING_PROMPT)
    print("\nRouting:\n", shared_memory["routing"])

    if "NO SAFE PATH" in shared_memory["routing"] or "->" in shared_memory["routing"]:
        print("\nSimulation finished.")
        break
