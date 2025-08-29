# main.py
import os
import uuid
import zipfile
import json
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, RootModel
from typing import Dict
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
load_dotenv()

# Configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_AGENT1 = os.getenv("AGENT1_VOICE_ID")
ELEVENLABS_VOICE_AGENT2 = os.getenv("AGENT2_VOICE_ID")
QUESTION_AGENT = os.getenv("QUESTION_AGENT")

perplexity_client = OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ensure directories exist
Path("debate_audio").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
OUTPUT_DIR = Path("debate_audio")

# Active websocket connections
active_websockets = {}

class DebateRound(BaseModel):
    question: str
    agent1: str
    agent2: str

class ResearchOutput(RootModel[Dict[str, DebateRound]]):
    pass

def slugify(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:50]

def calculate_words_per_turn(time_per_turn: int) -> int:
    """Calculate target words based on time (120-125 words per minute)"""
    return int(time_per_turn * 125)


def generate_debate_script(topic: str, turns: int, words_per_turn: int):
    """Generate complete debate script using Perplexity API with advanced academic prompting"""
    
    prompt = f"""# Advanced Debate Agent System

## Primary Objective
You are an expert debate script generator that creates high-quality, academically rigorous debates between two skilled debaters. Your role is to produce substantive, well-researched arguments with authentic citations from credible sources.

## Debate Structure Requirements

### Format Specifications
- Generate exactly {turns} rounds of debate
- Each debater gets exactly 1 minute speaking time per turn
- Target approximately {words_per_turn} words per response (assuming average speaking pace of 120-130 words/minute)
- Maintain strict alternating turns: Agent 1 (Pro) → Agent 2 (Con) → Agent 1 (Pro) → Agent 2 (Con)

### Content Quality Standards

#### 1. Academic Rigor
- **Primary Sources Only**: Use peer-reviewed research, official government data, established academic institutions
- **Book Citations**: Reference specific chapters, page numbers, and editions from authoritative books
- **Historical Documentation**: Include primary historical documents, treaties, official records
- **Expert Testimony**: Quote recognized subject matter experts with their credentials
- **Statistical Evidence**: Use recent, verifiable data from reputable organizations (WHO, UNESCO, World Bank, etc.)

#### 2. Citation Requirements
Each argument MUST include:
- **Minimum 2-3 credible citations per response**
- **Full attribution format**: "According to [Author Name] in [Publication/Book Title] ([Year]), '[exact quote]'"
- **Famous personality quotes**: When relevant, include verified quotes from historical figures, Nobel laureates, world leaders
- **Academic studies**: Reference methodology and sample sizes when citing research
- **Book references**: Include author, title, publication year, and specific page/chapter when possible

#### 3. Argumentation Excellence
- **Logical Structure**: Each response should follow: Claim → Evidence → Reasoning → Citation
- **Counter-anticipation**: Address potential counterarguments within responses
- **Nuanced Positioning**: Avoid absolute statements; acknowledge complexity while maintaining clear stance
- **Evidence Hierarchy**: Prioritize empirical evidence > expert opinion > theoretical frameworks > anecdotal evidence

## Debater Profiles

### Agent 1 (Pro/Affirmative)
**Character**: Dr. Alexandra Chen - Academic researcher with expertise in the debate topic
**Speaking Style**: 
- Measured, authoritative tone
- Uses phrases like "The empirical evidence suggests..." "Research consistently demonstrates..."
- Builds arguments systematically with clear logical progression
- Cites academic studies and expert consensus

**Argumentation Strategy**:
- Lead with strongest statistical evidence
- Reference foundational texts and seminal works
- Quote respected authorities and thought leaders
- Use comparative analysis with similar cases/studies

### Agent 2 (Con/Negative) 
**Character**: Professor Marcus Rodriguez - Critical analyst and policy expert
**Speaking Style**:
- Intellectually rigorous but accessible
- Uses phrases like "However, we must consider..." "The data reveals a more complex picture..."
- Challenges assumptions and examines underlying premises
- Cites contrarian studies and alternative interpretations

**Argumentation Strategy**:
- Identify methodological flaws in opposing studies
- Present alternative explanations for presented data
- Reference historical precedents and cautionary examples
- Quote dissenting expert opinions and minority reports

## Quality Control Guidelines

### Source Verification Standards
- **Books**: Only reference real, published works by verified authors
- **Studies**: Include institution, lead researcher, publication journal
- **Statistics**: Specify data collection period, methodology, sample size
- **Quotes**: Verify authenticity; include context and original source
- **Expert Opinions**: Include credentials, institutional affiliation, relevant expertise

### Prohibited Sources
- Social media posts or unverified online content
- Blogs or opinion pieces without academic backing
- Outdated statistics (>5 years old unless historical context required)
- Misattributed or unverified quotes
- Theoretical frameworks presented as empirical fact

## Response Generation Protocol

### For Each Round:
1. **Question Formulation**: Create specific, focused questions that explore different dimensions of the topic
2. **Research Integration**: Weave citations naturally into conversational flow
3. **Argument Development**: Build each point with evidence → reasoning → implication
4. **Professional Tone**: Maintain respectful but vigorous intellectual exchange

### Output Format Requirements
Return ONLY a valid JSON object with this structure:
{{
  "round1": {{
    "question": "Specific aspect/dimension of the topic to explore",
    "agent1": "Pro response with minimum 2-3 citations, approximately {words_per_turn} words",
    "agent2": "Con response with minimum 2-3 citations, approximately {words_per_turn} words"
  }},
  "round2": {{
    "question": "Different aspect/dimension of the topic",
    "agent1": "Pro response with citations",
    "agent2": "Con response with citations"
  }}
}}

## Example Citation Formats

### Academic Study Citation
"According to a longitudinal study by Dr. Sarah Williams published in the Journal of Applied Psychology (2023), involving 15,000 participants over 10 years, 'consistent implementation of this policy resulted in a 34% improvement in measurable outcomes.'"

### Book Citation
"As Nobel laureate Daniel Kahneman argues in 'Thinking, Fast and Slow' (2011, p. 187), 'System 1 operates automatically and quickly, with little or no effort and no sense of voluntary control.'"

### Famous Quote Citation
"Winston Churchill once observed, 'Democracy is the worst form of government, except for all the others that have been tried' - a sentiment that speaks directly to this debate about institutional reform."

### Statistical Evidence
"The World Health Organization's 2024 Global Health Report indicates that countries implementing this approach saw a 45% reduction in adverse outcomes compared to control groups (WHO, 2024, Table 3.2)."

## Success Criteria
Your debate script succeeds when:
- Each response contains verifiable, high-quality citations
- Arguments demonstrate deep subject matter expertise
- Responses feel natural and conversational despite academic rigor
- Both sides present compelling, evidence-based cases
- Citations enhance rather than interrupt the flow of argument
- Content is substantial enough to fill 1-minute speaking time
- Each round explores a distinct aspect of the overall topic

## CRITICAL INSTRUCTIONS
- Generate debates worthy of Oxford Union or Cambridge Union Society standards
- Both debaters must sound like subject matter experts with extensive research
- Every claim must be backed by authentic, verifiable sources
- Maintain intellectual honesty - acknowledge when evidence is mixed or limited
- Focus on the strongest possible arguments for each side
- Ensure citations are from real, authoritative sources
- Make responses engaging and dynamic while maintaining academic standards

## Topic for This Debate: "{topic}"

Generate exactly {turns} rounds of structured debate. Each round should explore a different dimension of this topic, with both debaters providing evidence-based arguments supported by authentic citations from credible sources."""

    response = perplexity_client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,  # Reduced for more consistent, reliable outputs
        top_p=0.9,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "schema": ResearchOutput.model_json_schema()
            }
        },
        extra_body={
            "search_mode": "academic",
            "search_depth": "deep",
            "return_citations": True,  # Ensure citations are returned
        },
        web_search_options={
            "search_context_size": "high",
            "include_domains": [
                    "scholar.google.com", 
                    "jstor.org", 
                    "pubmed.ncbi.nlm.nih.gov", 
                    "nature.com", 
                    "science.org",
                    "cambridge.org",
                    "oxford.org",
                    "wiley.com",
                    "springer.com",
                    "who.int",
                    "worldbank.org"
                ],
                "exclude_domains": [
                    "wikipedia.org", 
                    "reddit.com", 
                    "twitter.com", 
                    "facebook.com",
                    "blog.*",
                    "medium.com"
                ]
        },
        max_tokens=4000,
    )
    
    return response.choices[0].message.content

def text_to_speech(text: str, voice_id: str, output_path: str):
    """Convert text to speech using ElevenLabs"""
    response = elevenlabs.text_to_speech.convert(
        voice_id=voice_id,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",

        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.7,
            style=0.0,
        ),
    )

    # Writing the audio to a file
    with open(output_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

async def process_debate(websocket: WebSocket, topic: str, turns: int, time_per_turn: int):
    """Process the entire debate flow (fixed: serialize Pydantic objects & define topic_slug)."""
    try:
        # Prepare
        words_per_turn = calculate_words_per_turn(time_per_turn)
        await websocket.send_json({
            "type": "status",
            "message": f"Generating debate script for {turns} rounds ({words_per_turn} words per turn)..."
        })

        # --- Generate and parse script ---
        raw_script = generate_debate_script(topic, turns, words_per_turn)
        parsed_script = ResearchOutput.model_validate_json(raw_script)
        debate_script = parsed_script.root  # Dict[str, DebateRound]

        # Convert Pydantic objects to plain serializable dicts for websocket / logging
        serializable_script = {k: v.model_dump() for k, v in debate_script.items()}

        await websocket.send_json({
            "type": "script_generated",
            "message": "Debate script generated successfully!",
            "script": serializable_script
        })

        # --- Prepare output directories and names ---
        topic_slug = slugify(topic)
        topic_dir = OUTPUT_DIR / topic_slug
        topic_dir.mkdir(parents=True, exist_ok=True)

        total_audio_files = []

        # --- Process each round ---
        for round_num in range(1, turns + 1):
            round_key = f"round{round_num}"
            if round_key not in debate_script:
                # Inform client but continue
                await websocket.send_json({
                    "type": "warning",
                    "message": f"{round_key} missing in generated script; skipping."
                })
                continue

            round_data = debate_script[round_key]  # DebateRound instance

            await websocket.send_json({
                "type": "processing_round",
                "message": f"Processing Round {round_num}...",
                "round": round_num,
                "question": round_data.question
            })

            # Question audio
            question_path = topic_dir / f"question_{round_num}.mp3"
            question_text = f"Round {round_num}: {round_data.question}"
            text_to_speech(question_text, QUESTION_AGENT, str(question_path))
            total_audio_files.append(question_path)

            await websocket.send_json({
                "type": "audio_generated",
                "message": f"Question {round_num} audio generated",
                "path": str(question_path)
            })

            # Agent 1 audio
            agent1_path = topic_dir / f"agent1_answer_{round_num}.mp3"
            agent1_text = f"{round_data.agent1}"
            text_to_speech(agent1_text, ELEVENLABS_VOICE_AGENT1, str(agent1_path))
            total_audio_files.append(agent1_path)

            await websocket.send_json({
                "type": "audio_generated",
                "message": f"Agent 1 Round {round_num} audio generated",
                "path": str(agent1_path)
            })

            # Agent 2 audio
            agent2_path = topic_dir / f"agent2_answer_{round_num}.mp3"
            agent2_text = f"{round_data.agent2}"
            text_to_speech(agent2_text, ELEVENLABS_VOICE_AGENT2, str(agent2_path))
            total_audio_files.append(agent2_path)

            await websocket.send_json({
                "type": "audio_generated",
                "message": f"Agent 2 Round {round_num} audio generated",
                "path": str(agent2_path)
            })

            await websocket.send_json({
                "type": "round_complete",
                "message": f"Round {round_num} completed",
                "round": round_num
            })

        # --- Create ZIP file ---
        await websocket.send_json({"type": "status", "message": "Generating ZIP file..."})
        zip_path = OUTPUT_DIR / f"{topic_slug}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for audio_file in total_audio_files:
                # ensure audio_file is a Path
                audio_file = Path(audio_file)
                if audio_file.exists():
                    # put files inside a folder named by topic_slug in the zip
                    arcname = os.path.join(topic_slug, audio_file.name)
                    zipf.write(audio_file, arcname=arcname)

        await websocket.send_json({
            "type": "complete",
            "message": "Debate completed successfully!",
            "zip_url": f"/download/{topic_slug}.zip"
        })

    except Exception as e:
        # include the real exception message for debugging
        await websocket.send_json({
            "type": "error",
            "message": f"Error: {str(e)}"
        })

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-debate")
async def start_debate(
    topic: str = Form(...),
    turns: int = Form(3),
    time_per_turn: int = Form(1)
):
    debate_id = str(uuid.uuid4())
    return {
        "debate_id": debate_id,
        "topic": topic, 
        "turns": turns,
        "time_per_turn": time_per_turn
    }

@app.websocket("/ws/{debate_id}")
async def websocket_endpoint(websocket: WebSocket, debate_id: str):
    await websocket.accept()
    active_websockets[debate_id] = websocket
    
    try:
        # Wait for debate parameters
        data = await websocket.receive_json()
        topic = data["topic"]
        turns = data["turns"] 
        time_per_turn = data["time_per_turn"]
        
        await process_debate(websocket, topic, turns, time_per_turn)
        
    except WebSocketDisconnect:
        active_websockets.pop(debate_id, None)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error: {str(e)}"
        })
    finally:
        active_websockets.pop(debate_id, None)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = Path("debate_audio") / filename
    if file_path.exists():
        return FileResponse(file_path, media_type='application/zip', filename=filename)
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)