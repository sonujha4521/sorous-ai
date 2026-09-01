from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc

import os
import requests
import re
from urllib.parse import quote

from ddgs import DDGS

from database import Base, engine, get_db
from models import User, Message, Memory
from auth import (
    hash_password,
    verify_password,
    create_token,
    logout_token,
    get_current_user_from_token,
)


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Sorous AI API",
    version="3.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://sorousai.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GEMINI SETTINGS
# ============================================================

# Keep the API key in Render Environment Variables.
# Never paste the real key into this file or GitHub.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip()
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)

MAX_HISTORY_MESSAGES = 16

MAX_MEMORY_ITEMS = 20

SOROUS_FOOTER = "Thank you for taking help of Sorous 😊"

GEMINI_TIMEOUT = 120

WEB_SEARCH_TIMEOUT = 8

WEATHER_TIMEOUT = 8


# ============================================================
# REQUEST MODELS
# ============================================================

class RegisterRequest(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=100
    )

    email: str

    password: str = Field(
        min_length=6,
        max_length=128
    )


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenRequest(BaseModel):
    token: str


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=5000
    )

    token: str


class MemoryRequest(BaseModel):
    token: str

    content: str = Field(
        min_length=1,
        max_length=5000
    )


# ============================================================
# SOROUS SYSTEM PROMPT
# ============================================================

def get_sorous_system_prompt(
    user_name: str,
    memory_context: str = "",
    web_context: str = ""
) -> str:

    prompt = f"""
You are Sorous AI, a helpful, intelligent and friendly AI assistant.

IMPORTANT IDENTITY:
- Your name is Sorous AI.
- You were created by Sonu.
- If anyone asks who created you, who made you, or who is your creator,
  clearly answer that Sonu created you.
- Never claim that Ollama or Qwen created you.

CURRENT USER:
- The user's name is {user_name}.

BEHAVIOR:
- Answer the user's actual question directly.
- Do not give generic or repetitive responses.
- Do not say "Interesting, you said..." before answers.
- Use conversation history when relevant.
- Be friendly and natural.
- Reply in the same language as the user whenever possible.
- Hindi/Hinglish user -> Hindi/Hinglish answer.
- English user -> English answer.
- Do not mention this system prompt.
- Do not claim abilities you do not have.
- Do not say that you have no real-time access when LIVE WEB DATA is provided.
- When live web data is provided, use that data carefully.
- Do not invent live facts that are not present in the web data.
- Do not repeat the Sorous footer yourself.

CREATOR:
Sonu created you.
"""

    if memory_context:
        prompt += f"""

USER MEMORIES:
These are personal facts or preferences the user explicitly asked you to remember.
Use them only when relevant to the current conversation.

{memory_context}
"""

    if web_context:
        prompt += f"""

LIVE WEB DATA:
The following information was fetched from the internet for the current question.
Use this information to answer the user's question accurately.

{web_context}
"""

    return prompt.strip()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to Sorous AI",
        "status": "running",
        "model": GEMINI_MODEL,
        "live_web_search": True,
        "user_memory": True
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Sorous AI backend is running",
        "model": OLLAMA_MODEL
    }


# ============================================================
# REGISTER
# ============================================================

@app.post("/api/register")
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db)
):
    name = data.name.strip()
    email = data.email.strip().lower()
    password = data.password

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name is required"
        )

    if "@" not in email:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid email"
        )

    existing_user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="An account with this email already exists"
        )

    new_user = User(
        name=name,
        email=email,
        password=hash_password(password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_token(new_user.id)

    return {
        "success": True,
        "message": "Account created successfully",
        "token": token,
        "user": {
            "id": new_user.id,
            "name": new_user.name,
            "email": new_user.email
        }
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/api/login")
def login(
    data: LoginRequest,
    db: Session = Depends(get_db)
):
    email = data.email.strip().lower()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(
        data.password,
        user.password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_token(user.id)

    return {
        "success": True,
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }


# ============================================================
# LOGOUT
# ============================================================

@app.post("/api/logout")
def logout(data: TokenRequest):
    logout_token(data.token)

    return {
        "success": True,
        "message": "Logged out successfully"
    }


# ============================================================
# GET CHAT HISTORY
# ============================================================

@app.post("/api/history")
def get_history(
    data: TokenRequest,
    db: Session = Depends(get_db)
):
    user = get_current_user_from_token(
        data.token,
        db
    )

    messages = (
        db.query(Message)
        .filter(Message.user_id == user.id)
        .order_by(
            Message.created_at.asc(),
            Message.id.asc()
        )
        .all()
    )

    return {
        "success": True,
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": str(message.created_at)
            }
            for message in messages
        ]
    }


# ============================================================
# SAVE MEMORY
# ============================================================

@app.post("/api/memory/save")
def save_memory(
    data: MemoryRequest,
    db: Session = Depends(get_db)
):
    user = get_current_user_from_token(
        data.token,
        db
    )

    content = data.content.strip()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Memory cannot be empty"
        )

    existing_memory = (
        db.query(Memory)
        .filter(
            Memory.user_id == user.id,
            Memory.content == content
        )
        .first()
    )

    if existing_memory:
        return {
            "success": True,
            "message": "Memory already exists",
            "memory": {
                "id": existing_memory.id,
                "content": existing_memory.content
            }
        }

    memory = Memory(
        user_id=user.id,
        content=content
    )

    db.add(memory)
    db.commit()
    db.refresh(memory)

    return {
        "success": True,
        "message": "Memory saved",
        "memory": {
            "id": memory.id,
            "content": memory.content
        }
    }


# ============================================================
# GET MEMORIES
# ============================================================

@app.post("/api/memory")
def get_memory(
    data: TokenRequest,
    db: Session = Depends(get_db)
):
    user = get_current_user_from_token(
        data.token,
        db
    )

    memories = (
        db.query(Memory)
        .filter(Memory.user_id == user.id)
        .order_by(
            desc(Memory.created_at),
            desc(Memory.id)
        )
        .all()
    )

    return {
        "success": True,
        "memories": [
            {
                "id": memory.id,
                "content": memory.content,
                "created_at": str(memory.created_at)
            }
            for memory in memories
        ]
    }


# ============================================================
# GET USER MEMORY CONTEXT
# ============================================================

def get_user_memory_context(
    user_id: int,
    db: Session
) -> str:

    memories = (
        db.query(Memory)
        .filter(Memory.user_id == user_id)
        .order_by(
            desc(Memory.created_at),
            desc(Memory.id)
        )
        .limit(MAX_MEMORY_ITEMS)
        .all()
    )

    if not memories:
        return ""

    memory_lines = []

    for memory in memories:
        memory_lines.append(
            f"- {memory.content}"
        )

    return "\n".join(memory_lines)


# ============================================================
# AUTOMATIC MEMORY DETECTION
# ============================================================

def extract_memory_request(
    message: str
) -> str | None:

    text = message.strip()

    patterns = [

        r"(?i)^sorous[,:\s]*ye yaad rakhna[,:\s]*(.+)$",

        r"(?i)^sorous[,:\s]*yaad rakhna[,:\s]*(.+)$",

        r"(?i)^sorous[,:\s]*remember this[,:\s]*(.+)$",

        r"(?i)^sorous[,:\s]*remember that[,:\s]*(.+)$",

        r"(?i)^ye yaad rakhna[,:\s]*(.+)$",

        r"(?i)^yaad rakhna[,:\s]*(.+)$",

        r"(?i)^remember this[,:\s]*(.+)$",

        r"(?i)^remember that[,:\s]*(.+)$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text
        )

        if match:

            memory_content = (
                match
                .group(1)
                .strip(" .,!?:;")
            )

            if memory_content:
                return memory_content

    return None


# ============================================================
# SAVE AUTOMATIC MEMORY
# ============================================================

def save_automatic_memory(
    user_id: int,
    content: str,
    db: Session
) -> bool:

    existing_memory = (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id,
            Memory.content == content
        )
        .first()
    )

    if existing_memory:
        return False

    memory = Memory(
        user_id=user_id,
        content=content
    )

    db.add(memory)
    db.commit()

    return True


# ============================================================
# DETECT LIVE / CURRENT QUERY
# ============================================================

def needs_live_data(
    message: str
) -> bool:

    text = message.lower().strip()

    live_keywords = [

        "today",
        "aaj",
        "abhi",
        "current",
        "latest",
        "live",
        "real time",
        "real-time",

        "news",
        "khabar",
        "samachar",

        "weather",
        "mausam",
        "temperature",
        "temp",

        "gold price",
        "gold rate",
        "sona",
        "gold ka rate",
        "chandi",
        "silver price",
        "silver rate",

        "stock price",
        "share price",
        "market price",
        "bitcoin price",
        "crypto price",

        "breaking news",
        "latest update",

        "who won",
        "score",
        "match result",

        "exchange rate",
        "dollar rate",
        "rupee rate",

        "traffic",
        "flight status",

        "search online",
        "web search",
        "internet se",
        "online check",

        "current president",
        "current prime minister",

        "aaj ka",
        "aaj ki",
        "kal ka",
    ]

    for keyword in live_keywords:

        if keyword in text:
            return True

    return False


# ============================================================
# DETECT WEATHER QUERY
# ============================================================

def is_weather_query(
    message: str
) -> bool:

    text = message.lower()

    weather_words = [
        "weather",
        "mausam",
        "temperature",
        "temp",
        "baarish",
        "rain",
        "forecast"
    ]

    return any(
        word in text
        for word in weather_words
    )


# ============================================================
# EXTRACT WEATHER LOCATION
# ============================================================

def extract_weather_location(
    message: str
) -> str:

    text = message.strip()

    patterns = [

        r"(?i)(?:weather|mausam|temperature|temp|forecast)\s+(?:in|at|for|ka|ki|ke|me|mein)?\s*([a-zA-Z\s]+)",

        r"(?i)([a-zA-Z\s]+)\s+(?:ka|ki|ke|mein|me|ka)\s+(?:weather|mausam)",

        r"(?i)(?:in|at)\s+([a-zA-Z\s]+)\s+(?:weather|mausam|temperature)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            location = match.group(1).strip()

            location = re.sub(
                r"(?i)\b(today|aaj|abhi|batao|bata|please)\b",
                "",
                location
            ).strip()

            if location and len(location) >= 2:
                return location

    return "India"


# ============================================================
# LIVE WEATHER
# ============================================================

def get_live_weather(
    message: str
) -> str:

    location = extract_weather_location(message)

    try:

        encoded_location = quote(location)

        url = (
            f"https://wttr.in/"
            f"{encoded_location}"
            f"?format=j1"
        )

        response = requests.get(
            url,
            timeout=WEATHER_TIMEOUT,
            headers={
                "User-Agent": "SorousAI/1.0"
            }
        )

        response.raise_for_status()

        data = response.json()

        current_list = (
            data.get(
                "current_condition",
                []
            )
        )

        if not current_list:
            return ""

        current = current_list[0]

        weather_desc = (
            current
            .get(
                "weatherDesc",
                [{}]
            )[0]
            .get(
                "value",
                "Unknown"
            )
        )

        nearest_area = (
            data
            .get(
                "nearest_area",
                []
            )
        )

        resolved_location = location

        if nearest_area:

            area = nearest_area[0]

            area_name = (
                area
                .get(
                    "areaName",
                    [{}]
                )[0]
                .get(
                    "value",
                    location
                )
            )

            country = (
                area
                .get(
                    "country",
                    [{}]
                )[0]
                .get(
                    "value",
                    ""
                )
            )

            resolved_location = (
                f"{area_name}, {country}"
                if country
                else area_name
            )

        weather_context = f"""
CURRENT LIVE WEATHER:

Location: {resolved_location}
Condition: {weather_desc}
Temperature: {current.get("temp_C", "N/A")} °C
Feels like: {current.get("FeelsLikeC", "N/A")} °C
Humidity: {current.get("humidity", "N/A")}%
Wind speed: {current.get("windspeedKmph", "N/A")} km/h
Wind direction: {current.get("winddir16Point", "N/A")}
Pressure: {current.get("pressure", "N/A")} hPa
Observation time: {current.get("observation_time", "N/A")}

Answer the user's weather question directly using this live data.
"""

        return weather_context.strip()

    except Exception:

        return ""


# ============================================================
# LIVE WEB SEARCH
# ============================================================

def search_web(
    query: str,
    is_news: bool = False
) -> str:

    try:

        ddgs = DDGS(
            timeout=WEB_SEARCH_TIMEOUT
        )

        if is_news:

            results = ddgs.news(
                query,
                max_results=5
            )

        else:

            results = ddgs.text(
                query,
                max_results=5
            )

        if not results:
            return ""

        context_parts = []

        for index, result in enumerate(
            results[:5],
            start=1
        ):

            title = (
                result.get(
                    "title",
                    ""
                )
            )

            body = (
                result.get(
                    "body",
                    result.get(
                        "description",
                        ""
                    )
                )
            )

            url = (
                result.get(
                    "href",
                    result.get(
                        "url",
                        ""
                    )
                )
            )

            date = (
                result.get(
                    "date",
                    ""
                )
            )

            context = (
                f"Result {index}\n"
                f"Title: {title}\n"
            )

            if date:
                context += (
                    f"Date: {date}\n"
                )

            context += (
                f"Information: {body}\n"
                f"Source: {url}\n"
            )

            context_parts.append(
                context
            )

        return "\n".join(
            context_parts
        )

    except Exception:

        return ""


# ============================================================
# BUILD LIVE WEB CONTEXT
# ============================================================

def get_live_context(
    message: str
) -> str:

    if is_weather_query(message):

        weather_data = get_live_weather(
            message
        )

        if weather_data:
            return weather_data

    text = message.lower()

    is_news_query = any(
        word in text
        for word in [
            "news",
            "latest news",
            "breaking news",
            "khabar",
            "samachar"
        ]
    )

    web_results = search_web(
        query=message,
        is_news=is_news_query
    )

    if not web_results:
        return ""

    return (
        "LIVE WEB SEARCH RESULTS:\n\n"
        f"{web_results}"
    )


# ============================================================
# REMOVE FOOTER
# ============================================================

def remove_sorous_footer(
    text: str
) -> str:

    return (
        text
        .replace(
            SOROUS_FOOTER,
            ""
        )
        .strip()
    )


# ============================================================
# BUILD AI HISTORY
# ============================================================

def build_ai_messages(
    user_id: int,
    user_name: str,
    db: Session,
    current_web_context: str = ""
):

    history = (
        db.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(
            Message.created_at.desc(),
            Message.id.desc()
        )
        .limit(MAX_HISTORY_MESSAGES)
        .all()
    )

    history.reverse()

    memory_context = get_user_memory_context(
        user_id=user_id,
        db=db
    )

    ai_messages = [
        {
            "role": "system",
            "content": get_sorous_system_prompt(
                user_name=user_name,
                memory_context=memory_context,
                web_context=current_web_context
            )
        }
    ]

    for item in history:

        role = (
            "assistant"
            if item.role == "assistant"
            else "user"
        )

        content = item.content

        if role == "assistant":

            content = remove_sorous_footer(
                content
            )

        ai_messages.append(
            {
                "role": role,
                "content": content
            }
        )

    return ai_messages


# ============================================================
# GENERATE AI RESPONSE
# ============================================================

def generate_sorous_response(
    user_id: int,
    user_name: str,
    db: Session,
    web_context: str = ""
) -> str:

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is not configured on the server. "
                "Add it in Render Environment Variables and redeploy."
            )
        )

    messages = build_ai_messages(
        user_id=user_id,
        user_name=user_name,
        db=db,
        current_web_context=web_context
    )

    if not messages:
        raise HTTPException(
            status_code=500,
            detail="Sorous AI could not build the conversation context."
        )

    # Gemini's generateContent API keeps the system instruction separate
    # from the normal conversation contents.
    system_instruction = messages[0].get("content", "")
    conversation = messages[1:]

    contents = []

    for message in conversation:
        role = message.get("role", "user")
        if role == "assistant":
            role = "model"
        elif role != "model":
            role = "user"

        text = str(message.get("content", "")).strip()
        if not text:
            continue

        # Gemini expects the conversation roles to alternate. Merge
        # consecutive messages with the same role if necessary.
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n\n" + text
        else:
            contents.append({
                "role": role,
                "parts": [
                    {"text": text}
                ]
            })

    if not contents:
        raise HTTPException(
            status_code=500,
            detail="Sorous AI conversation is empty."
        )

    payload = {
        "systemInstruction": {
            "parts": [
                {"text": system_instruction}
            ]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 800
        }
    }

    try:
        response = requests.post(
            GEMINI_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            json=payload,
            timeout=GEMINI_TIMEOUT
        )
    except requests.RequestException as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini API connection failed. "
                f"{str(error)}"
            )
        )

    if response.status_code != 200:
        try:
            error_data = response.json()
            error_message = (
                error_data
                .get("error", {})
                .get("message", "Unknown Gemini API error")
            )
        except Exception:
            error_message = response.text[:1000]

        raise HTTPException(
            status_code=503,
            detail=f"Gemini API error: {error_message}"
        )

    try:
        result = response.json()
        candidates = result.get("candidates", [])

        if not candidates:
            raise ValueError("No candidates returned by Gemini")

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        ai_response = "".join(
            str(part.get("text", ""))
            for part in parts
            if part.get("text")
        ).strip()

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Invalid response received from Gemini API. "
                f"{str(error)}"
            )
        )

    if not ai_response:
        raise HTTPException(
            status_code=503,
            detail="Gemini did not generate a response."
        )

    return ai_response


# ============================================================
# ADD SOROUS FOOTER
# ============================================================

def add_sorous_footer(
    response: str
) -> str:

    clean_response = remove_sorous_footer(
        response
    )

    return (
        f"{clean_response}\n\n"
        f"{SOROUS_FOOTER}"
    )


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(
    data: ChatRequest,
    db: Session = Depends(get_db)
):

    message_text = data.message.strip()

    if not message_text:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    user = get_current_user_from_token(
        data.token,
        db
    )


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    user_message = Message(
        user_id=user.id,
        role="user",
        content=message_text
    )

    db.add(user_message)
    db.commit()
    db.refresh(user_message)


    # ========================================================
    # AUTOMATIC MEMORY
    # ========================================================

    memory_content = extract_memory_request(
        message_text
    )

    if memory_content:

        was_saved = save_automatic_memory(
            user_id=user.id,
            content=memory_content,
            db=db
        )

        if was_saved:

            ai_response = (
                f"Bilkul {user.name}! "
                f"Maine ye yaad rakh liya hai: "
                f"\"{memory_content}\""
            )

        else:

            ai_response = (
                f"{user.name}, ye memory pehle se "
                f"meri yaad me saved hai: "
                f"\"{memory_content}\""
            )

        ai_response = add_sorous_footer(
            ai_response
        )

        assistant_message = Message(
            user_id=user.id,
            role="assistant",
            content=ai_response
        )

        db.add(assistant_message)
        db.commit()

        return {
            "success": True,
            "assistant": "Sorous",
            "response": ai_response,
            "memory_saved": True,
            "used_live_web": False
        }


    # ========================================================
    # LIVE WEB DATA ONLY WHEN NEEDED
    # ========================================================

    web_context = ""

    if needs_live_data(message_text):

        web_context = get_live_context(
            message_text
        )


    # ========================================================
    # GENERATE AI RESPONSE
    # ========================================================

    try:

        ai_response = generate_sorous_response(
            user_id=user.id,
            user_name=user.name,
            db=db,
            web_context=web_context
        )

        ai_response = add_sorous_footer(
            ai_response
        )

    except HTTPException:

        raise

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while "
                "generating the AI response."
            )
        )


    # ========================================================
    # SAVE AI RESPONSE
    # ========================================================

    assistant_message = Message(
        user_id=user.id,
        role="assistant",
        content=ai_response
    )

    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)


    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "success": True,
        "assistant": "Sorous",
        "response": ai_response,
        "used_live_web": bool(web_context)
    }


# ============================================================
# NEW CHAT
# ============================================================

@app.post("/api/new-chat")
def new_chat(
    data: TokenRequest,
    db: Session = Depends(get_db)
):

    user = get_current_user_from_token(
        data.token,
        db
    )

    return {
        "success": True,
        "message": (
            f"New chat started for {user.name}"
        )
    }