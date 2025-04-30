import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import google.generativeai as genai
import tensorflow as tf
from textblob import TextBlob
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import random
import json
from config import settings
from utils.sentiment_analyzer import EmotionAnalyzer

# Configure TensorFlow to reduce warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Supabase client
try:
    supabase: Client = create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_key
    )
    if not supabase:
        raise ValueError("Failed to initialize Supabase client")
    logger.info("Initialized Supabase client")
except Exception as e:
    logger.error(f"Supabase initialization error: {str(e)}")
    raise

# Initialize Gemini
genai.configure(api_key=settings.google_api_key)
model = genai.GenerativeModel('gemini-2.0-flash-001')
logger.info("Initialized Gemini model")

# Initialize FastAPI with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        # Cleanup
        pass

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class ChatMessage(BaseModel):
    text: str
    sender: str
    created_at: str

class ChatSessionRequest(BaseModel):
    user_name: str
    session_date: str

class SaveChatRequest(BaseModel):
    user_name: str
    messages: List[dict]

class User(BaseModel):
    user_name: str
    password: str
    Name: str = ""
    Age: int = None
    Gender: str = ""
    Phone_no: int = None
    Email_id: str = ""
    Profession: str = ""
    State: str = ""
    City: str = ""

class LoginData(BaseModel):
    user_name: str
    password: str

class SaveChatData(BaseModel):
    user_name: str
    chat: list  # list of dicts with "sender" and "text"

class ChatRequest(BaseModel):
    user_name: str
    message: str
    session_id: int
    bot_name: str = "YANA"

class LoginRequest(BaseModel):
    user_name: str
    password: str

# Add new Pydantic model for end session request
class EndSessionRequest(BaseModel):
    user_name: str
    session_id: int
    farewell_message: str = "Talk to you later YANA"

# Simplified SentimentAnalyzer class
class SentimentAnalyzer:
    def analyze(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

# Add this new function after the model initialization
async def get_initial_greeting(username: str):
    try:
        # Fetch user details from Supabase
        response = supabase.table("Users").select("Name").eq("user_name", username).execute()
        
        # Get first name from full name or use username as fallback
        first_name = username
        if response.data and len(response.data) > 0 and response.data[0].get("Name"):
            first_name = response.data[0]["Name"].split()[0]  # Get first name only
        
        greeting_prompt = f"""As YANA, create a warm, friendly greeting for {first_name}.
        Requirements:
        1. Use their first name casually and naturally
        2. Keep it brief and conversational
        3. Ask about their day or feelings
        4. Show genuine interest
        5. End with an open-ended question that encourages sharing
        Remember: Be warm and informal, like talking to a friend"""
        
        response = model.generate_content(
            greeting_prompt,
            generation_config={
                "temperature": 0.8,  # Slightly increased for more natural conversation
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 100
            }
        )
        
        return response.text.strip()
    except Exception as e:
        logger.exception("Error generating initial greeting")
        # Fallback greeting using first name
        return f"Hey {first_name}! How are you feeling today?"

@app.post("/signup")
def signup(user: User):
    try:
        # Validate required fields
        if not user.user_name or not user.password:
            raise HTTPException(status_code=400, detail="Username and password are required")

        # Prepare data for insertion
        data = user.dict()
        data['created_at'] = datetime.utcnow().isoformat()

        # Check if user already exists
        existing_user = supabase.table("Users").select("user_name")\
            .eq("user_name", user.user_name)\
            .execute()

        if existing_user.data:
            raise HTTPException(status_code=400, detail="Username already exists")

        # Attempt to insert new user
        try:
            response = supabase.table("Users").insert(data).execute()
            if response.data:
                logger.info(f"Successfully created user: {user.user_name}")
                return {"success": True, "user": user.user_name}
            else:
                raise HTTPException(status_code=500, detail="Failed to create user")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            raise HTTPException(status_code=500, detail="Database connection error")

    except HTTPException as he:
        logger.error(f"HTTP error during signup: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during signup: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.post("/login")
async def login(request: LoginRequest):
    try:
        logger.info(f"Login attempt for user: {request.user_name}")
        
        # Query the Users table
        user = supabase.table("Users").select("*")\
            .eq("user_name", request.user_name)\
            .execute()
        
        logger.info(f"Database response: {user}")

        if not user.data:
            logger.error(f"User not found: {request.user_name}")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        stored_password = user.data[0]["password"]
        
        if stored_password == request.password:
            logger.info(f"Successful login for user: {request.user_name}")
            return {"status": "success", "user": request.user_name}
        else:
            logger.error(f"Invalid password for user: {request.user_name}")
            raise HTTPException(status_code=401, detail="Invalid credentials")

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_session")
async def create_chat_session(request: ChatSessionRequest):
    try:
        current_time = datetime.now()
        
        # Create new session
        session_data = {
            "user_id": request.user_name,
            "session_date": current_time.date().isoformat(),
            "session_start": current_time.isoformat(),
            "daily_sentiment_score": 0.0,
            "sentiment_status": "neutral"
        }
        
        new_session = supabase.table("chat_sessions").insert(session_data).execute()
        
        if not new_session.data:
            raise HTTPException(status_code=500, detail="Failed to create chat session")

        # Send welcome message
        greeting = await get_initial_greeting(request.user_name)
        
        # Save welcome message
        welcome_message = {
            "session_id": new_session.data[0]["id"],
            "message_text": greeting,
            "sender": "YANA",
            "sentiment_score": 0.0,
            "emotion_details": json.dumps({}),
            "created_at": current_time.isoformat()
        }
        
        supabase.table("chat_messages").insert(welcome_message).execute()

        return {
            "session_id": new_session.data[0]["id"],
            "greeting": greeting
        }
        
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_chat")
async def save_chat(request: SaveChatRequest):
    try:
        logger.info(f"Saving chat for user: {request.user_name}")
        
        current_date = datetime.now().date()
        current_time = datetime.now()
        
        # Create a new session first
        session_data = {
            "user_id": request.user_name,
            "session_date": current_date.isoformat(),  # Convert date to string
            "session_start": current_time.isoformat(),  # Convert datetime to string
            "daily_sentiment_score": 0.0,
            "sentiment_status": "neutral"
        }
        
        session_response = supabase.table("chat_sessions").insert(session_data).execute()
        
        if not session_response.data:
            raise HTTPException(status_code=500, detail="Failed to create chat session")
        
        session_id = session_response.data[0]['id']
        total_sentiment = 0.0
        message_count = 0

        # Process and save each message
        for msg in request.messages:
            sentiment_score = 0.0
            emotion_details = {}
            
            if msg["sender"] == "user":
                blob = TextBlob(msg["text"])
                sentiment_score = float(blob.sentiment.polarity)  # Convert to float
                emotion_details = {
                    "polarity": float(blob.sentiment.polarity),  # Convert to float
                    "subjectivity": float(blob.sentiment.subjectivity)  # Convert to float
                }
                total_sentiment += sentiment_score
                message_count += 1

            # Save message with proper datetime formatting
            message_data = {
                "session_id": session_id,
                "message_text": msg["text"],
                "sender": msg["sender"],
                "sentiment_score": sentiment_score,
                "emotion_details": json.dumps(emotion_details),  # Convert dict to JSON string
                "created_at": current_time.isoformat()  # Convert datetime to string
            }
            
            message_response = supabase.table("chat_messages").insert(message_data).execute()
            
            if not message_response.data:
                logger.error("Failed to save message")
                continue

        # Update session with average sentiment
        if message_count > 0:
            avg_sentiment = total_sentiment / message_count
            sentiment_status = get_sentiment_status(avg_sentiment)
            
            supabase.table("chat_sessions").update({
                "daily_sentiment_score": float(avg_sentiment),  # Convert to float
                "sentiment_status": sentiment_status,
                "session_end": current_time.isoformat()  # Convert datetime to string
            }).eq("id", session_id).execute()

            # Update mental health tracking
            await update_mental_health_tracking(request.user_name, float(avg_sentiment))

        return {
            "status": "success",
            "message": "Chat saved successfully",
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Error saving chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end_session")
async def end_session(request: EndSessionRequest):
    try:
        # Update session end time
        current_time = datetime.now()
        
        # Get all messages from this session for final sentiment calculation
        messages = supabase.table("chat_messages")\
            .select("*")\
            .eq("session_id", request.session_id)\
            .execute()
        
        # Calculate final sentiment score
        user_messages = [msg for msg in messages.data if msg["sender"] == "user"]
        if user_messages:
            total_sentiment = sum(msg["sentiment_score"] for msg in user_messages)
            avg_sentiment = total_sentiment / len(user_messages)
            sentiment_status = get_sentiment_status(avg_sentiment)
        else:
            avg_sentiment = 0.0
            sentiment_status = "neutral"

        # Update session
        session_update = supabase.table("chat_sessions").update({
            "session_end": current_time.isoformat(),
            "daily_sentiment_score": float(avg_sentiment),
            "sentiment_status": sentiment_status
        }).eq("id", request.session_id).execute()

        # Save farewell message
        farewell_data = {
            "session_id": request.session_id,
            "message_text": request.farewell_message,
            "sender": "user",
            "sentiment_score": 0.0,
            "emotion_details": json.dumps({}),
            "created_at": current_time.isoformat()
        }
        supabase.table("chat_messages").insert(farewell_data).execute()

        # Update mental health tracking
        await update_mental_health_tracking(request.user_name, float(avg_sentiment))

        return {
            "status": "success",
            "message": "Session ended successfully",
            "final_sentiment": sentiment_status
        }

    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_sentiment_status(score: float) -> str:
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    return "neutral"

async def update_mental_health_tracking(user_name: str, sentiment_score: float):
    try:
        today = datetime.now().date().isoformat()  # Convert date to string immediately
        
        tracking = supabase.table("mental_health_tracking")\
            .select("*")\
            .eq("user_id", user_name)\
            .eq("tracking_date", today)\
            .execute()

        negative_threshold = -0.2
        consultation_threshold = 5

        if tracking.data:
            current_tracking = tracking.data[0]
            consecutive_days = current_tracking["consecutive_negative_days"]
            
            if sentiment_score < negative_threshold:
                consecutive_days += 1
            else:
                consecutive_days = 0

            needs_consultation = consecutive_days >= consultation_threshold

            supabase.table("mental_health_tracking").update({
                "avg_sentiment_score": float(sentiment_score),
                "consecutive_negative_days": consecutive_days,
                "needs_consultation": needs_consultation,
                "updated_at": datetime.now().isoformat()  # Convert datetime to string
            }).eq("id", current_tracking["id"]).execute()
        else:
            supabase.table("mental_health_tracking").insert({
                "user_id": user_name,
                "tracking_date": today,  # Already converted to string
                "avg_sentiment_score": float(sentiment_score),
                "consecutive_negative_days": 1 if sentiment_score < negative_threshold else 0,
                "needs_consultation": False,
                "created_at": datetime.now().isoformat(),  # Convert datetime to string
                "updated_at": datetime.now().isoformat()  # Convert datetime to string
            }).execute()

    except Exception as e:
        logger.error(f"Error in update_mental_health_tracking: {str(e)}")
        # Don't raise the exception to prevent breaking the save_chat flow
        pass

sentiment_analyzer = SentimentAnalyzer()
emotion_analyzer = EmotionAnalyzer()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        current_time = datetime.now()
        
        # Analyze the message
        analysis = emotion_analyzer.analyze_text(request.message)
        
        # Save user message with enhanced analysis
        user_message = {
            "session_id": request.session_id,
            "message_text": request.message,
            "sender": "user",
            "sentiment_score": analysis["sentiment_score"],
            "emotion_details": json.dumps(analysis["emotion_details"]),
            "activities": json.dumps(analysis["activities"]),
            "created_at": current_time.isoformat()
        }
        
        supabase.table("chat_messages").insert(user_message).execute()
        
        # Generate and analyze bot response
        user_data = supabase.table("Users").select("Name")\
            .eq("user_name", request.user_name)\
            .execute()

        first_name = request.user_name
        if user_data.data and user_data.data[0].get("Name"):
            full_name = user_data.data[0]["Name"]
            first_name = full_name.split()[0]

        prompt = f"You are YANA, a friendly mental health assistant. Respond to {first_name}:\n{request.message}"
        model_response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 150
            }
        )

        if hasattr(model_response, "text"):
            reply = model_response.text.strip()
        else:
            reply = "I'm here for you. Could you tell me a bit more?"

        # Save bot response with neutral emotion
        bot_message = {
            "session_id": request.session_id,
            "message_text": reply,
            "sender": "YANA",
            "sentiment_score": 0.0,
            "emotion_details": json.dumps({
                "primary_emotion": "neutral",
                "intensity": 0.0,
                "secondary_emotions": {},
                "confidence": 1.0,
                "mood_indicators": ["neutral"]
            }),
            "activities": json.dumps([]),
            "created_at": current_time.isoformat()
        }
        
        supabase.table("chat_messages").insert(bot_message).execute()
        
        return {"reply": reply}
        
    except Exception as e:
        logger.exception("Error during chat processing")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{username}")
async def get_user(username: str):
    try:
        response = supabase.table("Users")\
            .select("Name")\
            .eq("user_name", username)\
            .execute()
        
        if response.data and len(response.data) > 0:
            return {"Name": response.data[0]["Name"]}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    
    except Exception as e:
        logger.error(f"Error fetching user name: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/{username}")
async def get_user_analytics(username: str):
    try:
        # Get current date and week ago date
        current_date = datetime.now()
        week_ago = current_date - timedelta(days=7)

        # Get all chat sessions for the user
        sessions = supabase.table("chat_sessions")\
            .select("*")\
            .eq("user_id", username)\
            .gte("created_at", week_ago.isoformat())\
            .execute()

        # Get all messages
        messages = supabase.table("chat_messages")\
            .select("*")\
            .in_("session_id", [s["id"] for s in sessions.data])\
            .execute()

        # Process data for analytics
        weekly_data = []
        activities = {}
        today_emotions = {
            "joy": 0, "contentment": 0, "anxiety": 0, 
            "sadness": 0, "anger": 0
        }

        # Process messages for analytics
        for message in messages.data:
            # Count activities
            if message.get("activities"):
                for activity in message["activities"]:
                    activities[activity] = activities.get(activity, 0) + 1

            # Process emotions for today
            if message.get("emotion_details") and \
               message["created_at"].date() == current_date.date():
                emotions = message["emotion_details"].get("secondary_emotions", {})
                for emotion, score in emotions.items():
                    if emotion in today_emotions:
                        today_emotions[emotion] = max(
                            today_emotions[emotion], 
                            score
                        )

        # Calculate daily averages for the week
        for i in range(7):
            date = current_date - timedelta(days=i)
            day_messages = [
                m for m in messages.data 
                if m["created_at"].date() == date.date()
            ]
            
            if day_messages:
                avg_sentiment = sum(m["sentiment_score"] for m in day_messages) / len(day_messages)
            else:
                avg_sentiment = 0

            weekly_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "averageSentiment": avg_sentiment
            })

        # Get today's activities with sentiment
        today_activities = []
        today_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        for message in messages.data:
            message_date = datetime.fromisoformat(message["created_at"].replace('Z', '+00:00'))
            
            if message_date >= today_start and message.get("activities"):
                for activity in message["activities"]:
                    today_activities.append({
                        "name": activity,
                        "sentiment": message["sentiment_score"],
                        "timestamp": message["created_at"]
                    })

        # Calculate overall statistics
        total_messages = len(messages.data)
        average_sentiment = sum(m["sentiment_score"] for m in messages.data) / total_messages if total_messages > 0 else 0
        most_common_activity = max(activities.items(), key=lambda x: x[1])[0] if activities else "No activities"

        # Get messages with activities
        messages_with_activities = [
            message for message in messages.data 
            if message.get("activities") and message["activities"] != "[]"
        ]

        return {
            "weeklyData": weekly_data,
            "activityBreakdown": activities,
            "todayEmotions": list(today_emotions.values()),
            "averageSentiment": average_sentiment,
            "totalSessions": len(sessions.data),
            "mostCommonActivity": most_common_activity,
            "activities": {
                "breakdown": activities,
                "today": sorted(today_activities, key=lambda x: x["timestamp"], reverse=True)
            },
            "messages": messages_with_activities,
        }

    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/today-emotions/{username}")
async def get_today_emotions(username: str):
    try:
        today = datetime.now().date()
        current_time = datetime.now()
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get all user messages from today
        messages = supabase.table("chat_messages")\
            .select("*")\
            .gte("created_at", today.isoformat())\
            .lt("created_at", (today + timedelta(days=1)).isoformat())\
            .eq("sender", "user")\
            .execute()

        # Initialize default emotion levels
        emotion_levels = {
            "joy": 0.0,
            "contentment": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0
        }

        # Process messages if they exist
        if messages.data:
            for msg in messages.data:
                if msg.get("emotion_details"):
                    details = msg["emotion_details"]
                    if isinstance(details, str):
                        details = json.loads(details)
                    
                    if details and "secondary_emotions" in details:
                        for emotion, level in details["secondary_emotions"].items():
                            emotion_lower = emotion.lower()
                            if emotion_lower in emotion_levels:
                                emotion_levels[emotion_lower] = max(
                                    emotion_levels[emotion_lower],
                                    float(level)
                                )

        # Process mood timeline
        mood_timeline = []
        if messages.data:
            for msg in messages.data:
                if msg["sender"] == "user":
                    mood_timeline.append({
                        "timestamp": msg["created_at"],
                        "sentiment_score": msg["sentiment_score"],
                        "emotion_details": json.loads(msg["emotion_details"]) if msg["emotion_details"] else {}
                    })

            # Sort by timestamp
            mood_timeline.sort(key=lambda x: x["timestamp"])

        return {
            "emotionLevels": list(emotion_levels.values()),
            "hasData": len(messages.data) > 0,
            "timeRange": "today" if current_time >= today_start else "last24h",
            "moodTimeline": mood_timeline
        }

    except Exception as e:
        logger.error(f"Error fetching emotions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-sessions/{username}")
async def get_user_sessions(username: str):
    try:
        today = datetime.now().date()
        
        # Get all sessions for the user from today
        sessions = supabase.table("chat_sessions")\
            .select("*")\
            .eq("user_id", username)\
            .eq("session_date", today.isoformat())\
            .execute()

        # Format session data
        formatted_sessions = []
        for session in sessions.data:
            formatted_sessions.append({
                "timestamp": session["created_at"],
                "sentiment_score": session["daily_sentiment_score"],
                "sentiment_status": session["sentiment_status"],
                "start_time": session["session_start"],
                "end_time": session["session_end"]
            })

        # Sort by timestamp
        formatted_sessions.sort(key=lambda x: x["timestamp"])

        return {
            "sessions": formatted_sessions
        }

    except Exception as e:
        logger.error(f"Error fetching user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to YANA Mental Health Assistant API"}

# --- Run the App ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

