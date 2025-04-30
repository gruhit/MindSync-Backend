from datetime import datetime, timedelta
import statistics

class SessionManager:
    def __init__(self, supabase_client):
        self.supabase = supabase_client

    async def handle_chat_session(self, user_name: str):
        today = datetime.now().date()
        
        # Get all sessions for today
        sessions = self.supabase.table("chat_sessions")\
            .select("*")\
            .eq("user_id", user_name)\
            .eq("session_date", today)\
            .is_("merged_with_id", "null")\
            .execute()

        if not sessions.data:
            # Create new session
            return await self.create_new_session(user_name)
        
        # If multiple sessions exist, merge them
        if len(sessions.data) > 1:
            return await self.merge_sessions(sessions.data)
        
        return sessions.data[0]["id"]

    async def merge_sessions(self, sessions):
        main_session_id = sessions[0]["id"]
        
        # Calculate combined sentiment
        all_messages = []
        for session in sessions:
            messages = self.supabase.table("chat_messages")\
                .select("sentiment_score")\
                .eq("session_id", session["id"])\
                .execute()
            all_messages.extend(messages.data)

        avg_sentiment = statistics.mean([m["sentiment_score"] for m in all_messages])
        
        # Update main session
        self.supabase.table("chat_sessions")\
            .update({
                "daily_sentiment_score": avg_sentiment,
                "sentiment_status": self.get_sentiment_status(avg_sentiment),
                "session_end": datetime.now()
            })\
            .eq("id", main_session_id)\
            .execute()

        # Mark other sessions as merged
        for session in sessions[1:]:
            self.supabase.table("chat_sessions")\
                .update({"merged_with_id": main_session_id})\
                .eq("id", session["id"])\
                .execute()

        return main_session_id