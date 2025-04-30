import os
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_supabase_connection():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Supabase credentials
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not all([supabase_url, supabase_key]):
            raise ValueError("Missing Supabase credentials in .env file")
            
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Test table connections
        tables_to_test = ["Users", "chat_sessions", "chat_messages", "mental_health_tracking"]
        
        for table in tables_to_test:
            try:
                response = supabase.table(table).select("*").limit(1).execute()
                logger.info(f"✅ Successfully connected to {table} table")
                logger.info(f"Sample data: {response.data}")
            except Exception as table_error:
                logger.error(f"❌ Failed to query {table} table: {str(table_error)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Supabase connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🔄 Testing Supabase connection...")
    success = test_supabase_connection()
    
    if success:
        logger.info("✅ Supabase connection test completed successfully")
    else:
        logger.error("❌ Supabase connection test failed")