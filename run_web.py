import uvicorn
import os

if __name__ == "__main__":
    print("Starting LexAI Web Server...")
    # Ensure we are in the right directory or use absolute paths if needed
    uvicorn.run("ui.app:app", host="0.0.0.0", port=8000, reload=True)
