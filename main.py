"""
FastAPI Application for Voice-to-Action AI Pipeline
Main endpoint: POST /voice-agent/parse
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from nlp.intent_classifier import IntentClassifier
from nlp.entity_extractor import EntityExtractor
from nlp.action_decider import ActionDecider
import os


# Initialize FastAPI app
app = FastAPI(
    title="Voice-to-Action AI Pipeline",
    description="Logistics query processing system",
    version="1.0.0"
)

# Initialize components
print("Initializing Voice Agent...")
intent_classifier = IntentClassifier()
entity_extractor = EntityExtractor()
action_decider = ActionDecider()

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_model.pkl")
try:
    intent_classifier.load(MODEL_PATH)
    print(f"✓ Intent classifier loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load intent model. Train the model first!")
    print(f"  Error: {e}")


# Request/Response models
class QueryRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "pickup karna hai Andheri se Powai, 2 boxes"
            }
        }


class QueryResponse(BaseModel):
    query: str
    intent: Dict
    entities: Dict
    next_action: Dict


# Endpoints
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Voice-to-Action AI Pipeline",
        "version": "1.0.0"
    }


@app.post("/voice-agent/parse", response_model=QueryResponse)
def parse_query(request: QueryRequest):
    """
    Main endpoint: Parse user query and return intent, entities, and next action
    
    Args:
        request: QueryRequest with 'text' field
        
    Returns:
        QueryResponse with intent, entities, and next_action
    """
    try:
        query_text = request.text.strip()
        
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        # Step 1: Intent Classification
        try:
            intent_result = intent_classifier.predict(query_text)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Intent classification failed. Make sure model is trained. Error: {str(e)}"
            )
        
        # Step 2: Entity Extraction
        entities = entity_extractor.extract(query_text)
        
        # Step 3: Action Decision
        next_action = action_decider.decide_action(
            intent=intent_result['intent'],
            entities=entities,
            confidence=intent_result['confidence']
        )
        
        # Return structured response
        return QueryResponse(
            query=query_text,
            intent=intent_result,
            entities=entities,
            next_action=next_action
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/voice-agent/intent-only")
def classify_intent_only(request: QueryRequest):
    """
    Classify intent only (for testing)
    """
    try:
        result = intent_classifier.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice-agent/entities-only")
def extract_entities_only(request: QueryRequest):
    """
    Extract entities only (for testing)
    """
    try:
        entities = entity_extractor.extract(request.text)
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)