# -*- coding: utf-8 -*-

import os
import re
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables from .env file
load_dotenv()

# Typed state structure
class FlashCardState(TypedDict):
    transcript_file_path: Optional[str]
    transcript: Optional[str]
    keypoints: Optional[List[str]]
    flashcards: Optional[List[Dict[str, Any]]]
    flashcard_format: Optional[str]
    messages: List[Dict[str, Any]]
    is_transcribed: Optional[bool]
    is_keypoints_extracted: Optional[bool]
    is_flashcards_generated: Optional[bool]

# Load API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables.")

# LLM model setup
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# Node 1: PDF to transcript
def transcript_from_pdf(state: FlashCardState):
    path = state.get("transcript_file_path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")

    loader = PyPDFLoader(path)
    docs = loader.load()
    transcript = "\n".join(doc.page_content for doc in docs)

    return {
        "transcript": transcript,
        "is_transcribed": True,
        "messages": state.get("messages", []) + [
            {"role": "transcriber", "content": transcript}
        ]
    }

# Node 2: Extract keypoints
def keypoint_extractor(state: FlashCardState):
    transcript = state["transcript"]
    prompt = f"Extract 7-8 keypoints from the following transcript:\n\n{transcript}"
    response = model.invoke([HumanMessage(content=prompt)])
    points = [line.strip("-• ").strip() for line in response.content.split("\n") if line.strip()]

    return {
        "keypoints": points,
        "is_keypoints_extracted": True,
        "messages": state.get("messages", []) + [
            {"role": "key_point_extractor", "content": response.content}
        ]
    }

# Node 3: Generate flashcards
def flashcard_generator(state: FlashCardState):
    keypoints = state["keypoints"]
    prompt = "Convert the following keypoints into Q&A flashcards:\n\n" + "\n".join(keypoints)
    response = model.invoke([HumanMessage(content=prompt)])

    cards = []
    current_q = current_a = None
    lines = response.content.splitlines()

    for line in lines:
        line = line.strip()
        q_match = re.match(r"\*?\s*\*\*Q:\*\*\s*(.+)", line)
        a_match = re.match(r"\*?\s*\*\*A:\*\*\s*(.+)", line)

        if q_match:
            current_q = q_match.group(1).strip()
        elif a_match:
            current_a = a_match.group(1).strip()
            if current_q and current_a:
                cards.append({"question": current_q, "answer": current_a})
                current_q = current_a = None

    return {
        "flashcards": cards,
        "is_flashcards_generated": True,
        "messages": state.get("messages", []) + [
            {"role": "flashcard_generator", "content": response.content}
        ]
    }

# Node 4: Display flashcards
def display_flashcards(state: FlashCardState):
    print("\nFinal Flashcards:\n" + "=" * 50)
    for i, card in enumerate(state["flashcards"], 1):
        print(f"\nCard {i}")
        print(f"Question:\n{card['question']}")
        print(f"Answer:\n{card['answer']}")
        print("-" * 50)
    return {}

# Define the LangGraph workflow
builder = StateGraph(FlashCardState)
builder.add_node("transcript", transcript_from_pdf)
builder.add_node("keypoint_extractor", keypoint_extractor)
builder.add_node("flashcard_generator", flashcard_generator)
builder.add_node("display", display_flashcards)

builder.set_entry_point("transcript")
builder.add_edge("transcript", "keypoint_extractor")
builder.add_edge("keypoint_extractor", "flashcard_generator")
builder.add_edge("flashcard_generator", "display")
builder.add_edge("display", END)

graph = builder.compile()

# Example initial state
state: FlashCardState = {
    "transcript_file_path": "transcript.pdf",  # Replace with your actual PDF path
    "messages": []
}

# Run the graph
graph.invoke(state)
