# RAG IT Law (Diritto dell'Informatica)

A university project based on RAG (Retrieval-Augmented Generation) designed to test your knowledge with multiple-choice questions (A/B/C) on the IT Law syllabus. It extracts information directly from the course slides.

## Features

- Automatically extracts topics from the "Review" (Ripasso) sections of `.md` files.
- Generates multiple-choice questions without requiring you to remember exact law numbers.
- The first questions for each topic are generated on the fly (which takes a few seconds), then they are cached for lightning-fast subsequent retrievals.

## How to Use

1. Start your local Ollama instance in the background with the required models:
   `ollama serve` (make sure you have pulled `gemma3:4b` and `nomic-embed-text`)
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example` and insert the path to your PDF folder.
4. Start the quiz simulator:
   ```bash
   python3 model.py
   ```
5. Follow the interactive menu:
   - Choose a topic (by number or name).
   - Answer with A, B, or C.
   - Use special commands (starting with `/`):
     - `/esci` ➔ exit the system.
     - `/cambio` ➔ change the topic during the session.
     - `/argomenti` ➔ show the topics menu again.

## Project Files

- `main.py`: script to initialize and build the Chroma vector database.
- `model.py`: main loop for the CLI quiz simulator.
- `question_manager.py`: core logic for topic extraction, answer validation, and RAG interaction with the LLM.
- `test_extraction.py`: utility script to verify that question extraction from the slides works properly.

Happy studying!
