import glob
import os
import re
import random
import logging
import json
from typing import Dict, List, Optional, Tuple, Any

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Logging configuration
logger = logging.getLogger(__name__)

class QuestionOutput(BaseModel):
    domanda: str = Field(description="Il testo della domanda a scelta multipla")
    opzione_A: str = Field(description="Testo per l'opzione A")
    opzione_B: str = Field(description="Testo per l'opzione B")
    opzione_C: str = Field(description="Testo per l'opzione C")
    risposta_corretta: str = Field(description="Lettera della risposta corretta: A, B o C")
    spiegazione: str = Field(description="Breve spiegazione del perché la risposta è corretta basata sul contesto")

class QuestionManager:
    def __init__(self) -> None:
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.db = Chroma(persist_directory="./chroma_db_dir", embedding_function=self.embeddings)
        self.llm: Optional[ChatOllama] = None
        self.current_topic: Optional[str] = None
        self.current_question: Optional[Dict[str, str]] = None
        self.questions_history: List[Dict[str, str]] = []
        self.cached_questions: Dict[str, Dict[str, str]] = {}
        self.questions_by_topic: Optional[Dict[str, List[Dict[str, str]]]] = None
        self.parser = JsonOutputParser(pydantic_object=QuestionOutput)

    def _init_llm(self) -> None:
        if self.llm is None:
            # Initialize the LLM, forcing JSON format where natively supported
            # Temperature at 0.0 for maximum context adherence and determinism
            self.llm = ChatOllama(model="gemma3:4b", temperature=0.0, format="json")

    def extract_questions_from_files(self) -> Dict[str, List[Dict[str, str]]]:
        if self.questions_by_topic is not None:
            return self.questions_by_topic

        questions_by_topic: Dict[str, List[Dict[str, str]]] = {}
        md_files = glob.glob(os.path.join("./md_files", "*.md"))

        for file_path in sorted(md_files):
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # The regex tolerates the presence or absence of asterisks (*) and reads the entire title up to the newline
                # Captures titles like "Attenzione ripasso", "Ripasso", "Cose da sapere ASSOLUTAMENTE"
                pattern = r"#{1,4}\s*(?:Attenzione\s*)?(?:Ripasso|Cose da sapere)[^\n]*\n+(.*?)(?=\n#{1,3}|$)"
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)

                topics_in_file: List[Dict[str, str]] = []
                for match in matches:
                    section_text = match.group(1).strip()

                    bullet_patterns = [
                        r"●\s*([^\n●]+)",
                        r"^[\d+\-\*]\s+([^\n]+)",
                    ]

                    all_topics: List[str] = []
                    for bp in bullet_patterns:
                        bullets = re.finditer(bp, section_text, re.MULTILINE)
                        for bullet in bullets:
                            topic = bullet.group(1).strip().strip("[]()").strip()
                            if topic and len(topic.split()) > 1:
                                all_topics.append(topic)

                    for topic in all_topics:
                        if topic not in [t["topic"] for t in topics_in_file]:
                            topics_in_file.append({
                                "topic": topic,
                                "source_file": filename,
                            })

                if topics_in_file:
                    category = re.sub(r'^\d+_', '', filename)
                    category = category.replace("_DEF.md", "").replace(".md", "")
                    questions_by_topic[category] = topics_in_file
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
                continue

        self.questions_by_topic = questions_by_topic
        return questions_by_topic

    def get_available_topics(self) -> List[str]:
        questions = self.extract_questions_from_files()
        return list(questions.keys())

    def generate_question_with_options(self, topic_text: str, source_file: str) -> Dict[str, str]:
        cache_key = f"{source_file}|{topic_text[:50]}"
        if cache_key in self.cached_questions:
            return self.cached_questions[cache_key]

        self._init_llm()

        # Retrieves the actual text from the slides via Chroma, filtering ONLY the current module
        try:
            docs = self.db.similarity_search(topic_text, k=2, filter={"origin": source_file})
            context = "\n".join(doc.page_content for doc in docs) if docs else "Nessun contesto disponibile."
        except Exception as e:
            logger.error(f"Error searching the vector DB: {e}")
            context = "Nessun contesto disponibile per problemi di ricerca."

        prompt_template = """Contesto dalle slide:
{context}

Argomento di ripasso: {topic}

Basandoti unicamente sul contesto fornito, crea UNA SOLA domanda a scelta multipla (3 opzioni A, B, C). NON richiedere numeri di leggi specifici.

Regole IMPORTANTI:
1. NON creare domande la cui risposta è contenuta nel testo della domanda stessa (no tautologie).
2. Le opzioni sbagliate (distrattori) devono essere plausibili ma inequivocabilmente false.
3. Evita di usare logica ingannevole (es. aggiungendo "Solo se..." a concetti che sono veri anche in altri casi).
4. La domanda deve testare la comprensione del concetto.

Restituisci SOLO un oggetto JSON valido. Usa ESATTAMENTE questo formato:
{{
  "domanda": "Il testo della tua domanda",
  "opzione_A": "La prima opzione",
  "opzione_B": "La seconda opzione",
  "opzione_C": "La terza opzione",
  "risposta_corretta": "A",
  "spiegazione": "Breve spiegazione della risposta esatta"
}}
"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Handle the potential instability of the local gemma JSON parser by forcing the chain
        chain = prompt | self.llm | self.parser

        try:
            result = chain.invoke({
                "topic": topic_text[:100], 
                "context": context
            })
            
            # Normalize the keys to tolerate imprecise formatting from the local LLM
            normalized_result = {
                "domanda": result.get("domanda", result.get("Domanda", result.get("question", "Domanda non generata correttamente"))),
                "opzione_A": result.get("opzione_A", result.get("Opzione_A", result.get("A", "Opzione mancante"))),
                "opzione_B": result.get("opzione_B", result.get("Opzione_B", result.get("B", "Opzione mancante"))),
                "opzione_C": result.get("opzione_C", result.get("Opzione_C", result.get("C", "Opzione mancante"))),
                "risposta_corretta": result.get("risposta_corretta", result.get("Risposta_corretta", result.get("Risposta_Corretta", "B"))),
                "spiegazione": result.get("spiegazione", result.get("Spiegazione", "Spiegazione mancante"))
            }
            
            self.cached_questions[cache_key] = normalized_result
            return normalized_result
        except Exception as e:
            logger.error(f"Error during LLM generation for '{topic_text}': {e}")
            # Smart fallback instead of pure hardcoded values
            fallback_dict = {
                "domanda": f"Si è verificato un errore generando la domanda. L'argomento era: {topic_text[:30]}...",
                "opzione_A": "Errore",
                "opzione_B": "Scegli questa per andare avanti (Risposta Corretta di Fallback)",
                "opzione_C": "Errore",
                "risposta_corretta": "B",
                "spiegazione": f"Il RAG ha fallito per via di un errore di generazione ({str(e)}). Assicurati che ollama stia girando e il modello sia installato."
            }
            return fallback_dict

    def set_topic(self, topic_name: str) -> bool:
        questions = self.extract_questions_from_files()
        if topic_name in questions:
            self.current_topic = topic_name
            self.questions_history = []
            return True
        return False

    def get_next_question(self) -> Optional[str]:
        if not self.current_topic:
            return None

        questions = self.extract_questions_from_files()
        topics_list = questions.get(self.current_topic, [])

        if not topics_list:
            return None

        # Track ALL topics already asked for this subject to avoid repetitions
        asked_topics = [q["topic"] for q in self.questions_history]
        available_topics = [t for t in topics_list if t["topic"] not in asked_topics]

        if not available_topics:
            # If we have exhausted all topics, reset the history and start over
            self.questions_history = []
            available_topics = topics_list

        selected = random.choice(available_topics)

        logger.info(f"Generating question for topic: {selected['topic']}")
        question_data = self.generate_question_with_options(
            selected["topic"],
            selected["source_file"]
        )

        self.current_question = {
            "testo_domanda": question_data["domanda"],
            "opzione_A": question_data["opzione_A"],
            "opzione_B": question_data["opzione_B"],
            "opzione_C": question_data["opzione_C"],
            "risposta_corretta": question_data["risposta_corretta"],
            "spiegazione": question_data["spiegazione"],
            "topic": selected["topic"],
            "source": selected["source_file"],
        }

        self.questions_history.append(selected)

        # Build the user-visible string
        visible_text = (
            f"DOMANDA: {self.current_question['testo_domanda']}\n"
            f"A) {self.current_question['opzione_A']}\n"
            f"B) {self.current_question['opzione_B']}\n"
            f"C) {self.current_question['opzione_C']}\n"
        )
        return visible_text

    def parse_answer(self, answer_text: str) -> Tuple[bool, str]:
        if not self.current_question:
            return False, "Nessuna domanda in corso."

        answer = answer_text.strip().lower()
        if answer not in ['a', 'b', 'c']:
            return False, "Risposta non valida. Scegli tra A, B o C."

        correct_answer = self.current_question["risposta_corretta"].strip().lower()
        
        # Sometimes the LLM returns "Opzione B" or "B)". Extract only the main letter.
        if "a" in correct_answer: correct_answer = "a"
        elif "b" in correct_answer: correct_answer = "b"
        elif "c" in correct_answer: correct_answer = "c"

        is_correct = answer == correct_answer
        explanation = self.current_question["spiegazione"]

        return is_correct, f"La risposta corretta era la {correct_answer.upper()}.\n📌 Dalle slide: {explanation}"
