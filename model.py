import sys
import logging
from typing import Optional
from question_manager import QuestionManager

# Logging configuration to avoid cluttering the command line interface
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIApp:
    def __init__(self) -> None:
        self.qm = QuestionManager()
        self.available_topics = self.qm.get_available_topics()
        self.running = True

    def print_topics(self) -> None:
        print("\nArgomenti disponibili:")
        for i, topic in enumerate(self.available_topics, 1):
            print(f"  {i}. {topic}")
        print("\nComandi utili: /esci, /argomenti, /cambio")

    def select_topic(self) -> bool:
        self.print_topics()
        while True:
            user_input = input("\nScegli un argomento (numero o nome) o un comando: ").strip()
            
            if self.handle_commands(user_input):
                if user_input.lower() == '/esci':
                    return False
                continue

            try:
                choice = int(user_input) - 1
                if 0 <= choice < len(self.available_topics):
                    topic = self.available_topics[choice]
                else:
                    print("❌ Scelta numerica non valida.")
                    continue
            except ValueError:
                matching = [t for t in self.available_topics if user_input.lower() in t.lower()]
                if matching:
                    topic = matching[0]
                else:
                    print("❌ Argomento non trovato.")
                    continue

            if self.qm.set_topic(topic):
                print(f"\n✅ Argomento selezionato: {topic}")
                return True
            else:
                print("❌ Errore durante l'impostazione dell'argomento.")

    def handle_commands(self, user_input: str) -> bool:
        if not user_input.startswith('/'):
            return False
            
        cmd = user_input.lower()
        if cmd == '/esci':
            print("Esame finito. Ciao e buono studio! 👋")
            self.running = False
            self.qm.current_topic = None
            return True
        elif cmd in ('/argomenti', '/cambio'):
            self.qm.current_topic = None
            self.print_topics()
            return True
        else:
            print("❌ Comando non valido.")
            return True

    def ask_question(self) -> bool:
        print("\n" + "-"*40)
        print("Generazione prossima domanda in corso...")
        try:
            question = self.qm.get_next_question()
            if not question:
                print("Impossibile recuperare ulteriori domande per questo argomento.")
                return False
            print("\n" + question)
        except Exception as e:
            logger.error(f"Unhandled error during generation: {e}")
            print("\n⚠️ Si è verificato un errore inaspettato durante la generazione della domanda.")
            return False
        
        while True:
            user_input = input("\nRisposta (A/B/C) o /comando: ").strip()
            
            if self.handle_commands(user_input):
                if user_input.lower() in ('/esci', '/cambio', '/argomenti'):
                    return False
                continue

            if user_input.lower() in ['a', 'b', 'c']:
                is_correct, feedback = self.qm.parse_answer(user_input)
                if is_correct:
                    print(f"\n✅ Corretto! {feedback}")
                else:
                    print(f"\n❌ Sbagliato! {feedback}")
                
                # Exit the response loop to generate a new question
                return True
            else:
                print("❌ Rispondi con A, B o C. Oppure usa /cambio per cambiare argomento.")

    def run(self) -> None:
        print("🎓 Caricamento RAG Diritto dell'Informatica...")
        if not self.available_topics:
            print("Nessun argomento trovato. Assicurati di avere le slide in formato markdown e prova ad avviare main.py")
            sys.exit(1)

        while self.running:
            if not self.qm.current_topic:
                if not self.select_topic():
                    break # /esci pressed
            
            # Loop for questions on the current topic
            while self.qm.current_topic and self.running:
                if not self.ask_question():
                    # Can return False if /cambio or /esci was used, or if there's an error
                    self.qm.current_topic = None
                    break # Break out to re-select the topic or exit

def start_questioning() -> None:
    app = CLIApp()
    app.run()

if __name__ == "__main__":
    start_questioning()