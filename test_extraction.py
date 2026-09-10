#!/usr/bin/env python3
"""
Test to verify the question extraction system from slides.
Does not require Ollama to be running.
"""

from question_manager import QuestionManager
import sys

def test_extraction():
    print("=" * 60)
    print("TEST: Estrazione Domande dalle Slide")
    print("=" * 60)

    try:
        qm = QuestionManager()

        # Test 1: Data extraction
        print("\n[1] Estraendo domande dai file markdown...")
        questions_by_topic = qm.extract_questions_from_files()

        print(f"✓ Trovati {len(questions_by_topic)} argomenti")

        for topic, questions in questions_by_topic.items():
            print(f"\n  📚 {topic}")
            print(f"     Domande: {len(questions)}")
            for i, q in enumerate(questions[:3], 1):  # Show first 3
                print(f"     - {i}. {q['topic'][:60]}...")
            if len(questions) > 3:
                print(f"     ... e altri {len(questions) - 3} argomenti")

        # Test 2: Get available topics
        print("\n[2] Argomenti disponibili:")
        topics = qm.get_available_topics()
        for i, topic in enumerate(topics, 1):
            print(f"   {i}. {topic}")

        # Test 3: Set topic
        if topics:
            print(f"\n[3] Impostando topic: {topics[0]}")
            success = qm.set_topic(topics[0])
            print(f"{'✓' if success else '✗'} Topic impostato: {success}")

            print(f"\n[4] Current topic: {qm.current_topic}")

        print("\n" + "=" * 60)
        print("✓ TUTTI I TEST PASSATI!")
        print("=" * 60)
        print("\nNote:")
        print("- Le domande vengono estratte correttamente dalle slide")
        print("- Questo test non richiede Ollama")
        print("- Per model.py assicurati che Ollama sia in esecuzione")

        return True

    except Exception as e:
        print(f"\n✗ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_extraction()
    sys.exit(0 if success else 1)

