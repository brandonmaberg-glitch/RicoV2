import keyboard
from app.io.ptt import record_while_held

def run_loop(cfg, stt, llm, tts, memory):
    print("\nHold SPACE to talk. Press ESC to quit.\n")

    while True:
        if keyboard.is_pressed(cfg.quit_key):
            break

        if keyboard.is_pressed(cfg.ptt_key):
            print("Recording... (release SPACE to stop)")
            audio = record_while_held(cfg.sample_rate, cfg.max_seconds, cfg.ptt_key)
            if audio is None:
                continue

            user_text = stt.transcribe(audio, cfg.sample_rate)
            if not user_text:
                print("Heard nothing.")
                continue

            # Simple commands (no tools yet)
            if user_text.strip().lower() == "reset":
                memory.reset()
                print("RICO: Memory cleared.")
                tts.speak("Certainly, sir. Memory cleared.")
                continue
            if user_text.strip().lower() == "exit":
                break

            print(f"You: {user_text}")
            memory.add_user(user_text)

            rico = llm.chat(memory.get())
            print(f"RICO: {rico}")
            memory.add_assistant(rico)

            tts.speak(rico)

    print("Bye.")
