import keyboard

from app.io.ptt import record_while_held


def run_loop(cfg, stt, llm, tts, memory_service):
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

            command = user_text.strip().lower()
            if command == "reset":
                memory_service.reset()
                print("RICO: Memory cleared.")
                tts.speak("Certainly, sir. Memory cleared.")
                continue
            if command == "exit":
                break

            print(f"You: {user_text}")
            memory_service.ingest_user_message(user_text)

            context_block = memory_service.build_context_for_prompt(user_text)
            messages = [
                {"role": "system", "content": cfg.system_prompt},
                {
                    "role": "system",
                    "content": (
                        "Use this local memory context when helpful. "
                        "Do not mention internal IDs.\n\n" + context_block
                    ),
                },
                {"role": "user", "content": user_text},
            ]
            rico = llm.chat(messages)
            print(f"RICO: {rico}")

            memory_service.ingest_assistant_message(rico)
            tts.speak(rico)

    print("Bye.")
