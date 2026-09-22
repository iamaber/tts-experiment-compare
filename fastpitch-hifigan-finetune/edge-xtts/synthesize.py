#!/usr/bin/env python3
import argparse
import asyncio
import edge_tts


async def synthesize(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def main():
    parser = argparse.ArgumentParser(description="High-Quality TTS with Edge-TTS")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.mp3", help="Output file")
    parser.add_argument(
        "--voice",
        type=str,
        default="en-US-AriaNeural",
        help="Voice name (use --list to see all)",
    )
    parser.add_argument("--list", action="store_true", help="List available voices")

    args = parser.parse_args()

    if args.list:
        print("\n🎤 Popular English Voices:\n")
        voices = [
            ("en-US-AriaNeural", "Female, friendly"),
            ("en-US-GuyNeural", "Male, professional"),
            ("en-US-JennyNeural", "Female, warm"),
            ("en-US-TonyNeural", "Male, energetic"),
            ("en-GB-SoniaNeural", "Female, British"),
            ("en-GB-RyanNeural", "Male, British"),
            ("en-AU-NatashaNeural", "Female, Australian"),
            ("en-AU-WilliamNeural", "Male, Australian"),
        ]
        for voice, desc in voices:
            print(f"  {voice:25} - {desc}")
        print("\n💡 Use: --voice <name>")
        return

    print(f"🎤 Synthesizing with {args.voice}...")
    print(f"   Text: '{args.text}'")

    asyncio.run(synthesize(args.text, args.voice, args.output))

    print(f"💾 Saved: {args.output}")
    print(f"\n✅ Done! Natural, fast, high-quality speech!")


if __name__ == "__main__":
    main()
