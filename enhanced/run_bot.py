"""
Simple launcher script for the Enhanced Gaming Bot
Run this file to start the bot with default settings
"""

from enhanced_bot import SmartEducationalBot
from config import BotConfig

def main():
    print("🎮 Enhanced Gaming Bot Launcher")
    print("=" * 50)
    print("Features:")
    print("  🎯 Smart object detection and prioritization")
    print("  🚶 Right-click walking with advanced search patterns")
    print("  👁️ Real-time vision system showing bot's perspective")
    print("  🧠 Memory system to avoid revisiting areas")
    print("  🐎 Automatic mounting after interactions")
    print("  😴 Realistic idle behaviors")
    print("  📊 Comprehensive statistics tracking")
    print("=" * 50)
    print("\nControls:")
    print("  Q - Quit bot")
    print("  P - Pause for 3 seconds")
    print("  S - Show statistics")
    print("  V - Toggle vision system")
    print("=" * 50)
    
    try:
        # Initialize bot with configuration
        bot = SmartEducationalBot(
            model_path=BotConfig.MODEL_PATH,
            monitor_config=BotConfig.MONITOR_CONFIG
        )
        
        # Start the bot
        bot.run()
        
    except FileNotFoundError:
        print("❌ Error: YOLO model file not found!")
        print(f"Please ensure your model file exists at: {BotConfig.MODEL_PATH}")
        print("Update the MODEL_PATH in config.py if needed.")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
