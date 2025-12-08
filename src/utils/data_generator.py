import json
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

class QuoteGenerator:
    def __init__(self):
        self.templates = [
            "{quote}",
            "\"{quote}\"",
            "\"{quote}\" - {author}",
            "{author} once said: \"{quote}\"",
            "As {author} famously said: \"{quote}\""
        ]
        
        self.topics = [
            "Leadership", "Courage", "Wisdom", "Innovation", 
            "Perseverance", "Hope", "Freedom", "Equality",
            "Science", "Art", "Love", "Friendship", "Success"
        ]
        
        self.eras = ["Ancient", "Medieval", "Renaissance", "Enlightenment", 
                    "19th Century", "20th Century", "21st Century"]
    
    def generate_sample_quotes(self, count=20):
        base_quotes = [
            {
                "quote": "The unexamined life is not worth living.",
                "author": "Socrates",
                "era": "Ancient",
                "topic": "Philosophy, Self-reflection"
            },
            {
                "quote": "To be, or not to be: that is the question.",
                "author": "William Shakespeare",
                "era": "Renaissance",
                "topic": "Existence, Philosophy"
            },
            {
                "quote": "Cogito, ergo sum. (I think, therefore I am.)",
                "author": "René Descartes",
                "era": "Enlightenment",
                "topic": "Philosophy, Existence"
            },
            {
                "quote": "Eureka!",
                "author": "Archimedes",
                "era": "Ancient",
                "topic": "Discovery, Science"
            },
            {
                "quote": "Veni, vidi, vici. (I came, I saw, I conquered.)",
                "author": "Julius Caesar",
                "era": "Ancient",
                "topic": "Conquest, Achievement"
            },
            {
                "quote": "Knowledge is power.",
                "author": "Francis Bacon",
                "era": "Renaissance",
                "topic": "Education, Power"
            },
            {
                "quote": "An eye for an eye makes the whole world blind.",
                "author": "Mahatma Gandhi",
                "era": "20th Century",
                "topic": "Nonviolence, Justice"
            },
            {
                "quote": "The only true wisdom is in knowing you know nothing.",
                "author": "Socrates",
                "era": "Ancient",
                "topic": "Wisdom, Humility"
            },
            {
                "quote": "Life is what happens to you while you're busy making other plans.",
                "author": "John Lennon",
                "era": "20th Century",
                "topic": "Life, Perspective"
            },
            {
                "quote": "In the middle of difficulty lies opportunity.",
                "author": "Albert Einstein",
                "era": "20th Century",
                "topic": "Opportunity, Challenge"
            }
        ]
        
        generated = base_quotes[:min(count, len(base_quotes))]
        
        for i, quote in enumerate(generated):
            quote["id"] = i + 1
            quote["context"] = f"Famous quote by {quote['author']} from the {quote['era']} era."
            quote["source"] = "Various historical sources"
            quote["tags"] = [tag.strip() for tag in quote["topic"].split(",")]
            quote["language"] = "English"
        
        return generated
    
    def save_to_file(self, quotes, file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(quotes, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(quotes)} quotes to {file_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save quotes: {e}")
            return False
    
    def load_and_expand(self, input_file, output_file, target_count=100):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                existing_quotes = json.load(f)
            
            print(f"📊 Loaded {len(existing_quotes)} existing quotes")
            
            if len(existing_quotes) >= target_count:
                print(f"✅ Already have {len(existing_quotes)} quotes (target: {target_count})")
                return existing_quotes
            
            needed = target_count - len(existing_quotes)
            print(f"🔄 Generating {needed} additional quotes...")
            
            additional = self.generate_sample_quotes(needed)
            
            start_id = max([q.get("id", 0) for q in existing_quotes]) + 1
            for i, quote in enumerate(additional):
                quote["id"] = start_id + i
            
            all_quotes = existing_quotes + additional
            
            self.save_to_file(all_quotes, output_file)
            
            return all_quotes
            
        except Exception as e:
            print(f"❌ Failed to expand quotes: {e}")
            return []

if __name__ == "__main__":
    generator = QuoteGenerator()
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    
    quotes = generator.generate_sample_quotes(50)
    
    output_file = data_dir / "expanded_quotes.json"
    generator.save_to_file(quotes, output_file)
    
    print(f"\n✅ Generated {len(quotes)} quotes")
    print(f"📁 Saved to: {output_file}")
    
    print("\n📝 Sample quotes:")
    for i, quote in enumerate(quotes[:3]):
        print(f"{i+1}. \"{quote['quote'][:50]}...\" - {quote['author']}")
